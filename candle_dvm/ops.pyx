# cython: language_level=3
"""DVM NDObject graph ops -- phase-1 subset.

Implements the NDObject hierarchy used to build a computation graph and
emit DVM bytecode instructions.  Phase-1 scope covers:

- NDLoad   -- load from global memory
- NDStore  -- store to global memory
- BinaryOp -- element-wise binary (add only in phase 1)

Instruction encoding is bit-compatible with the upstream DVM runtime.
"""

from candle_dvm.code cimport Code, RelocAddr
from candle_dvm.isa import (
    V_LOAD,
    V_STORE,
    V_ADD,
    V_ADD_FP16,
    V_SUB,
    V_SUB_FP16,
    V_MUL,
    V_MUL_FP16,
    V_DIV,
    V_DIV_FP16,
    V_MAX,
    V_MAX_FP16,
    V_MIN,
    V_MIN_FP16,
    V_HEAD_ID_OFFSET,
    V_HEAD_EXT_OFFSET,
    V_HEAD_SIZE_OFFSET,
    V_HEAD_SIMD_FLAG_OFFSET,
    V_HEAD_SET_FLAG_OFFSET,
    V_HEAD_WAIT_FLAG_OFFSET,
    V_HEAD_BACK_SET_OFFSET,
    V_HEAD_BACK_WAIT_OFFSET,
    V_HEAD_SET_EVENT_OFFSET,
    V_HEAD_WAIT_EVENT_OFFSET,
    V_HEAD_B_SET_EVENT_OFFSET,
    V_HEAD_B_WAIT_EVENT_OFFSET,
    V_M_HEAD_EXT_OFFSET,
    V_M_HEAD_SIZE_OFFSET,
    V_M_HEAD_SET_FLAG_OFFSET,
    V_M_HEAD_WAIT_FLAG_OFFSET,
    V_M_HEAD_SET_EVENT_OFFSET,
    V_M_HEAD_WAIT_EVENT_OFFSET,
    V_X_MASK,
    V_C_X_BITS,
    UNARY_OPCODE_TABLE,
    BINARY_SCALAR_OPCODE_TABLE,
    COMPARE_OPCODE_TABLE,
    COMPARE_SCALAR_OPCODE_TABLE,
)
from candle_dvm.isa import (
    BINS_ADD as BINS_ADD,
    BINS_MUL as BINS_MUL,
    BINS_DIV as BINS_DIV,
    BINS_MAX as BINS_MAX,
    BINS_MIN as BINS_MIN,
    CMP_EQ as CMP_EQ,
    CMP_NE as CMP_NE,
    CMP_GT as CMP_GT,
    CMP_GE as CMP_GE,
    CMP_LT as CMP_LT,
    CMP_LE as CMP_LE,
)
from candle_dvm.isa cimport make_acc_head, make_simd_head, encode_unary, encode_binary_scalar, encode_compare, encode_compare_scalar

import struct as _struct


# ===================================================================
# Constants  -- match upstream dvm.h enums
# ===================================================================

# DataType  (dvm.h)
DTYPE_BOOL   = 0
DTYPE_FP16   = 1
DTYPE_BF16   = 2
DTYPE_F32    = 3
DTYPE_INT32  = 4

# ITEM_SIZE lookup (bytes per element, indexed by DataType)
cdef int _ITEM_SIZE[5]
_ITEM_SIZE[0] = 1   # kBool
_ITEM_SIZE[1] = 2   # kFloat16
_ITEM_SIZE[2] = 2   # kBFloat16
_ITEM_SIZE[3] = 4   # kFloat32
_ITEM_SIZE[4] = 4   # kInt32

# SIMD width per dtype (elements per SIMD block = 32 / item_size)
# This is for computing lead_stride with RoundUp
cdef int _SIMD_WIDTH[5]
_SIMD_WIDTH[0] = 32  # kBool
_SIMD_WIDTH[1] = 16  # kFloat16
_SIMD_WIDTH[2] = 16  # kBFloat16
_SIMD_WIDTH[3] = 8   # kFloat32
_SIMD_WIDTH[4] = 8   # kInt32

# UnaryType  (dvm.h)
UNARY_SQRT = 0
UNARY_ABS = 1
UNARY_LOG = 2
UNARY_EXP = 3
UNARY_ISFINITE = 5
UNARY_ROUND = 7
UNARY_FLOOR = 8
UNARY_CEIL = 9
UNARY_TRUNC = 10

# BinaryType  (dvm.h)
BIN_ADD = 6
BIN_SUB = 7
BIN_MUL = 8
BIN_DIV = 9
BIN_MAX = 11
BIN_MIN = 12

# ObjectType  (ops.h)
OBJ_LOAD_DUMMY = 0
OBJ_MULTI_LOAD = 1
OBJ_VIEW_LOAD  = 2
OBJ_LOAD       = 3
OBJ_PAD_STORE  = 4
OBJ_STORE      = 5
OBJ_UNARY      = 12
OBJ_BINARY     = 14
OBJ_BINARY_S   = 15
OBJ_COMPARE    = 16
OBJ_COMPARE_S  = 17

# Instruction format constants
# vLoad: RELOC_OFFSET=1, ROUND_OFFSET=3
# vStore: RELOC_OFFSET=2, ROUND_OFFSET=3
# vBinary: 2 words, no reloc

# binary_id_list lookup: (BinaryType, DataType) -> vSimdInsnID
cdef dict _BINARY_ID_TABLE
_BINARY_ID_TABLE = {
    # add
    (BIN_ADD, DTYPE_F32):  V_ADD,
    (BIN_ADD, DTYPE_FP16): V_ADD_FP16,
    # sub
    (BIN_SUB, DTYPE_F32):  V_SUB,
    (BIN_SUB, DTYPE_FP16): V_SUB_FP16,
    # mul
    (BIN_MUL, DTYPE_F32):  V_MUL,
    (BIN_MUL, DTYPE_FP16): V_MUL_FP16,
    # div
    (BIN_DIV, DTYPE_F32):  V_DIV,
    (BIN_DIV, DTYPE_FP16): V_DIV_FP16,
    # max
    (BIN_MAX, DTYPE_F32):  V_MAX,
    (BIN_MAX, DTYPE_FP16): V_MAX_FP16,
    # min
    (BIN_MIN, DTYPE_F32):  V_MIN,
    (BIN_MIN, DTYPE_FP16): V_MIN_FP16,
}


# ===================================================================
# Helper: RoundUp
# ===================================================================
cdef inline long long _round_up(long long num, long long rnd):
    if rnd == 0:
        return 0
    return ((num + rnd - 1) // rnd) * rnd


# ===================================================================
# Helper: vCompactX / vDeCompactX  (matches isa.h macros)
# ===================================================================
cdef inline unsigned long long _compact_x(unsigned long long x):
    return x >> 5

cdef inline unsigned long long _decompact_x(unsigned long long x):
    return x << 5


# ===================================================================
# NDObject -- base class
# ===================================================================

cdef class NDObject:
    """Base class for all NDObject graph nodes.

    Attributes
    ----------
    obj_id : int
        ObjectType enum value.
    type_id : int
        DataType enum value.
    shape_ref : tuple
        Shape as a Python tuple of ints.
    lhs : NDObject or None
        Left-hand side input.
    rhs : NDObject or None
        Right-hand side input.
    index : int
        Position in the kernel's objects list.
    xbuf : int
        Local buffer slot assigned during codegen.
    normalized : bool
        Whether normalize() has been called.
    """

    def __init__(self, int obj_id, int type_id, tuple shape_ref,
                 NDObject lhs=None, NDObject rhs=None):
        self.obj_id = obj_id
        self.type_id = type_id
        self.shape_ref = shape_ref
        self.lhs = lhs
        self.rhs = rhs
        self.index = -1
        self.xbuf = 0
        self.normalized = False
        # Sync flags for pipeline synchronization (set by codegen)
        self.sync_set = 0       # set flag
        self.sync_wait = 0      # wait flag
        self.sync_set_event = 0
        self.sync_wait_event = 0
        # Additional sync flags for SIMD instructions
        self.sync_back_set = 0
        self.sync_back_wait = 0
        self.sync_b_set_event = 0
        self.sync_b_wait_event = 0

    def normalize(self):
        """Re-infer shape from inputs. Override in subclasses."""
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit instructions into code buffer. Override in subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__}.emit() not implemented"
        )


# ===================================================================
# NDAccess -- IO base (adds io_index)
# ===================================================================

cdef class NDAccess(NDObject):
    """Base class for load/store operations that access global memory.

    Attributes
    ----------
    io_index : int
        Index of this I/O operand in the kernel's input/output list.
    """

    def __init__(self, int obj_id, int type_id, tuple shape_ref,
                 int io_index, NDObject lhs=None):
        super().__init__(obj_id, type_id, shape_ref, lhs, None)
        self.io_index = io_index


# ===================================================================
# NDLoad -- input load
# ===================================================================

cdef class NDLoad(NDAccess):
    """Load from global memory.

    Parameters
    ----------
    io_index : int
        Input operand index.
    shape : tuple
        Shape of the tensor to load.
    dtype : int
        DataType enum value (default DTYPE_F32).
    """

    def __init__(self, int io_index, tuple shape, int dtype=DTYPE_F32):
        super().__init__(OBJ_LOAD, dtype, shape, io_index)

    def normalize(self):
        """Validate shape and set output shape = input shape."""
        if not self.shape_ref or len(self.shape_ref) == 0:
            raise ValueError("NDLoad: shape must be non-empty")
        # type_id stays as constructor value
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vLoad instruction.

        Simplified phase-1 path (contiguous, no tiling, no shard):
        - Builds a vLoad struct: {from, xn, tile_stride, body_iter,
          tail_iter, iter_size, pad_size, round_rank=0}
        - Uses vLoad::Encode format:
            pc[0] = vMakeAccHead(V_LOAD, tile_stride<<13 | compact_xn, size)
            pc[1] = from (GM address -- reloc placeholder = 0)
            pc[2] = round_rank<<60 | pad_size<<52 | iter_size<<34 |
                     tail_iter<<16 | body_iter
        """
        cdef int ndim = len(self.shape_ref)
        cdef int item_size = _ITEM_SIZE[self.type_id]
        cdef int simd_width = _SIMD_WIDTH[self.type_id]

        # Compute NDSpaceData-like dims and strides (reversed order)
        # dims[0] = last shape dim, dims[1] = second-to-last, etc.
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.shape_ref[ndim - 1 - i])

        # lead_dim = dims[0] (the innermost / fastest-varying dimension)
        cdef long long lead_dim = dims[0]
        cdef long long lead_align = _round_up(lead_dim, simd_width)

        # stride_back = product of all dims = total elements per tile
        cdef long long stride_back = 1
        for i in range(ndim):
            stride_back *= dims[i]

        # Compute strides (like NDSpaceData::UpdateStride)
        # For contiguous data: strides[0] = lead_align, strides[i] = dims[i] * strides[i-1]
        # stride_back uses strides[-1] from UpdateStride

        # Simplified: for phase-1 contiguous case
        cdef long long src_tile_stride = stride_back // lead_align * lead_dim

        cdef long long body_iter = stride_back // lead_align
        cdef long long iter_size = lead_dim * item_size
        cdef long long pad_size = lead_align * item_size - iter_size
        cdef long long tail_iter

        if body_iter == 1:
            tail_iter = iter_size
        else:
            tail_iter = body_iter
            if pad_size == 0:
                tail_iter *= iter_size
                iter_size *= body_iter
                body_iter = 1

        cdef unsigned long long tile_stride_bytes = src_tile_stride * item_size
        cdef int round_rank = 0  # phase 1: no rounding
        cdef int insn_size = 3   # vLoad::ROUND_OFFSET + 0

        # Encode vLoad
        # pc[0] = vMakeAccHead(V_LOAD, tile_stride<<13 | compact_xn, size)
        #         + sync flags from codegen
        cdef unsigned long long c_xn = _compact_x(<unsigned long long>self.xbuf)
        cdef unsigned long long ext = (tile_stride_bytes << 13) | c_xn
        cdef unsigned long long head = make_acc_head(V_LOAD, ext, insn_size)
        # Apply sync flags (acc head format)
        head |= (<unsigned long long>self.sync_set << V_M_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_M_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_M_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_M_HEAD_WAIT_EVENT_OFFSET)
        code.append_u64(head)

        # pc[1] = from (GM address placeholder -- to be patched via reloc)
        cdef Py_ssize_t reloc_offset = code.size
        code.append_u64(0)  # placeholder for GM address
        relocs.append(RelocAddr(reloc_offset))

        # pc[2] = round_rank<<60 | pad_size<<52 | iter_size<<34 |
        #          tail_iter<<16 | body_iter
        cdef unsigned long long word2 = (
            (<unsigned long long>round_rank << 60) |
            (<unsigned long long>(pad_size & 0xFF) << 52) |
            (<unsigned long long>(iter_size & 0x3FFFF) << 34) |
            (<unsigned long long>(tail_iter & 0x3FFFF) << 16) |
            (<unsigned long long>(body_iter & 0xFFFF))
        )
        code.append_u64(word2)


# ===================================================================
# NDStore -- output store
# ===================================================================

cdef class NDStore(NDAccess):
    """Store to global memory.

    Parameters
    ----------
    io_index : int
        Output operand index.
    src : NDObject
        Source operation producing the data to store.
    """

    def __init__(self, int io_index, NDObject src):
        super().__init__(OBJ_STORE, src.type_id, src.shape_ref, io_index, src)

    def normalize(self):
        """Shape and dtype come from the source."""
        if self.lhs is None:
            raise ValueError("NDStore: source must not be None")
        self.shape_ref = self.lhs.shape_ref
        self.type_id = self.lhs.type_id
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vStore instruction.

        Simplified phase-1 path (contiguous, no tiling, no shard):
        - Uses vStore::Encode format:
            pc[0] = vMakeAccHead(V_STORE, tile_stride<<13 | compact_xn, size)
            pc[1] = round_rank<<60 | pad_size<<52 | iter_size<<34 |
                     iter_tail<<16 | iter_num
            pc[2] = to (GM address -- reloc placeholder = 0)
        """
        cdef NDObject src = self.lhs
        cdef int ndim = len(self.shape_ref)
        cdef int item_size = _ITEM_SIZE[self.type_id]
        cdef int simd_width = _SIMD_WIDTH[self.type_id]

        # Compute dims in reversed order
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.shape_ref[ndim - 1 - i])

        cdef long long lead_dim = dims[0]
        cdef long long lead_align = _round_up(lead_dim, simd_width)

        cdef long long stride_back = 1
        for i in range(ndim):
            stride_back *= dims[i]

        cdef long long dst_tile_stride = stride_back // lead_align * lead_dim

        cdef long long body_iter = stride_back // lead_align
        cdef long long iter_size_val = lead_dim * item_size
        cdef long long pad_size_val = lead_align * item_size - iter_size_val
        cdef long long tail_iter

        if body_iter == 1:
            tail_iter = iter_size_val
        else:
            tail_iter = body_iter
            if pad_size_val == 0:
                tail_iter *= iter_size_val
                iter_size_val *= body_iter
                body_iter = 1

        cdef unsigned long long tile_stride_bytes = dst_tile_stride * item_size
        cdef int round_rank = 0
        cdef int insn_size = 3  # vStore::ROUND_OFFSET + 0

        # Encode vStore
        # pc[0] = vMakeAccHead(V_STORE, tile_stride<<13 | compact_xn, size)
        #         + sync flags from codegen
        cdef unsigned long long c_xn = _compact_x(<unsigned long long>src.xbuf)
        cdef unsigned long long ext = (tile_stride_bytes << 13) | c_xn
        cdef unsigned long long head = make_acc_head(V_STORE, ext, insn_size)
        # Apply sync flags (acc head format)
        head |= (<unsigned long long>self.sync_set << V_M_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_M_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_M_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_M_HEAD_WAIT_EVENT_OFFSET)
        code.append_u64(head)

        # pc[1] = round_rank<<60 | pad_size<<52 | iter_size<<34 |
        #          iter_tail<<16 | iter_num
        cdef unsigned long long word1 = (
            (<unsigned long long>round_rank << 60) |
            (<unsigned long long>(pad_size_val & 0xFF) << 52) |
            (<unsigned long long>(iter_size_val & 0x3FFFF) << 34) |
            (<unsigned long long>(tail_iter & 0x3FFFF) << 16) |
            (<unsigned long long>(body_iter & 0xFFFF))
        )
        code.append_u64(word1)

        # pc[2] = to (GM address placeholder -- to be patched via reloc)
        cdef Py_ssize_t reloc_offset = code.size
        code.append_u64(0)  # placeholder for GM address
        relocs.append(RelocAddr(reloc_offset))


# ===================================================================
# FlexOp -- ops that may use workspace
# ===================================================================

cdef class FlexOp(NDObject):
    """Base class for operations that may require workspace buffers."""

    def __init__(self, int obj_id, int type_id, tuple shape_ref,
                 NDObject lhs=None, NDObject rhs=None):
        super().__init__(obj_id, type_id, shape_ref, lhs, rhs)
        self.workspace_xbuf = 0

    def workspace_slots(self):
        """Return the number of workspace slots this op needs (default 0)."""
        return 0


# ===================================================================
# BinaryOp -- element-wise binary
# ===================================================================

cdef class BinaryOp(FlexOp):
    """Element-wise binary operation.

    Parameters
    ----------
    op_type : int
        BinaryType enum value (e.g. BIN_ADD).
    lhs : NDObject
        Left-hand side input.
    rhs : NDObject
        Right-hand side input.
    """

    def __init__(self, int op_type, NDObject lhs, NDObject rhs):
        super().__init__(OBJ_BINARY, lhs.type_id, lhs.shape_ref, lhs, rhs)
        self.op_type = op_type

    def normalize(self):
        """Check shapes match and propagate shape/dtype from lhs."""
        if self.lhs is None or self.rhs is None:
            raise ValueError("BinaryOp: both lhs and rhs must be set")
        if self.lhs.shape_ref != self.rhs.shape_ref:
            raise ValueError(
                f"BinaryOp: shape mismatch: "
                f"{self.lhs.shape_ref} vs {self.rhs.shape_ref}"
            )
        self.shape_ref = self.lhs.shape_ref
        self.type_id = self.lhs.type_id
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vBinary instruction.

        Uses vBinary::Encode format:
            pc[0] = vMakeSimdHead(id, xn, 2)  -- xn goes in ext field
            pc[1] = count << 48 | xd << 18 | xm

        The instruction ID comes from binary_id_list lookup.
        For fp32 add: id = V_ADD (=18).
        count = stride_back() which for a contiguous tensor is the
        last dim size (innermost dimension).
        """
        cdef int ndim = len(self.shape_ref)
        cdef int simd_width = _SIMD_WIDTH[self.type_id]

        # stride_back: for phase 1 (contiguous), this is
        # the last stride from NDSpaceData, which equals RoundUp(last_dim, simd_width)
        # when ndim==1, or for ndim>=2 the full product-based stride.
        # In upstream: nd_.stride_back() for the BinaryOp, which after Normalize
        # equals lhs_->nd_.stride_back()
        #
        # NDSpaceData::UpdateStride computes:
        #   strides[0] = RoundUp(dims[0], simd_width)
        #   strides[i] = dims[i] * strides[i-1]
        # stride_back() = strides[size-1]
        #
        # But BinaryOp::Emit uses nd_.stride_back() as 'count'.
        # This is the total number of SIMD-aligned elements.
        # For phase-1 we compute it the same way.

        # dims in reversed order (like NDSpaceData)
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.shape_ref[ndim - 1 - i])

        cdef long long stride = _round_up(<long long>dims[0], simd_width)
        for i in range(1, ndim):
            stride = <long long>dims[i] * stride
        cdef unsigned long long count = <unsigned long long>stride

        # Look up instruction ID
        cdef tuple key = (self.op_type, self.type_id)
        if key not in _BINARY_ID_TABLE:
            raise ValueError(
                f"BinaryOp: unsupported op_type={self.op_type}, "
                f"dtype={self.type_id}"
            )
        cdef unsigned long long insn_id = <unsigned long long>_BINARY_ID_TABLE[key]

        cdef unsigned long long xd = <unsigned long long>self.xbuf
        cdef unsigned long long xn = <unsigned long long>self.lhs.xbuf
        cdef unsigned long long xm = <unsigned long long>self.rhs.xbuf
        cdef int insn_size = 2

        # pc[0] = vMakeSimdHead(id, xn, 2) + sync flags from codegen
        cdef unsigned long long head = make_simd_head(insn_id, xn, insn_size)
        # Apply sync flags (simd head format)
        head |= (<unsigned long long>self.sync_set << V_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_back_set << V_HEAD_BACK_SET_OFFSET)
        head |= (<unsigned long long>self.sync_back_wait << V_HEAD_BACK_WAIT_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_HEAD_WAIT_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_set_event & 0x7) << V_HEAD_B_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_wait_event & 0x7) << V_HEAD_B_WAIT_EVENT_OFFSET)
        code.append_u64(head)

        # pc[1] = count << 48 | xd << 18 | xm
        cdef unsigned long long word1 = (count << 48) | (xd << 18) | xm
        code.append_u64(word1)


# ===================================================================
# UnaryOp -- element-wise unary (Batch A)
# ===================================================================

cdef class UnaryOp(FlexOp):
    """Element-wise unary operation.

    Parameters
    ----------
    op_type : int
        UnaryType enum value (e.g. UNARY_SQRT).
    src : NDObject
        Input operand.
    """

    def __init__(self, int op_type, NDObject src):
        super().__init__(OBJ_UNARY, src.type_id, src.shape_ref, src, None)
        self.op_type = op_type

    def normalize(self):
        """Output shape = input shape.  Output dtype = input dtype for most ops.
        isfinite (UNARY_ISFINITE) returns DTYPE_BOOL.
        """
        if self.lhs is None:
            raise ValueError("UnaryOp: source must not be None")
        self.shape_ref = self.lhs.shape_ref
        self.type_id = self.lhs.type_id
        if self.op_type == UNARY_ISFINITE:
            self.type_id = DTYPE_BOOL
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vUnary instruction.

        Uses vUnary::Encode format (2 words):
            pc[0] = vMakeSimdHead(opcode, xd, 2)
            pc[1] = xn << 32 | count

        Note: for vUnary the ext field carries xd (the destination),
        unlike vBinary where ext = xn.
        count = nd_.stride_back() = SIMD-aligned element count.
        """
        cdef int ndim = len(self.shape_ref)
        cdef int simd_width = _SIMD_WIDTH[self.lhs.type_id]

        # stride_back: same calculation as BinaryOp
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.lhs.shape_ref[ndim - 1 - i])

        cdef long long stride = _round_up(<long long>dims[0], simd_width)
        for i in range(1, ndim):
            stride = <long long>dims[i] * stride
        cdef unsigned long long count = <unsigned long long>stride

        # Look up instruction opcode
        cdef tuple key = (self.op_type, self.lhs.type_id)
        if key not in UNARY_OPCODE_TABLE:
            raise NotImplementedError(
                f"UnaryOp: unsupported op_type={self.op_type}, "
                f"dtype={self.lhs.type_id}"
            )
        cdef unsigned long long insn_id = <unsigned long long>UNARY_OPCODE_TABLE[key]

        # Use encode_unary helper
        cdef list words = encode_unary(insn_id, <unsigned long long>self.xbuf,
                                        <unsigned long long>self.lhs.xbuf, count)

        # Apply sync flags to head word
        cdef unsigned long long head = <unsigned long long>words[0]
        head |= (<unsigned long long>self.sync_set << V_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_back_set << V_HEAD_BACK_SET_OFFSET)
        head |= (<unsigned long long>self.sync_back_wait << V_HEAD_BACK_WAIT_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_HEAD_WAIT_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_set_event & 0x7) << V_HEAD_B_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_wait_event & 0x7) << V_HEAD_B_WAIT_EVENT_OFFSET)
        code.append_u64(head)

        code.append_u64(<unsigned long long>words[1])


# ===================================================================
# Scalar bit packing helpers
# ===================================================================

cdef unsigned long long _float_to_fp32_bits(double val):
    """Convert a Python float to raw IEEE754 32-bit bits."""
    cdef bytes b = _struct.pack('<f', val)
    return <unsigned long long>_struct.unpack('<I', b)[0]

cdef unsigned long long _float_to_fp16_bits(double val):
    """Convert a Python float to raw IEEE754 16-bit half bits."""
    cdef bytes b = _struct.pack('<e', val)
    return <unsigned long long>_struct.unpack('<H', b)[0]


# ===================================================================
# BinaryScalarOp -- element-wise binary with scalar (Batch C)
# ===================================================================

cdef class BinaryScalarOp(FlexOp):
    """Element-wise binary operation with a scalar constant.

    Parameters
    ----------
    op_type : int
        BinarySOpType enum value (e.g. BINS_ADD).
    src : NDObject
        Input operand (tensor).
    scalar : float
        Scalar constant value.
    """

    def __init__(self, int op_type, NDObject src, double scalar):
        super().__init__(OBJ_BINARY_S, src.type_id, src.shape_ref, src, None)
        self.op_type = op_type
        self.scalar = scalar

    def normalize(self):
        """Output shape = input shape.  Output dtype = input dtype."""
        if self.lhs is None:
            raise ValueError("BinaryScalarOp: source must not be None")
        self.shape_ref = self.lhs.shape_ref
        self.type_id = self.lhs.type_id
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vBinaryS instruction.

        Uses vBinaryS::Encode format (2 words):
            pc[0] = make_simd_head(opcode, xn, 2)
            pc[1] = scalar_bits << 32 | vCompactX(xd) << 16 | count

        count = nd_.stride_back() = SIMD-aligned element count.
        """
        cdef int ndim = len(self.shape_ref)
        cdef int simd_width = _SIMD_WIDTH[self.lhs.type_id]

        # stride_back: same calculation as BinaryOp
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.lhs.shape_ref[ndim - 1 - i])

        cdef long long stride = _round_up(<long long>dims[0], simd_width)
        for i in range(1, ndim):
            stride = <long long>dims[i] * stride
        cdef unsigned long long count = <unsigned long long>stride

        # Look up instruction opcode
        cdef tuple key = (self.op_type, self.lhs.type_id)
        if key not in BINARY_SCALAR_OPCODE_TABLE:
            raise NotImplementedError(
                f"BinaryScalarOp: unsupported op_type={self.op_type}, "
                f"dtype={self.lhs.type_id}"
            )
        cdef unsigned long long insn_id = <unsigned long long>BINARY_SCALAR_OPCODE_TABLE[key]

        # Pack scalar bits according to dtype
        cdef unsigned long long scalar_bits
        if self.lhs.type_id == DTYPE_FP16:
            scalar_bits = _float_to_fp16_bits(self.scalar)
        else:
            # fp32
            scalar_bits = _float_to_fp32_bits(self.scalar)

        # Use encode_binary_scalar helper
        cdef list words = encode_binary_scalar(
            insn_id,
            <unsigned long long>self.lhs.xbuf,
            <unsigned long long>self.xbuf,
            count,
            scalar_bits,
        )

        # Apply sync flags to head word
        cdef unsigned long long head = <unsigned long long>words[0]
        head |= (<unsigned long long>self.sync_set << V_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_back_set << V_HEAD_BACK_SET_OFFSET)
        head |= (<unsigned long long>self.sync_back_wait << V_HEAD_BACK_WAIT_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_HEAD_WAIT_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_set_event & 0x7) << V_HEAD_B_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_wait_event & 0x7) << V_HEAD_B_WAIT_EVENT_OFFSET)
        code.append_u64(head)

        code.append_u64(<unsigned long long>words[1])


# ===================================================================
# CompareOp -- element-wise tensor-tensor compare (Batch D)
# ===================================================================

cdef class CompareOp(FlexOp):
    """Element-wise tensor-tensor compare operation.

    Parameters
    ----------
    cmp_type : int
        Compare semantic type (CMP_EQ, CMP_NE, CMP_GT, CMP_GE, CMP_LT, CMP_LE).
    lhs : NDObject
        Left-hand side input tensor.
    rhs : NDObject
        Right-hand side input tensor.
    """

    def __init__(self, int cmp_type, NDObject lhs, NDObject rhs):
        super().__init__(OBJ_COMPARE, DTYPE_BOOL, lhs.shape_ref, lhs, rhs)
        self.cmp_type = cmp_type

    def workspace_slots(self):
        return 1

    def normalize(self):
        """Check shapes match and set output dtype to DTYPE_BOOL."""
        if self.lhs is None or self.rhs is None:
            raise ValueError("CompareOp: both lhs and rhs must be set")
        if self.lhs.shape_ref != self.rhs.shape_ref:
            raise ValueError(
                f"CompareOp: shape mismatch: "
                f"{self.lhs.shape_ref} vs {self.rhs.shape_ref}"
            )
        self.shape_ref = self.lhs.shape_ref
        self.type_id = DTYPE_BOOL
        self.normalized = True

    def emit(self, Code code, list relocs):
        """Emit a vCompare instruction (2 words)."""
        cdef int src_dtype = self.lhs.type_id
        if src_dtype not in COMPARE_OPCODE_TABLE:
            raise NotImplementedError(
                f"CompareOp: unsupported dtype={src_dtype}"
            )
        cdef unsigned long long insn_id = <unsigned long long>COMPARE_OPCODE_TABLE[src_dtype]

        cdef int ndim = len(self.lhs.shape_ref)
        cdef int simd_width = _SIMD_WIDTH[src_dtype]
        cdef list dims = []
        cdef int i
        for i in range(ndim):
            dims.append(self.lhs.shape_ref[ndim - 1 - i])
        cdef long long stride = _round_up(<long long>dims[0], simd_width)
        for i in range(1, ndim):
            stride = <long long>dims[i] * stride
        cdef unsigned long long count = <unsigned long long>stride

        cdef list words = encode_compare(
            insn_id,
            <unsigned long long>self.cmp_type,
            <unsigned long long>self.lhs.xbuf,
            <unsigned long long>self.rhs.xbuf,
            <unsigned long long>self.xbuf,
            <unsigned long long>self.workspace_xbuf,
            count,
        )

        cdef unsigned long long head = <unsigned long long>words[0]
        head |= (<unsigned long long>self.sync_set << V_HEAD_SET_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_wait << V_HEAD_WAIT_FLAG_OFFSET)
        head |= (<unsigned long long>self.sync_back_set << V_HEAD_BACK_SET_OFFSET)
        head |= (<unsigned long long>self.sync_back_wait << V_HEAD_BACK_WAIT_OFFSET)
        head |= (<unsigned long long>(self.sync_set_event & 0x7) << V_HEAD_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_wait_event & 0x7) << V_HEAD_WAIT_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_set_event & 0x7) << V_HEAD_B_SET_EVENT_OFFSET)
        head |= (<unsigned long long>(self.sync_b_wait_event & 0x7) << V_HEAD_B_WAIT_EVENT_OFFSET)
        code.append_u64(head)
        code.append_u64(<unsigned long long>words[1])
