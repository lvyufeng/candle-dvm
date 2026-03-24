# cython: language_level=3
"""DVM kernel hierarchy -- phase-1 static vector kernel.

Implements the kernel hierarchy for building and compiling DVM compute
kernels:

- VKernel (base)
  - VectorKernel (holds objects list, Code, xbuf accounting)
    - VKernelS (static shape vector kernel -- fully implemented)
    - VKernelD (placeholder for dynamic shape)

Phase-1 scope: contiguous tensors, element-wise add, single-tile codegen
on Ascend 910B (architecture c220).
"""

from candle_dvm.code cimport Code, RelocAddr
from candle_dvm.ops cimport NDObject, NDAccess, NDLoad, NDStore, FlexOp, BinaryOp
from candle_dvm.pass_ import run_passes


# ===================================================================
# Constants
# ===================================================================

# 910B phase-1 block_dim -- locked from upstream trace
# (tests/fixtures/upstream_add_trace.txt first line: block_dim=1)
PHASE1_BLOCK_DIM_910B = 1

# xbuf base offset: first 512 bytes (0x200) of local memory are reserved
# on 910B for hardware use (sync flags, stack, etc.)
DEF XBUF_BASE_OFFSET = 0x200

# ITEM_SIZE lookup (bytes per element, indexed by DataType)
cdef int _ITEM_SIZE[5]
_ITEM_SIZE[0] = 1   # kBool
_ITEM_SIZE[1] = 2   # kFloat16
_ITEM_SIZE[2] = 2   # kBFloat16
_ITEM_SIZE[3] = 4   # kFloat32
_ITEM_SIZE[4] = 4   # kInt32

# ObjectType constants (match ops.pyx)
cdef int _OBJ_LOAD  = 3
cdef int _OBJ_STORE = 5


# ===================================================================
# Helpers
# ===================================================================

cdef inline long long _round_up_32(long long val):
    """Round *val* up to the nearest multiple of 32."""
    return (val + 31) & ~<long long>31


cdef inline long long _slot_size(tuple shape, int type_id):
    """Compute xbuf slot size: total elements * element_size, rounded up to 32."""
    cdef long long total = 1
    cdef int i
    for i in range(len(shape)):
        total *= <long long>shape[i]
    cdef long long raw = total * _ITEM_SIZE[type_id]
    return _round_up_32(raw)


# ===================================================================
# VKernel -- base class
# ===================================================================

cdef class VKernel:
    """Base class for all DVM kernel objects.

    Attributes
    ----------
    code : Code
        The code buffer holding compiled bytecode.
    relocs : list[RelocAddr]
        Relocation entries for IO operands.
    ktype : int
        Kernel type (0 = vector).
    flags : int
        Kernel flags (0 = static).
    """

    def __init__(self, int ktype=0, int flags=0):
        self.code = Code()
        self.relocs = []
        self.ktype = ktype
        self.flags = flags

    def append(self, NDObject obj):
        """Append an NDObject to the kernel. Override in subclasses."""
        pass

    def normalize(self):
        """Normalize the kernel. Override in subclasses."""
        pass

    def codegen(self):
        """Run code generation. Override in subclasses."""
        pass


# ===================================================================
# VectorKernel -- holds objects list, Code, xbuf accounting
# ===================================================================

cdef class VectorKernel(VKernel):
    """Intermediate class for vector kernels.

    Adds the objects list, tile tracking, and target type.

    Attributes
    ----------
    objects : list[NDObject]
        Normalized (flat) object list.
    tile_num : int
        Number of tiles in the computation domain.
    tile_size : int
        Element count per tile.
    target : int
        Target type (0=vector, 1=cube, 2=mix).
    """

    def __init__(self, int ktype=0, int flags=0):
        super().__init__(ktype, flags)
        self.objects = []
        self.tile_num = 0
        self.tile_size = 0
        self.target = 0  # kTargetVec


# ===================================================================
# VKernelS -- static shape vector kernel (fully implemented)
# ===================================================================

cdef class VKernelS(VectorKernel):
    """Static-shape vector kernel with full phase-1 codegen pipeline.

    Codegen pipeline:
    1. gather build ops
    2. normalize each op
    3. run the no-op pass manager
    4. initialize Code buffer
    5. reserve head words (ffts_addr=0, entry=0)
    6. assign xbuf slots to each op (monotonic allocation)
    7. emit each normalized op into the code buffer
    8. append terminating zero instruction
    9. compute tile_num from the domain
    10. finalize entry via Code.gen_entry_v(tile_num, block_dim, data_size-16)
    11. update code head with the entry word
    12. set code.block_dim and code.target

    Attributes
    ----------
    build_ops : list[NDObject]
        Raw ops appended before normalize.
    """

    def __init__(self, int flags=0):
        super().__init__(0, flags)  # ktype=0 (vector)
        self.build_ops = []

    def append(self, NDObject obj):
        """Append an NDObject to the build list."""
        self.build_ops.append(obj)

    def normalize(self):
        """Normalize all build ops and flatten into objects list.

        Steps:
        1. Normalize each build op.
        2. Run the no-op pass manager.
        3. Flatten into self.objects with indices assigned.
        """
        cdef list ops = self.build_ops
        cdef NDObject obj
        cdef int idx

        # Step 1: normalize each op
        for obj in ops:
            obj.normalize()

        # Step 2: run passes (no-op in phase 1)
        ops = run_passes(ops)

        # Step 3: flatten into objects with assigned indices
        self.objects = []
        idx = 0
        for obj in ops:
            obj.index = idx
            self.objects.append(obj)
            idx += 1

    def codegen(self):
        """Run the full phase-1 codegen pipeline.

        Produces a complete code buffer with header, instructions,
        relocation list, and entry word.
        """
        cdef Code code = self.code
        cdef list relocs = []
        cdef list objects = self.objects
        cdef NDObject obj
        cdef long long slot_sz

        # Step 4-5: reserve head words (ffts_addr=0, entry=0)
        code.append_u64(0)  # word 0: ffts_addr placeholder
        code.append_u64(0)  # word 1: entry placeholder

        # Step 6: assign xbuf slots (monotonic allocation)
        # xbuf addresses start at XBUF_BASE_OFFSET (0x200) on 910B
        cdef long long xbuf_cursor = XBUF_BASE_OFFSET
        for obj in objects:
            if obj.obj_id == _OBJ_LOAD:
                # NDLoad: allocate a new slot
                slot_sz = _slot_size(obj.shape_ref, obj.type_id)
                obj.xbuf = <int>xbuf_cursor
                xbuf_cursor += slot_sz
            elif obj.obj_id == _OBJ_STORE:
                # NDStore: share source's xbuf (no new allocation)
                obj.xbuf = obj.lhs.xbuf
            else:
                # BinaryOp / other SIMD ops: allocate a new slot
                slot_sz = _slot_size(obj.shape_ref, obj.type_id)
                obj.xbuf = <int>xbuf_cursor
                xbuf_cursor += slot_sz

        # Step 6b: assign sync flags for pipeline synchronization
        # Pattern for load-...-simd-store graphs (phase 1):
        #   First load:  wait simd_load_sync event 0
        #   Last load:   set load_simd_sync event 0
        #   SIMD ops:    wait load_simd_sync + wait store_simd_sync,
        #                set simd_store_sync + set simd_load_sync
        #   Store:       wait simd_store_sync, set store_simd_sync
        cdef list loads = []
        cdef list simd_ops = []
        cdef list stores = []
        cdef NDObject sync_obj
        for sync_obj in objects:
            if sync_obj.obj_id == _OBJ_LOAD:
                loads.append(sync_obj)
            elif sync_obj.obj_id == _OBJ_STORE:
                stores.append(sync_obj)
            else:
                simd_ops.append(sync_obj)

        if loads:
            # First load: wait simd_load_sync
            (<NDObject>loads[0]).sync_wait = 1
            (<NDObject>loads[0]).sync_wait_event = 0
            # Last load: set load_simd_sync
            (<NDObject>loads[-1]).sync_set = 1
            (<NDObject>loads[-1]).sync_set_event = 0

        for sync_obj in simd_ops:
            # SIMD ops: wait load_simd_sync + wait store_simd_sync
            sync_obj.sync_wait = 1
            sync_obj.sync_wait_event = 0
            sync_obj.sync_back_wait = 1
            sync_obj.sync_b_wait_event = 0
            # Set simd_store_sync + simd_load_sync
            sync_obj.sync_set = 1
            sync_obj.sync_set_event = 0
            sync_obj.sync_back_set = 1
            sync_obj.sync_b_set_event = 0

        for sync_obj in stores:
            # Store: wait simd_store_sync, set store_simd_sync
            sync_obj.sync_wait = 1
            sync_obj.sync_wait_event = 0
            sync_obj.sync_set = 1
            sync_obj.sync_set_event = 0

        # Step 7: emit each op into the code buffer
        for obj in objects:
            obj.emit(code, relocs)

        # Step 8: append terminating zero instruction
        code.append_u64(0)

        # Step 9: compute tile_num.
        # Phase 1: all data fits in local memory (no tiling needed),
        # so tile_num = 1 for contiguous tensors on 910B.
        # Multi-tile support will be added in a later phase.
        cdef long long _tile_num = 1
        self.tile_num = _tile_num

        # Step 10-11: compute data_size and finalize entry
        cdef int data_size = code.size
        code.data_size = data_size

        cdef int block_dim = PHASE1_BLOCK_DIM_910B
        cdef long long tile_num = self.tile_num

        # GenEntryV: data_size for the entry is total - head (16 bytes)
        cdef unsigned long long entry = Code.gen_entry_v(
            <int>tile_num, block_dim, data_size - 16
        )

        # Update code head: write entry at word 1, ffts at word 0
        # code buffer already has words 0 and 1 written; overwrite them
        # We need to write directly. Since we can only append, we use
        # the bind_relocs trick or re-create. But actually Code has
        # read_u64_at. Let's use a more direct approach:
        # Write ffts_addr=0 at offset 0 and entry at offset 8.
        cdef list head_relocs = [RelocAddr(0), RelocAddr(8)]
        cdef list head_values = [0, entry]
        code.bind_relocs(head_relocs, head_values)

        # Step 12: set code.block_dim and code.target
        code.block_dim = block_dim
        self.target = 0  # kTargetVec

        # Store relocs on the kernel
        self.relocs = relocs

    def debug_header(self):
        """Return a dict summarising the kernel header fields.

        Returns
        -------
        dict
            Keys: ``target``, ``block_dim``, ``data_size``, ``tile_num``.
        """
        return {
            "target": self.target,
            "block_dim": self.code.block_dim,
            "data_size": self.code.data_size,
            "tile_num": self.tile_num,
        }


# ===================================================================
# VKernelD -- placeholder for dynamic shape
# ===================================================================

cdef class VKernelD(VKernelS):
    """Dynamic-shape vector kernel -- placeholder for future phases."""

    def __init__(self, int flags=1):
        super().__init__(flags)
