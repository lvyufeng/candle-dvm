"""Tests for candle_dvm.ops -- NDObject hierarchy, normalize, emit.

Also tests candle_dvm.pass_ (no-op pass manager).
"""

import pytest
from candle_dvm import isa
from candle_dvm.code import Code, RelocAddr
from candle_dvm.ops import (
    DTYPE_BOOL,
    DTYPE_F32,
    DTYPE_FP16,
    DTYPE_INT32,
    BIN_ADD,
    BIN_SUB,
    BIN_MUL,
    BIN_DIV,
    BIN_MAX,
    BIN_MIN,
    BINS_ADD,
    BINS_DIV,
    UNARY_ISFINITE,
    UNARY_SQRT,
    OBJ_LOAD,
    OBJ_STORE,
    OBJ_BINARY,
    CMP_EQ,
    CMP_GT,
    CMP_LE,
    NDLoad,
    NDStore,
    UnaryOp,
    BinaryOp,
    BinaryScalarOp,
    CompareOp,
    CompareScalarOp,
)
from candle_dvm.pass_ import run_passes


# ===================================================================
# Constants
# ===================================================================

def test_dtype_f32_value():
    """DTYPE_F32 must be 3 (matching upstream kFloat32)."""
    assert DTYPE_F32 == 3


def test_bin_add_value():
    """BIN_ADD must be 6 (matching upstream kAdd index in BinaryType)."""
    assert BIN_ADD == 6


def test_obj_type_constants():
    """ObjectType enum values must match upstream."""
    assert OBJ_LOAD == 3      # kLoad
    assert OBJ_STORE == 5     # kStore
    assert OBJ_BINARY == 14   # kBinary


# ===================================================================
# NDLoad -- construction and normalize
# ===================================================================

class TestNDLoad:
    def test_construction(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        assert load.io_index == 0
        assert load.shape_ref == (4, 8)
        assert load.type_id == DTYPE_F32
        assert load.obj_id == OBJ_LOAD

    def test_normalize_sets_shape(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        assert load.shape_ref == (4, 8)
        assert load.type_id == DTYPE_F32
        assert load.normalized is True

    def test_normalize_1d(self):
        load = NDLoad(io_index=0, shape=(16,), dtype=DTYPE_F32)
        load.normalize()
        assert load.shape_ref == (16,)

    def test_emit_appends_nonzero_words(self):
        """NDLoad.emit must append non-zero instruction words to code."""
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        code = Code()
        relocs = []
        try:
            load.emit(code, relocs)
            assert code.size > 0
            # At least one word must be non-zero
            found_nonzero = False
            for i in range(0, code.size, 8):
                if code.read_u64_at(i) != 0:
                    found_nonzero = True
                    break
            assert found_nonzero, "emit produced all-zero words"
        finally:
            code.free()

    def test_emit_registers_reloc(self):
        """NDLoad.emit must register exactly one RelocAddr for GM address."""
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        code = Code()
        relocs = []
        try:
            load.emit(code, relocs)
            assert len(relocs) == 1
            assert isinstance(relocs[0], RelocAddr)
        finally:
            code.free()

    def test_emit_uses_v_load_opcode(self):
        """The head word's ID field must contain ACCESS_FUNC_OFFSET[V_LOAD]."""
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        code = Code()
        relocs = []
        try:
            load.emit(code, relocs)
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.ACCESS_FUNC_OFFSET[isa.V_LOAD]
        finally:
            code.free()


# ===================================================================
# NDStore -- construction and normalize
# ===================================================================

class TestNDStore:
    def test_construction(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        store = NDStore(io_index=0, src=load)
        assert store.io_index == 0
        assert store.obj_id == OBJ_STORE

    def test_normalize_propagates_from_src(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        store = NDStore(io_index=0, src=load)
        store.normalize()
        assert store.shape_ref == load.shape_ref
        assert store.type_id == load.type_id
        assert store.normalized is True

    def test_emit_appends_nonzero_words(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        store = NDStore(io_index=0, src=load)
        store.normalize()
        store.xbuf = 1

        code = Code()
        relocs = []
        try:
            store.emit(code, relocs)
            assert code.size > 0
            found_nonzero = False
            for i in range(0, code.size, 8):
                if code.read_u64_at(i) != 0:
                    found_nonzero = True
                    break
            assert found_nonzero
        finally:
            code.free()

    def test_emit_registers_reloc(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        store = NDStore(io_index=0, src=load)
        store.normalize()
        store.xbuf = 1

        code = Code()
        relocs = []
        try:
            store.emit(code, relocs)
            assert len(relocs) == 1
            assert isinstance(relocs[0], RelocAddr)
        finally:
            code.free()

    def test_emit_uses_v_store_opcode(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        load.normalize()
        load.xbuf = 0

        store = NDStore(io_index=0, src=load)
        store.normalize()
        store.xbuf = 1

        code = Code()
        relocs = []
        try:
            store.emit(code, relocs)
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.ACCESS_FUNC_OFFSET[isa.V_STORE]
        finally:
            code.free()


# ===================================================================
# BinaryOp -- construction and normalize
# ===================================================================

class TestBinaryOp:
    def _make_loads(self, shape=(4, 8)):
        lhs = NDLoad(io_index=0, shape=shape, dtype=DTYPE_F32)
        lhs.normalize()
        lhs.xbuf = 0
        rhs = NDLoad(io_index=1, shape=shape, dtype=DTYPE_F32)
        rhs.normalize()
        rhs.xbuf = 1
        return lhs, rhs

    def test_construction(self):
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        assert op.obj_id == OBJ_BINARY
        assert op.lhs is lhs
        assert op.rhs is rhs

    def test_normalize_propagates_shape_and_dtype(self):
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        assert op.shape_ref == lhs.shape_ref
        assert op.type_id == lhs.type_id
        assert op.normalized is True

    def test_normalize_mismatched_shapes_raises(self):
        lhs = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        lhs.normalize()
        rhs = NDLoad(io_index=1, shape=(4, 16), dtype=DTYPE_F32)
        rhs.normalize()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        with pytest.raises(ValueError):
            op.normalize()

    def test_emit_appends_exactly_two_words(self):
        """BinaryOp.emit should append exactly 2 u64 words (vBinary.Encode)."""
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16  # 2 words * 8 bytes
        finally:
            code.free()

    def test_emit_no_relocs(self):
        """BinaryOp.emit should not register any relocations."""
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert len(relocs) == 0
        finally:
            code.free()

    def test_emit_head_uses_v_add_opcode(self):
        """For fp32 add, the head word ID must be SIMD_FUNC_OFFSET[V_ADD]."""
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_ADD]
        finally:
            code.free()

    def test_emit_head_has_simd_flag(self):
        """The SIMD flag (bit 0) must be set in the head word."""
        lhs, rhs = self._make_loads()
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            head = code.read_u64_at(0)
            assert (head & 1) == 1
        finally:
            code.free()

    def test_emit_payload_encodes_xbuf_and_count(self):
        """Verify second word packs count, xd, xm per vBinary::Encode.

        pc[1] = count << 48 | xd << 18 | xm

        count = nd_.stride_back() which for NDSpaceData is strides[-1].
        For shape (4, 8), dtype fp32 (simd_width=8):
          dims reversed = [8, 4]
          strides[0] = RoundUp(8, 8) = 8
          strides[1] = 4 * 8 = 32
          stride_back = 32
        """
        lhs, rhs = self._make_loads(shape=(4, 8))
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            payload = code.read_u64_at(8)
            xm = payload & isa.V_X_MASK
            xd = (payload >> 18) & isa.V_X_MASK
            count = payload >> 48
            assert xd == 2   # op.xbuf
            assert xm == 1   # rhs.xbuf
            assert count == 32  # stride_back for shape (4, 8) fp32
        finally:
            code.free()

    def test_emit_head_ext_encodes_xn(self):
        """The ext field in the head must contain lhs.xbuf (= xn).

        head = vMakeSimdHead(id, xn, 2)
        ext = (head >> V_HEAD_EXT_OFFSET) & V_HEAD_EXT_MASK
        """
        lhs, rhs = self._make_loads()
        lhs.xbuf = 5
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            head = code.read_u64_at(0)
            ext = (head >> isa.V_HEAD_EXT_OFFSET) & isa.V_HEAD_EXT_MASK
            assert ext == 5  # lhs.xbuf
        finally:
            code.free()


# ===================================================================
# Full graph: load -> add -> store  (emit pipeline)
# ===================================================================

class TestFullGraph:
    def test_load_add_store_emit_pipeline(self):
        """Build a minimal graph: load0, load1 -> add -> store.

        Verify the entire emit pipeline produces non-zero code
        with correct number of relocations.
        """
        load0 = NDLoad(io_index=0, shape=(2, 4), dtype=DTYPE_F32)
        load1 = NDLoad(io_index=1, shape=(2, 4), dtype=DTYPE_F32)
        add_op = BinaryOp(BIN_ADD, load0, load1)
        store = NDStore(io_index=0, src=add_op)

        # Normalize all
        load0.normalize()
        load1.normalize()
        add_op.normalize()
        store.normalize()

        # Assign xbuf slots
        load0.xbuf = 0
        load1.xbuf = 1
        add_op.xbuf = 2
        store.xbuf = 3  # not used by store but set for completeness

        # Emit all
        code = Code()
        relocs = []
        try:
            load0.emit(code, relocs)
            load1.emit(code, relocs)
            add_op.emit(code, relocs)
            store.emit(code, relocs)

            # Code should have content
            assert code.size > 0

            # Two loads + one store = 3 relocs
            assert len(relocs) == 3
        finally:
            code.free()


# ===================================================================
# pass_ -- no-op pass manager
# ===================================================================

class TestPassManager:
    def test_run_passes_returns_same_list(self):
        objs = [1, 2, 3]
        result = run_passes(objs)
        assert result is objs

    def test_run_passes_empty(self):
        result = run_passes([])
        assert result == []

    def test_run_passes_with_ndobjects(self):
        load = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
        result = run_passes([load])
        assert result[0] is load


# ===================================================================
# UnaryOp -- construction, normalize, emit
# ===================================================================

class TestUnaryOp:
    def _make_load(self, shape=(4, 8), dtype=DTYPE_F32):
        load = NDLoad(io_index=0, shape=shape, dtype=dtype)
        load.normalize()
        load.xbuf = 0
        return load

    def test_normalize_preserves_shape_and_dtype_for_sqrt(self):
        """UnaryOp normalize should set output shape = input shape
        and output dtype = input dtype for sqrt."""
        src = self._make_load(shape=(4, 8), dtype=DTYPE_F32)
        op = UnaryOp(UNARY_SQRT, src)
        op.normalize()
        assert op.shape_ref == (4, 8)
        assert op.type_id == DTYPE_F32
        assert op.normalized is True

    def test_normalize_sets_bool_dtype_for_isfinite(self):
        """UnaryOp(isfinite) should set output dtype to DTYPE_BOOL."""
        src = self._make_load(shape=(4, 8), dtype=DTYPE_F32)
        op = UnaryOp(UNARY_ISFINITE, src)
        op.normalize()
        assert op.shape_ref == (4, 8)
        assert op.type_id == DTYPE_BOOL

    def test_emit_appends_exactly_two_words(self):
        """UnaryOp.emit should append exactly 2 u64 words (16 bytes)."""
        src = self._make_load(shape=(4, 8), dtype=DTYPE_F32)
        op = UnaryOp(UNARY_SQRT, src)
        op.normalize()
        op.xbuf = 1

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16  # 2 words * 8 bytes
        finally:
            code.free()

    def test_emit_registers_zero_relocations(self):
        """UnaryOp.emit should register 0 relocations."""
        src = self._make_load(shape=(4, 8), dtype=DTYPE_F32)
        op = UnaryOp(UNARY_SQRT, src)
        op.normalize()
        op.xbuf = 1

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert len(relocs) == 0
        finally:
            code.free()

    def test_emit_raises_for_unsupported_dtype(self):
        """UnaryOp.emit should raise NotImplementedError for INT32 + SQRT."""
        src = self._make_load(shape=(4, 8), dtype=DTYPE_INT32)
        op = UnaryOp(UNARY_SQRT, src)
        op.normalize()
        op.xbuf = 1

        code = Code()
        relocs = []
        try:
            with pytest.raises(NotImplementedError):
                op.emit(code, relocs)
        finally:
            code.free()


# ===================================================================
# BinaryOp -- Batch B expansion (sub, mul, div, max, min + fp16)
# ===================================================================

class TestBinaryOpBatchB:
    """Tests for expanded binary ops: sub, mul, div, max, min for fp32 and fp16."""

    def _make_loads(self, shape=(4, 8), dtype=DTYPE_F32):
        lhs = NDLoad(io_index=0, shape=shape, dtype=dtype)
        lhs.normalize()
        lhs.xbuf = 0
        rhs = NDLoad(io_index=1, shape=shape, dtype=dtype)
        rhs.normalize()
        rhs.xbuf = 1
        return lhs, rhs

    # --- constants ---

    def test_bin_sub_constant(self):
        assert BIN_SUB == 7

    def test_bin_mul_constant(self):
        assert BIN_MUL == 8

    def test_bin_div_constant(self):
        assert BIN_DIV == 9

    def test_bin_max_constant(self):
        assert BIN_MAX == 11

    def test_bin_min_constant(self):
        assert BIN_MIN == 12

    # --- sub fp32 ---

    def test_binary_sub_emit_uses_v_sub_fp32(self):
        """BinaryOp(BIN_SUB) with fp32 inputs should emit 2 words (16 bytes)
        and use V_SUB opcode."""
        lhs, rhs = self._make_loads(shape=(4, 8), dtype=DTYPE_F32)
        op = BinaryOp(BIN_SUB, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16  # 2 words
            assert len(relocs) == 0
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_SUB]
        finally:
            code.free()

    # --- mul fp16 ---

    def test_binary_mul_emit_uses_v_mul_fp16(self):
        """BinaryOp(BIN_MUL) with fp16 inputs should emit 2 words (16 bytes)
        and use V_MUL_FP16 opcode."""
        lhs, rhs = self._make_loads(shape=(4, 16), dtype=DTYPE_FP16)
        op = BinaryOp(BIN_MUL, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16  # 2 words
            assert len(relocs) == 0
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_MUL_FP16]
        finally:
            code.free()

    # --- existing add still works ---

    def test_binary_add_fp32_still_works(self):
        """Existing BIN_ADD + fp32 path must remain functional."""
        lhs, rhs = self._make_loads(shape=(4, 8), dtype=DTYPE_F32)
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_ADD]
        finally:
            code.free()

    # --- add fp16 ---

    def test_binary_add_fp16(self):
        """BIN_ADD + fp16 should use V_ADD_FP16."""
        lhs, rhs = self._make_loads(shape=(4, 16), dtype=DTYPE_FP16)
        op = BinaryOp(BIN_ADD, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_ADD_FP16]
        finally:
            code.free()

    # --- div fp32 ---

    def test_binary_div_fp32(self):
        """BIN_DIV + fp32 should use V_DIV."""
        lhs, rhs = self._make_loads(shape=(4, 8), dtype=DTYPE_F32)
        op = BinaryOp(BIN_DIV, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_DIV]
        finally:
            code.free()

    # --- max fp32 ---

    def test_binary_max_fp32(self):
        """BIN_MAX + fp32 should use V_MAX."""
        lhs, rhs = self._make_loads(shape=(4, 8), dtype=DTYPE_F32)
        op = BinaryOp(BIN_MAX, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_MAX]
        finally:
            code.free()

    # --- min fp16 ---

    def test_binary_min_fp16(self):
        """BIN_MIN + fp16 should use V_MIN_FP16."""
        lhs, rhs = self._make_loads(shape=(4, 16), dtype=DTYPE_FP16)
        op = BinaryOp(BIN_MIN, lhs, rhs)
        op.normalize()
        op.xbuf = 2

        code = Code()
        relocs = []
        try:
            op.emit(code, relocs)
            assert code.size == 16
            head = code.read_u64_at(0)
            opcode = (head >> isa.V_HEAD_ID_OFFSET) & isa.V_HEAD_ID_MASK
            assert opcode == isa.SIMD_FUNC_OFFSET[isa.V_MIN_FP16]
        finally:
            code.free()


# ===================================================================
# BinaryScalarOp -- construction, normalize, emit (Batch C Task 3)
# ===================================================================

def test_binary_scalar_normalize_preserves_shape_and_dtype():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = BinaryScalarOp(op_type=BINS_ADD, src=x, scalar=1.0)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_F32


def test_binary_scalar_emit_appends_two_words():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    x.normalize(); x.xbuf = 0x200
    op = BinaryScalarOp(op_type=BINS_ADD, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400
    code = Code(capacity=4096)
    try:
        op.emit(code, [])
        assert code.size == 16
    finally:
        code.free()


def test_binary_scalar_emit_raises_for_unsupported_dtype():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_INT32)
    x.normalize(); x.xbuf = 0x200
    op = BinaryScalarOp(op_type=BINS_DIV, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400
    code = Code(capacity=4096)
    try:
        with pytest.raises(NotImplementedError):
            op.emit(code, [])
    finally:
        code.free()


# ===================================================================
# CompareOp tests  (Batch D)
# ===================================================================

def test_compare_op_normalize_sets_bool_dtype():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_BOOL


def test_compare_op_workspace_slots_is_one():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    assert op.workspace_slots() == 1


def test_compare_op_emit_appends_two_words():
    a = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(4, 8), dtype=DTYPE_F32)
    a.normalize(); b.normalize()
    a.xbuf = 0x200; b.xbuf = 0x400
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    op.normalize(); op.xbuf = 0x600; op.workspace_xbuf = 0x800
    code = Code(capacity=4096)
    try:
        op.emit(code, [])
        assert code.size == 16
    finally:
        code.free()


def test_compare_op_shape_mismatch_raises():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(16, 16), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    with pytest.raises(ValueError):
        op.normalize()


# ===================================================================
# CompareScalarOp tests  (Batch D)
# ===================================================================

def test_compare_scalar_op_normalize_sets_bool_dtype():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_BOOL


def test_compare_scalar_op_workspace_slots_is_one():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    assert op.workspace_slots() == 1


def test_compare_scalar_op_emit_appends_three_words():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    x.normalize(); x.xbuf = 0x200
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400; op.workspace_xbuf = 0x600
    code = Code(capacity=4096)
    try:
        op.emit(code, [])
        assert code.size == 24  # 3 words * 8 bytes
    finally:
        code.free()


def test_compare_scalar_op_unsupported_dtype_raises():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_INT32)
    x.normalize(); x.xbuf = 0x200
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400; op.workspace_xbuf = 0x600
    code = Code(capacity=4096)
    try:
        with pytest.raises(NotImplementedError):
            op.emit(code, [])
    finally:
        code.free()
