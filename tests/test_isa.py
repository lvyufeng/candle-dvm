"""Tests for candle_dvm.isa -- ISA constants and encode helpers."""

from candle_dvm import isa


# ---------------------------------------------------------------------------
# Access (load/store) opcode constants  -- vAccInsnID in isa.h
# ---------------------------------------------------------------------------

def test_access_opcode_constants_match_upstream():
    assert isa.V_LOAD == 0
    assert isa.V_STORE == 10


def test_access_opcode_enum_ordering():
    """Verify a broader slice of vAccInsnID values."""
    assert isa.V_LOAD == 0
    assert isa.V_LOAD_DUMMY == 1
    assert isa.V_LOAD_VIEW == 2
    assert isa.V_SLOAD == 3
    assert isa.V_LOAD_CC == 4
    assert isa.V_MULTI_LOAD == 5
    assert isa.V_PINGPONG_LOAD == 6
    assert isa.V_PINGPONG_PEER_LOAD == 7
    assert isa.V_PEER_LOAD == 8
    assert isa.V_PEER_LOAD_MIX == 9
    assert isa.V_STORE == 10
    assert isa.V_STORE_ATOMIC == 11
    assert isa.V_STORE_COND == 12
    assert isa.V_SSTORE == 13
    assert isa.V_SLICE_STORE == 14
    assert isa.V_STORE_AG == 15
    assert isa.V_STORE_RS == 16
    assert isa.V_PEER_STORE == 17
    assert isa.V_PEER_STORE_MIX == 18
    assert isa.V_ACCESS_NONE == 19


# ---------------------------------------------------------------------------
# SIMD opcode constants  -- vSimdInsnID in isa.h
# ---------------------------------------------------------------------------

def test_simd_opcode_constants_match_upstream():
    assert isa.V_ADD == 18
    assert isa.V_COPY == 0


def test_simd_opcode_enum_ordering():
    """Verify a broader slice of vSimdInsnID values."""
    assert isa.V_COPY == 0
    assert isa.V_COPY_CUBE_TILE == 1
    assert isa.V_NOP == 2
    assert isa.V_BROADCAST_Y == 3
    assert isa.V_BROADCAST_S == 4
    assert isa.V_SQRT == 5
    assert isa.V_ABS == 6
    assert isa.V_LOG == 7
    assert isa.V_EXP == 8
    assert isa.V_ADD == 18
    assert isa.V_SUB == 19
    assert isa.V_MUL == 20
    assert isa.V_NONE == 110  # last entry in vSimdInsnID


# ---------------------------------------------------------------------------
# Bitfield offset / mask constants
# ---------------------------------------------------------------------------

def test_bitfield_offset_constants():
    assert isa.V_HEAD_SIMD_FLAG_OFFSET == 0
    assert isa.V_HEAD_ID_OFFSET == 48
    assert isa.V_HEAD_EXT_OFFSET == 22
    assert isa.V_HEAD_SIZE_OFFSET == 7
    assert isa.V_M_HEAD_EXT_OFFSET == 14
    assert isa.V_M_HEAD_SIZE_OFFSET == 4


def test_entry_constants():
    assert isa.V_ENTRY_TYPE_V == 0
    assert isa.V_ENTRY_CODE_SIZE_OFFSET == 8
    assert isa.V_ENTRY_V_TILE_TAIL_OFFSET == 32
    assert isa.V_ENTRY_V_TILE_BODY_OFFSET == 40


def test_mask_constants():
    assert isa.V_X_MASK == 0x3FFFF  # 18-bit mask


# ---------------------------------------------------------------------------
# make_acc_head  -- corresponds to vMakeAccHead (without func_offset lookup)
# ---------------------------------------------------------------------------

def test_make_acc_head_packs_fields():
    head = isa.make_acc_head(isa.V_LOAD, 3, 2)
    hw_id = isa.ACCESS_FUNC_OFFSET[isa.V_LOAD]
    expected = (hw_id << isa.V_HEAD_ID_OFFSET) | \
               (3 << isa.V_M_HEAD_EXT_OFFSET) | \
               (2 << isa.V_M_HEAD_SIZE_OFFSET)
    assert head == expected


def test_make_acc_head_zero():
    head = isa.make_acc_head(0, 0, 0)
    hw_id = isa.ACCESS_FUNC_OFFSET[0]
    expected = hw_id << isa.V_HEAD_ID_OFFSET
    assert head == expected


def test_make_acc_head_large_ext():
    head = isa.make_acc_head(isa.V_STORE, 0x3FFFFFFFF, 0xF)
    # ext is 34 bits, size is 4 bits for load/store
    hw_id = isa.ACCESS_FUNC_OFFSET[isa.V_STORE]
    id_part = hw_id << isa.V_HEAD_ID_OFFSET
    ext_part = 0x3FFFFFFFF << isa.V_M_HEAD_EXT_OFFSET
    size_part = 0xF << isa.V_M_HEAD_SIZE_OFFSET
    assert head == (id_part | ext_part | size_part)


# ---------------------------------------------------------------------------
# make_simd_head  -- corresponds to vMakeSimdHead (without func_offset lookup)
# ---------------------------------------------------------------------------

def test_make_simd_head_packs_fields():
    head = isa.make_simd_head(isa.V_ADD, 5, 2)
    hw_id = isa.SIMD_FUNC_OFFSET[isa.V_ADD]
    expected = (hw_id << isa.V_HEAD_ID_OFFSET) | \
               (5 << isa.V_HEAD_EXT_OFFSET) | \
               (2 << isa.V_HEAD_SIZE_OFFSET) | \
               (1 << isa.V_HEAD_SIMD_FLAG_OFFSET)
    assert head == expected


def test_make_simd_head_has_simd_flag_set():
    head = isa.make_simd_head(0, 0, 0)
    assert (head & 1) == 1  # SIMD flag at bit 0


def test_make_simd_head_zero_opcode():
    head = isa.make_simd_head(0, 0, 1)
    hw_id = isa.SIMD_FUNC_OFFSET[0]
    expected = (hw_id << isa.V_HEAD_ID_OFFSET) | \
               (1 << isa.V_HEAD_SIZE_OFFSET) | (1 << isa.V_HEAD_SIMD_FLAG_OFFSET)
    assert head == expected


# ---------------------------------------------------------------------------
# Unary op constants  -- match UnaryType in dvm.h
# ---------------------------------------------------------------------------

def test_unary_op_constants():
    """Verify unary op enum indices match upstream UnaryType."""
    assert isa.UNARY_SQRT == 0
    assert isa.UNARY_ABS == 1
    assert isa.UNARY_LOG == 2
    assert isa.UNARY_EXP == 3
    assert isa.UNARY_ISFINITE == 5
    assert isa.UNARY_ROUND == 7
    assert isa.UNARY_FLOOR == 8
    assert isa.UNARY_CEIL == 9
    assert isa.UNARY_TRUNC == 10


# ---------------------------------------------------------------------------
# Dtype constants
# ---------------------------------------------------------------------------

def test_dtype_constants():
    """Verify dtype constants match upstream DataType enum."""
    assert isa.DTYPE_F32 == 3
    assert isa.DTYPE_FP16 == 1


# ---------------------------------------------------------------------------
# Binary op constants  -- match BinaryType in dvm.h
# ---------------------------------------------------------------------------

def test_binary_op_constants():
    """Verify binary op enum indices match upstream BinaryType."""
    assert isa.BIN_ADD == 6
    assert isa.BIN_SUB == 7
    assert isa.BIN_MUL == 8
    assert isa.BIN_DIV == 9
    assert isa.BIN_MAX == 11
    assert isa.BIN_MIN == 12


# ---------------------------------------------------------------------------
# UNARY_OPCODE_TABLE routing tests
# ---------------------------------------------------------------------------

class TestUnaryOpcodeTable:
    """Tests for UNARY_OPCODE_TABLE: (unary_op, dtype) -> SIMD opcode."""

    # -- sqrt --
    def test_sqrt_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_SQRT, isa.DTYPE_F32)] == isa.V_SQRT

    def test_sqrt_fp16(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_SQRT, isa.DTYPE_FP16)] == isa.V_SQRT_FP16

    # -- abs --
    def test_abs_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_ABS, isa.DTYPE_F32)] == isa.V_ABS

    def test_abs_fp16(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_ABS, isa.DTYPE_FP16)] == isa.V_ABS_FP16

    # -- log --
    def test_log_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_LOG, isa.DTYPE_F32)] == isa.V_LOG

    def test_log_fp16(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_LOG, isa.DTYPE_FP16)] == isa.V_LOG_FP16

    # -- exp --
    def test_exp_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_EXP, isa.DTYPE_F32)] == isa.V_EXP

    def test_exp_fp16(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_EXP, isa.DTYPE_FP16)] == isa.V_EXP_FP16

    # -- round (f32 only) --
    def test_round_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_ROUND, isa.DTYPE_F32)] == isa.V_ROUND

    def test_round_fp16_not_present(self):
        assert (isa.UNARY_ROUND, isa.DTYPE_FP16) not in isa.UNARY_OPCODE_TABLE

    # -- floor (f32 only) --
    def test_floor_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_FLOOR, isa.DTYPE_F32)] == isa.V_FLOOR

    def test_floor_fp16_not_present(self):
        assert (isa.UNARY_FLOOR, isa.DTYPE_FP16) not in isa.UNARY_OPCODE_TABLE

    # -- ceil (f32 only) --
    def test_ceil_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_CEIL, isa.DTYPE_F32)] == isa.V_CEIL

    def test_ceil_fp16_not_present(self):
        assert (isa.UNARY_CEIL, isa.DTYPE_FP16) not in isa.UNARY_OPCODE_TABLE

    # -- trunc (f32 only) --
    def test_trunc_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_TRUNC, isa.DTYPE_F32)] == isa.V_TRUNC

    def test_trunc_fp16_not_present(self):
        assert (isa.UNARY_TRUNC, isa.DTYPE_FP16) not in isa.UNARY_OPCODE_TABLE

    # -- isfinite --
    def test_isfinite_f32(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_ISFINITE, isa.DTYPE_F32)] == isa.V_ISFINITE

    def test_isfinite_fp16(self):
        assert isa.UNARY_OPCODE_TABLE[(isa.UNARY_ISFINITE, isa.DTYPE_FP16)] == isa.V_ISFINITE_FP16

    # -- completeness: exactly 14 entries --
    def test_table_size(self):
        """9 unary ops: 5 with fp16 variant (10 entries) + 4 f32-only (4 entries) = 14."""
        assert len(isa.UNARY_OPCODE_TABLE) == 14


# ---------------------------------------------------------------------------
# BINARY_OPCODE_TABLE routing tests
# ---------------------------------------------------------------------------

class TestBinaryOpcodeTable:
    """Tests for BINARY_OPCODE_TABLE: (binary_op, dtype) -> SIMD opcode."""

    # -- add --
    def test_add_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_ADD, isa.DTYPE_F32)] == isa.V_ADD

    def test_add_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_ADD, isa.DTYPE_FP16)] == isa.V_ADD_FP16

    # -- sub --
    def test_sub_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_SUB, isa.DTYPE_F32)] == isa.V_SUB

    def test_sub_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_SUB, isa.DTYPE_FP16)] == isa.V_SUB_FP16

    # -- mul --
    def test_mul_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MUL, isa.DTYPE_F32)] == isa.V_MUL

    def test_mul_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MUL, isa.DTYPE_FP16)] == isa.V_MUL_FP16

    # -- div --
    def test_div_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_DIV, isa.DTYPE_F32)] == isa.V_DIV

    def test_div_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_DIV, isa.DTYPE_FP16)] == isa.V_DIV_FP16

    # -- max --
    def test_max_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MAX, isa.DTYPE_F32)] == isa.V_MAX

    def test_max_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MAX, isa.DTYPE_FP16)] == isa.V_MAX_FP16

    # -- min --
    def test_min_f32(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MIN, isa.DTYPE_F32)] == isa.V_MIN

    def test_min_fp16(self):
        assert isa.BINARY_OPCODE_TABLE[(isa.BIN_MIN, isa.DTYPE_FP16)] == isa.V_MIN_FP16

    # -- completeness: exactly 12 entries (6 ops x 2 dtypes) --
    def test_table_size(self):
        """6 binary ops x 2 dtypes (f32 + fp16) = 12 entries."""
        assert len(isa.BINARY_OPCODE_TABLE) == 12


# ---------------------------------------------------------------------------
# BINARY_SCALAR_OPCODE_TABLE routing
# ---------------------------------------------------------------------------

def test_binary_scalar_opcode_routing_fp32_and_fp16():
    expected = {
        (isa.BINS_ADD, isa.DTYPE_F32): isa.V_ADDS,
        (isa.BINS_ADD, isa.DTYPE_FP16): isa.V_ADDS_FP16,
        (isa.BINS_MUL, isa.DTYPE_F32): isa.V_MULS,
        (isa.BINS_MUL, isa.DTYPE_FP16): isa.V_MULS_FP16,
        (isa.BINS_DIV, isa.DTYPE_F32): isa.V_DIVS,
        (isa.BINS_DIV, isa.DTYPE_FP16): isa.V_DIVS_FP16,
        (isa.BINS_MAX, isa.DTYPE_F32): isa.V_MAXS,
        (isa.BINS_MAX, isa.DTYPE_FP16): isa.V_MAXS_FP16,
        (isa.BINS_MIN, isa.DTYPE_F32): isa.V_MINS,
        (isa.BINS_MIN, isa.DTYPE_FP16): isa.V_MINS_FP16,
    }
    for key, val in expected.items():
        assert isa.BINARY_SCALAR_OPCODE_TABLE[key] == val


# ---------------------------------------------------------------------------
# encode_unary  -- vUnary 2-word instruction encoding
# ---------------------------------------------------------------------------

def test_encode_unary_fp32_matches_vUnary_layout():
    words = isa.encode_unary(opcode=isa.V_SQRT, xd=0x200, xn=0x400, count=32)
    assert len(words) == 2
    # head ext field should contain xd (not xn) for vUnary
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == 0x200
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_SQRT]
    assert words[1] == (0x400 << 32) | 32


# ---------------------------------------------------------------------------
# encode_binary_scalar  -- vBinaryS 2-word instruction encoding
# ---------------------------------------------------------------------------

def test_encode_binary_scalar_matches_vBinaryS_layout():
    # Upstream vBinaryS::Encode uses vCompactX(xd), and upstream isa.h defines
    # vCompactX(x) as x >> 5 for xbuf addresses aligned to 32 bytes.
    words = isa.encode_binary_scalar(opcode=isa.V_ADDS, xn=0x200, xd=0x400, count=32, scalar_bits=0x3F800000)
    assert len(words) == 2
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == 0x200
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_ADDS]
    assert words[1] == (0x3F800000 << 32) | ((0x400 >> 5) << 16) | 32
