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
