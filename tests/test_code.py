"""Tests for candle_dvm.code -- Code buffer and relocation support."""

import math
from candle_dvm import isa
from candle_dvm.code import Code, RelocAddr


# ---------------------------------------------------------------------------
# gen_entry_v -- entry word generation
# ---------------------------------------------------------------------------

def test_gen_entry_v_low_bits_are_v_entry_type_v():
    """Low 3 bits of the entry word must equal V_ENTRY_TYPE_V (== 0)."""
    entry = Code.gen_entry_v(tile_num=8, block_dim=4, data_size=32)
    assert (entry & isa.V_ENTRY_MASK_TYPE) == isa.V_ENTRY_TYPE_V


def test_gen_entry_v_code_size_field():
    """data_size is stored as data_size//8 in the code-size field."""
    entry = Code.gen_entry_v(tile_num=8, block_dim=4, data_size=32)
    code_size = (entry >> isa.V_ENTRY_CODE_SIZE_OFFSET) & ((1 << isa.V_ENTRY_CODE_SIZE_BITS) - 1)
    assert code_size == 32 // 8


def test_gen_entry_v_block_tile_block_tail_packing():
    """Verify block_tile and block_tail from the DVM formula:

        block_tile = ceil(tile_num / block_dim)
        block_tail = block_dim * block_tile - tile_num

    For tile_num=10, block_dim=4:
        block_tile = ceil(10/4) = 3
        block_tail = 4*3 - 10 = 2
    """
    tile_num, block_dim, data_size = 10, 4, 64
    entry = Code.gen_entry_v(tile_num=tile_num, block_dim=block_dim, data_size=data_size)

    expected_block_tile = math.ceil(tile_num / block_dim)  # 3
    expected_block_tail = block_dim * expected_block_tile - tile_num  # 2

    actual_block_tail = (entry >> isa.V_ENTRY_V_TILE_TAIL_OFFSET) & ((1 << isa.V_ENTRY_V_TILE_TAIL_BITS) - 1)
    actual_block_tile = (entry >> isa.V_ENTRY_V_TILE_BODY_OFFSET) & ((1 << isa.V_ENTRY_V_TILE_BODY_BITS) - 1)

    assert actual_block_tile == expected_block_tile
    assert actual_block_tail == expected_block_tail


def test_gen_entry_v_exact_value():
    """Full bit-exact check for a known set of parameters.

    tile_num=8, block_dim=4, data_size=32:
        block_tile = ceil(8/4) = 2
        block_tail = 4*2 - 8 = 0
        entry = 0 | (4 << 8) | (0 << 32) | (2 << 40)
    """
    entry = Code.gen_entry_v(tile_num=8, block_dim=4, data_size=32)
    expected = (
        isa.V_ENTRY_TYPE_V
        | ((32 // 8) << isa.V_ENTRY_CODE_SIZE_OFFSET)
        | (0 << isa.V_ENTRY_V_TILE_TAIL_OFFSET)
        | (2 << isa.V_ENTRY_V_TILE_BODY_OFFSET)
    )
    assert entry == expected


def test_gen_entry_v_non_divisible():
    """When tile_num is not divisible by block_dim, tail must be non-zero."""
    entry = Code.gen_entry_v(tile_num=7, block_dim=3, data_size=24)
    # block_tile = ceil(7/3) = 3
    # block_tail = 3*3 - 7 = 2
    block_tail = (entry >> isa.V_ENTRY_V_TILE_TAIL_OFFSET) & ((1 << isa.V_ENTRY_V_TILE_TAIL_BITS) - 1)
    block_tile = (entry >> isa.V_ENTRY_V_TILE_BODY_OFFSET) & ((1 << isa.V_ENTRY_V_TILE_BODY_BITS) - 1)
    assert block_tile == 3
    assert block_tail == 2


# ---------------------------------------------------------------------------
# Code buffer -- basic operations
# ---------------------------------------------------------------------------

def test_code_default_capacity():
    """Code() creates a buffer with default capacity of 4096 bytes."""
    c = Code()
    try:
        assert c.capacity == 4096
        assert c.size == 0
    finally:
        c.free()


def test_code_custom_capacity():
    c = Code(capacity=8192)
    try:
        assert c.capacity == 8192
    finally:
        c.free()


def test_append_and_read_u64():
    c = Code()
    try:
        c.append_u64(0xDEADBEEFCAFEBABE)
        c.append_u64(42)
        assert c.size == 16  # 2 words x 8 bytes
        assert c.read_u64_at(0) == 0xDEADBEEFCAFEBABE
        assert c.read_u64_at(8) == 42
    finally:
        c.free()


# ---------------------------------------------------------------------------
# bind_relocs -- relocation patching
# ---------------------------------------------------------------------------

def test_bind_relocs_overwrites_slot():
    """bind_relocs should overwrite the entire 64-bit slot at the reloc offset."""
    c = Code()
    try:
        # Write a placeholder value
        c.append_u64(0)           # offset 0: ffts_addr placeholder
        c.append_u64(0)           # offset 8: entry placeholder
        c.append_u64(0xAAAAAAAA)  # offset 16: will be patched

        reloc = RelocAddr(offset=16)
        c.bind_relocs([reloc], [0xDEADBEEF12345678])

        assert c.read_u64_at(16) == 0xDEADBEEF12345678
        # Other slots unchanged
        assert c.read_u64_at(0) == 0
        assert c.read_u64_at(8) == 0
    finally:
        c.free()


def test_bind_relocs_multiple():
    """bind_relocs patches all given relocations."""
    c = Code()
    try:
        c.append_u64(0)  # 0
        c.append_u64(0)  # 8
        c.append_u64(0)  # 16

        relocs = [RelocAddr(offset=0), RelocAddr(offset=16)]
        values = [0x1111, 0x2222]
        c.bind_relocs(relocs, values)

        assert c.read_u64_at(0) == 0x1111
        assert c.read_u64_at(8) == 0
        assert c.read_u64_at(16) == 0x2222
    finally:
        c.free()


# ---------------------------------------------------------------------------
# debug_header -- introspection helper
# ---------------------------------------------------------------------------

def test_debug_header_returns_dict():
    """debug_header() must return a dict with target, block_dim, data_size."""
    c = Code()
    try:
        c.target = "vector"
        c.block_dim = 4
        c.data_size = 32
        hdr = c.debug_header()
        assert isinstance(hdr, dict)
        assert hdr["target"] == "vector"
        assert hdr["block_dim"] == 4
        assert hdr["data_size"] == 32
    finally:
        c.free()


def test_debug_header_defaults():
    """Freshly created Code should have sensible defaults."""
    c = Code()
    try:
        hdr = c.debug_header()
        assert "target" in hdr
        assert "block_dim" in hdr
        assert "data_size" in hdr
    finally:
        c.free()


# ---------------------------------------------------------------------------
# RelocAddr
# ---------------------------------------------------------------------------

def test_reloc_addr_stores_offset():
    r = RelocAddr(offset=24)
    assert r.offset == 24


def test_reloc_addr_repr():
    r = RelocAddr(offset=16)
    assert "16" in repr(r)


# ---------------------------------------------------------------------------
# Negative-path tests
# ---------------------------------------------------------------------------

def test_append_past_capacity_raises_overflow():
    """Appending past the buffer capacity must raise OverflowError."""
    c = Code(capacity=8)  # only room for one u64
    try:
        c.append_u64(0)  # fills the buffer
        import pytest
        with pytest.raises(OverflowError):
            c.append_u64(1)  # should overflow
    finally:
        c.free()


def test_bind_relocs_bad_offset_raises_index_error():
    """bind_relocs with an out-of-range offset must raise IndexError."""
    import pytest
    c = Code()
    try:
        c.append_u64(0)  # offset 0 only
        reloc = RelocAddr(offset=8)  # beyond written size
        with pytest.raises(IndexError):
            c.bind_relocs([reloc], [0xDEAD])
    finally:
        c.free()


def test_bind_relocs_mismatched_lengths_raises_value_error():
    """bind_relocs with mismatched relocs/values lengths must raise ValueError."""
    import pytest
    c = Code()
    try:
        c.append_u64(0)
        reloc = RelocAddr(offset=0)
        with pytest.raises(ValueError):
            c.bind_relocs([reloc], [0xDEAD, 0xBEEF])  # 1 reloc, 2 values
    finally:
        c.free()


def test_read_u64_at_empty_buffer_raises_index_error():
    """read_u64_at on an empty buffer must raise IndexError with a clear message."""
    import pytest
    c = Code()
    try:
        with pytest.raises(IndexError, match="Code buffer is empty"):
            c.read_u64_at(0)
    finally:
        c.free()


def test_read_u64_at_unaligned_offset_raises_value_error():
    """read_u64_at with an unaligned offset must raise ValueError."""
    import pytest
    c = Code()
    try:
        c.append_u64(0xABCDEF)
        with pytest.raises(ValueError):
            c.read_u64_at(3)  # not 8-byte aligned
    finally:
        c.free()
