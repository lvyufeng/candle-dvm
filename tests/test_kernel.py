"""Tests for candle_dvm.kernel -- VKernelS codegen pipeline.

Verifies:
- codegen produces non-zero entry word
- code.data_size > 16 (more than just the header)
- target == 0 (vector)
- block_dim matches upstream trace fixture
- relocs list has correct count (2 inputs + 1 output = 3)
"""

import re
from pathlib import Path

import pytest

from candle_dvm.kernel import VKernelS
from candle_dvm.ops import NDLoad, NDStore, BinaryOp, FlexOp, CompareOp, DTYPE_F32, DTYPE_BOOL, BIN_ADD, CMP_EQ


# ===================================================================
# Helpers
# ===================================================================

def _trace_block_dim() -> int:
    """Extract block_dim from the first line of the upstream add trace."""
    text = (
        Path(__file__).parent / "fixtures" / "upstream_add_trace.txt"
    ).read_text().splitlines()[0]
    return int(re.search(r"block_dim=(\d+)", text).group(1))


def _build_add_kernel():
    """Build a simple a+b kernel with shape (32,32) fp32."""
    k = VKernelS()
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    c = BinaryOp(op_type=BIN_ADD, lhs=a, rhs=b)
    d = NDStore(io_index=0, src=c)
    for obj in [a, b, c, d]:
        k.append(obj)
    k.normalize()
    k.codegen()
    return k


# ===================================================================
# Tests
# ===================================================================

class TestVKernelSCodegen:

    def test_vkernel_s_codegen_uses_upstream_phase1_block_dim(self):
        """block_dim must match the value captured in the upstream trace."""
        block_dim = _trace_block_dim()
        assert isinstance(block_dim, int) and block_dim > 0
        k = _build_add_kernel()
        assert k.debug_header()["block_dim"] == block_dim

    def test_codegen_produces_nonzero_entry(self):
        """The entry word (word 1 of the code buffer) must be non-zero."""
        k = _build_add_kernel()
        entry = k.code.read_u64_at(8)  # word 1
        assert entry != 0, "entry word must be non-zero after codegen"

    def test_data_size_exceeds_header(self):
        """code.data_size must be greater than 16 (more than just the head)."""
        k = _build_add_kernel()
        assert k.code.data_size > 16, (
            f"data_size={k.code.data_size} should exceed 16-byte header"
        )

    def test_target_is_vector(self):
        """target must be 0 (vector)."""
        k = _build_add_kernel()
        assert k.debug_header()["target"] == 0

    def test_relocs_count(self):
        """relocs list must have 3 entries: 2 loads + 1 store."""
        k = _build_add_kernel()
        assert len(k.relocs) == 3, (
            f"expected 3 relocs (2 inputs + 1 output), got {len(k.relocs)}"
        )

    def test_debug_header_has_tile_num(self):
        """debug_header must include tile_num."""
        k = _build_add_kernel()
        hdr = k.debug_header()
        assert "tile_num" in hdr
        assert hdr["tile_num"] >= 1

    def test_xbuf_store_shares_src(self):
        """NDStore.xbuf must equal its source's xbuf after codegen."""
        k = VKernelS()
        a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
        b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
        c = BinaryOp(op_type=BIN_ADD, lhs=a, rhs=b)
        d = NDStore(io_index=0, src=c)
        for obj in [a, b, c, d]:
            k.append(obj)
        k.normalize()
        k.codegen()
        assert d.xbuf == c.xbuf, (
            f"NDStore.xbuf ({d.xbuf}) must equal src.xbuf ({c.xbuf})"
        )

    def test_xbuf_monotonic_allocation(self):
        """NDLoad and BinaryOp xbufs must be monotonically increasing."""
        k = VKernelS()
        a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
        b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
        c = BinaryOp(op_type=BIN_ADD, lhs=a, rhs=b)
        d = NDStore(io_index=0, src=c)
        for obj in [a, b, c, d]:
            k.append(obj)
        k.normalize()
        k.codegen()
        assert a.xbuf < b.xbuf < c.xbuf, (
            f"xbufs must be monotonically increasing: "
            f"a={a.xbuf}, b={b.xbuf}, c={c.xbuf}"
        )

    def test_xbuf_aligned_to_32(self):
        """All xbuf addresses must be multiples of 32."""
        k = VKernelS()
        a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
        b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
        c = BinaryOp(op_type=BIN_ADD, lhs=a, rhs=b)
        d = NDStore(io_index=0, src=c)
        for obj in [a, b, c, d]:
            k.append(obj)
        k.normalize()
        k.codegen()
        for name, obj in [("a", a), ("b", b), ("c", c), ("d", d)]:
            assert obj.xbuf % 32 == 0, (
                f"{name}.xbuf={obj.xbuf} is not 32-byte aligned"
            )

    def test_tile_num_computed_from_shape(self):
        """tile_num must be 1 for small tensors that fit in local memory.

        Phase 1: no tiling needed for (32, 32) fp32 = 4096 bytes,
        which fits in 910B local memory (192KB - 512B).
        The upstream trace confirms body_tile=1, tail_tile_diff=0.
        """
        k = _build_add_kernel()  # shape (32, 32)
        assert k.debug_header()["tile_num"] == 1, (
            f"expected tile_num=1 for shape (32, 32) in phase 1, "
            f"got {k.debug_header()['tile_num']}"
        )


# ===================================================================
# Workspace allocation tests
# ===================================================================

def test_flexop_workspace_slots_default_zero():
    """FlexOp.workspace_slots() should return 0 by default."""
    op = FlexOp(1, DTYPE_F32, (4, 8))
    assert op.workspace_slots() == 0


def test_flexop_workspace_xbuf_default_zero():
    """FlexOp.workspace_xbuf should be 0 by default."""
    op = FlexOp(1, DTYPE_F32, (4, 8))
    assert op.workspace_xbuf == 0


def test_existing_ops_have_zero_workspace_slots():
    """BinaryOp should not request workspace slots."""
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    op = BinaryOp(op_type=BIN_ADD, lhs=a, rhs=b)
    assert op.workspace_slots() == 0


def test_compare_op_workspace_is_source_dtype_sized():
    """CompareOp result + workspace slots must both be source-dtype-sized.
    For a (32,32) fp32 compare graph, both the compare result slot and the
    workspace slot should be sized as fp32 storage (4096 bytes each), even
    though the logical output dtype is DTYPE_BOOL.
    """
    k = VKernelS()
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    c = CompareOp(cmp_type=CMP_EQ, lhs=a, rhs=b)
    d = NDStore(io_index=0, src=c)
    for obj in [a, b, c, d]:
        k.append(obj)
    k.normalize()
    k.codegen()
    # compare result slot uses source dtype physical storage (fp32 = 4096)
    assert c.workspace_xbuf == c.xbuf + 4096
    # total allocation: a(4096) + b(4096) + c_result(4096) + workspace(4096)
    assert c.workspace_xbuf + 4096 == 0x200 + 4096 + 4096 + 4096 + 4096
