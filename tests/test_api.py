"""Tests for the public Kernel graph-builder API (candle_dvm.api)."""

from candle_dvm import Kernel, float32


def test_public_kernel_builds_add_graph():
    """Build a simple add graph and verify output shape."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    out = k.store(k.add(a, b))
    assert out.shape_ref == (32, 32)


def test_kernel_codegen_produces_valid_code():
    """Codegen should produce a header with expected fields."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    k.store(k.add(a, b))
    k.codegen()
    h = k.debug_header()
    assert h["target"] == 0
    assert h["block_dim"] > 0
    assert h["data_size"] > 16


def test_kernel_relocs_count():
    """After codegen, relocs should contain one entry per IO operand."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    k.store(k.add(a, b))
    k.codegen()
    # 2 loads + 1 store = 3 relocs
    assert len(k.get_relocs()) == 3
