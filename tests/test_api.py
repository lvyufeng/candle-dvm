"""Tests for the public Kernel graph-builder API (candle_dvm.api)."""

from candle_dvm import Kernel, float32
from candle_dvm.ops import DTYPE_BOOL
import pytest


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


def test_kernel_exposes_sqrt_and_log_methods():
    """Build a unary graph with sqrt and log and verify shapes propagate."""
    k = Kernel()
    a = k.load((16, 16), float32)
    s = k.sqrt(a)
    assert s.shape_ref == (16, 16)
    l = k.log(a)
    assert l.shape_ref == (16, 16)
    # Also verify the other unary methods exist and return correct shapes
    assert k.abs(a).shape_ref == (16, 16)
    assert k.exp(a).shape_ref == (16, 16)
    assert k.round(a).shape_ref == (16, 16)
    assert k.floor(a).shape_ref == (16, 16)
    assert k.ceil(a).shape_ref == (16, 16)
    assert k.trunc(a).shape_ref == (16, 16)


def test_kernel_exposes_isfinite_method():
    """isfinite should produce a node with DTYPE_BOOL after normalize."""
    k = Kernel()
    a = k.load((8, 8), float32)
    f = k.isfinite(a)
    assert f.shape_ref == (8, 8)
    # After codegen (which calls normalize), type_id should be DTYPE_BOOL
    k.store(f)
    k.codegen()
    assert f.type_id == DTYPE_BOOL


def test_float16_is_exported():
    """float16 should be importable and equal to DTYPE_FP16."""
    import candle_dvm as dvm
    from candle_dvm.ops import DTYPE_FP16
    assert hasattr(dvm, "float16")
    assert dvm.float16 == DTYPE_FP16


def test_reciprocal_is_not_exposed():
    """reciprocal should not be on the public Kernel API."""
    import candle_dvm as dvm
    assert not hasattr(dvm.Kernel, "reciprocal")


def test_logical_not_is_not_exposed():
    """logical_not should not be on the public Kernel API."""
    import candle_dvm as dvm
    assert not hasattr(dvm.Kernel, "logical_not")


def test_kernel_exposes_batch_b_binary_methods():
    """All Batch B binary methods return nodes with correct shapes."""
    k = Kernel()
    a = k.load((16, 16), float32)
    b = k.load((16, 16), float32)

    s = k.sub(a, b)
    assert s.shape_ref == (16, 16)

    m = k.mul(a, b)
    assert m.shape_ref == (16, 16)

    d = k.div(a, b)
    assert d.shape_ref == (16, 16)

    mx = k.maximum(a, b)
    assert mx.shape_ref == (16, 16)

    mn = k.minimum(a, b)
    assert mn.shape_ref == (16, 16)


def test_add_still_works():
    """Ensure add path still works after Batch B changes."""
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    out = k.add(a, b)
    assert out.shape_ref == (32, 32)
    k.store(out)
    k.codegen()
    assert len(k.get_relocs()) == 3


# ---------------------------------------------------------------
# Task 4: Kernel scalar routing tests
# ---------------------------------------------------------------

def test_kernel_add_accepts_tensor_scalar():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.add(x, 1.0)
    assert y.shape_ref == (32, 32)


def test_kernel_mul_accepts_tensor_scalar():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.mul(x, 2.0)
    assert y.shape_ref == (32, 32)


def test_kernel_div_accepts_tensor_scalar():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.div(x, 2.0)
    assert y.shape_ref == (32, 32)


def test_kernel_maximum_accepts_tensor_scalar():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.maximum(x, 0.0)
    assert y.shape_ref == (32, 32)


def test_kernel_add_accepts_scalar_left_commutative():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.add(1.0, x)
    assert y.shape_ref == (32, 32)


def test_kernel_div_scalar_left_raises():
    k = Kernel()
    x = k.load((32, 32), float32)
    with pytest.raises(NotImplementedError):
        k.div(1.0, x)


def test_kernel_both_scalars_raises():
    k = Kernel()
    with pytest.raises(TypeError):
        k.add(1.0, 2.0)


def test_kernel_does_not_expose_binary_dispatch():
    k = Kernel()
    assert not hasattr(k, "_binary_dispatch")


# ===================================================================
# Compare API tests  (Batch D)
# ===================================================================

def test_kernel_equal_tensor_tensor_returns_bool_node():
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    y = k.equal(a, b)
    k.store(y)
    k.codegen()
    assert y.shape_ref == (32, 32)
    assert y.type_id == DTYPE_BOOL


def test_kernel_greater_tensor_scalar_returns_bool_node():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.greater(x, 1.0)
    k.store(y)
    k.codegen()
    assert y.type_id == DTYPE_BOOL


def test_kernel_less_scalar_left_rewrites():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.less(1.0, x)
    k.store(y)
    k.codegen()
    assert y.type_id == DTYPE_BOOL


def test_kernel_compare_both_scalars_raises_type_error():
    k = Kernel()
    with pytest.raises(TypeError):
        k.equal(1.0, 2.0)


def test_kernel_all_six_compare_methods_exist():
    k = Kernel()
    for name in ("equal", "not_equal", "greater", "greater_equal", "less", "less_equal"):
        assert hasattr(k, name)
