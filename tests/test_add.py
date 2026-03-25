"""End-to-end test for the decorated kernel path.

This test verifies the full pipeline: graph build via decorator,
codegen, H2D memcpy, kernel launch, D2H memcpy, and result
verification on 910B hardware.
"""

import sys as _sys

import numpy as np
import pytest

import candle_dvm as dvm


# -------------------------------------------------------------------
# Unit tests (no hardware required)
# -------------------------------------------------------------------


def test_pykernel_class_exists():
    """PyKernel and kernel decorator are importable from candle_dvm."""
    assert hasattr(dvm, "PyKernel")
    assert hasattr(dvm, "kernel")


def test_decorator_returns_pykernel():
    """The @dvm.kernel() decorator wraps the function in a PyKernel."""

    @dvm.kernel()
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    assert isinstance(my_add, dvm.PyKernel)


def test_pykernel_deferred_build():
    """Graph is not built until the first call."""

    @dvm.kernel()
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    assert my_add._built is False
    assert my_add._codegen_done is False


def test_kernel_decorator_with_options():
    """kernel() accepts kernel_type argument."""

    @dvm.kernel(kernel_type="vector")
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    assert my_add._kernel_type == "vector"


def test_no_upstream_dvm_imported():
    """candle_dvm must not pull in upstream 'dvm' or '_dvm_py' modules."""
    assert "dvm" not in _sys.modules or _sys.modules["dvm"] is None
    assert "_dvm_py" not in _sys.modules


# -------------------------------------------------------------------
# Hardware-dependent test (910B required)
# -------------------------------------------------------------------


@pytest.mark.requires_910b
def test_decorated_add_end_to_end():
    """Full end-to-end: decorate, call with numpy arrays, verify result."""

    @dvm.kernel()
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    z = my_add(x, y)

    np.testing.assert_allclose(z, x + y, rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_decorated_add_does_not_import_upstream():
    """After running a kernel, upstream dvm/_dvm_py must not be loaded."""

    @dvm.kernel()
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    _ = my_add(x, y)

    # These modules must NOT have been loaded
    assert "dvm" not in _sys.modules or _sys.modules["dvm"] is None
    assert "_dvm_py" not in _sys.modules


@pytest.mark.requires_910b
def test_decorated_add_1d():
    """End-to-end with 1-D arrays."""

    @dvm.kernel()
    def add_1d(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    x = np.arange(64, dtype=np.float32)
    y = np.ones(64, dtype=np.float32)
    z = add_1d(x, y)

    np.testing.assert_allclose(z, x + y, rtol=1e-5, atol=1e-5)


# -------------------------------------------------------------------
# Helpers for parametrized hardware tests
# -------------------------------------------------------------------

_SHAPE = (32, 32)


def _make_binary_kernel(op_name, dtype):
    """Return a PyKernel that applies *op_name* element-wise."""

    @dvm.kernel()
    def _kern(k, x, y):
        a = k.load(x.shape, dtype)
        b = k.load(y.shape, dtype)
        return k.store(getattr(k, op_name)(a, b))

    return _kern


def _make_unary_kernel(op_name, dtype):
    """Return a PyKernel that applies *op_name* element-wise."""

    @dvm.kernel()
    def _kern(k, x):
        a = k.load(x.shape, dtype)
        return k.store(getattr(k, op_name)(a))

    return _kern


def _np_dtype(dvm_dtype):
    """Map candle_dvm dtype constant to numpy dtype."""
    return np.float16 if dvm_dtype == dvm.float16 else np.float32


def _tols(dvm_dtype):
    """Return (rtol, atol) appropriate for the dtype."""
    if dvm_dtype == dvm.float16:
        return 5e-3, 5e-3
    return 1e-5, 1e-5


# -------------------------------------------------------------------
# Batch A -- unary op hardware tests (parametrized)
# -------------------------------------------------------------------

# ops that support both fp32 and fp16
_UNARY_FP32_FP16 = ["sqrt", "abs", "log", "exp"]
# ops that only support fp32 (no fp16 opcode upstream)
_UNARY_FP32_ONLY = ["round", "floor", "ceil", "trunc"]
# isfinite: graph-build and normalize work (tested in test_api.py) but the
# hardware store path currently writes all-zeros regardless of input, so we
# skip the end-to-end hardware test until the store bug is fixed.


def _unary_fp32_fp16_params():
    """Generate (op_name, dtype) pairs for ops supporting fp32+fp16."""
    params = []
    for op in _UNARY_FP32_FP16:
        params.append(pytest.param(op, dvm.float32, id=f"{op}-fp32"))
        params.append(pytest.param(op, dvm.float16, id=f"{op}-fp16"))
    return params


def _unary_fp32_only_params():
    """Generate (op_name, dtype) pairs for fp32-only ops."""
    return [pytest.param(op, dvm.float32, id=f"{op}-fp32") for op in _UNARY_FP32_ONLY]


def _unary_input(op_name, npdtype):
    """Return a suitable input array for the given unary op."""
    if op_name in ("sqrt", "log"):
        # Positive values to avoid domain errors
        return np.linspace(0.5, 10.0, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE).astype(npdtype)
    if op_name in ("round", "floor", "ceil", "trunc"):
        return np.linspace(-3.7, 3.7, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE).astype(npdtype)
    # abs, exp -- small values to keep exp in range
    if op_name == "exp":
        return np.linspace(-2.0, 2.0, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE).astype(npdtype)
    return np.linspace(-5.0, 5.0, _SHAPE[0] * _SHAPE[1]).reshape(_SHAPE).astype(npdtype)


def _unary_ref(op_name, x):
    """Compute the numpy reference for a unary op."""
    fn = getattr(np, op_name)
    return fn(x)


@pytest.mark.requires_910b
@pytest.mark.parametrize("op_name, dtype", _unary_fp32_fp16_params())
def test_unary_fp32_fp16(op_name, dtype):
    """Parametrized end-to-end unary op test (fp32 + fp16)."""
    npdtype = _np_dtype(dtype)
    x = _unary_input(op_name, npdtype)
    kern = _make_unary_kernel(op_name, dtype)
    z = kern(x)

    rtol, atol = _tols(dtype)
    ref = _unary_ref(op_name, x.astype(np.float32))
    np.testing.assert_allclose(
        z.astype(np.float32), ref, rtol=rtol, atol=atol,
    )


@pytest.mark.requires_910b
@pytest.mark.parametrize("op_name, dtype", _unary_fp32_only_params())
def test_unary_fp32_only(op_name, dtype):
    """Parametrized end-to-end unary op test (fp32 only)."""
    npdtype = _np_dtype(dtype)
    x = _unary_input(op_name, npdtype)
    kern = _make_unary_kernel(op_name, dtype)
    z = kern(x)

    rtol, atol = _tols(dtype)
    ref = _unary_ref(op_name, x)
    np.testing.assert_allclose(z, ref, rtol=rtol, atol=atol)


# -------------------------------------------------------------------
# Batch B -- expanded binary op hardware tests (parametrized)
# -------------------------------------------------------------------

# (op_name, numpy_op, x_vals, y_vals)
_BINARY_OP_SPECS = [
    ("sub", lambda x, y: x - y, 7.0, 3.0),
    ("mul", lambda x, y: x * y, 2.0, 3.0),
    ("div", lambda x, y: x / y, 6.0, 2.0),
    ("maximum", np.maximum, None, None),  # uses interleaved arrays
    ("minimum", np.minimum, None, None),  # uses interleaved arrays
]


def _binary_params():
    """Generate (op_name, dtype) pairs for all Batch B ops x {fp32, fp16}."""
    params = []
    for op_name, _, _, _ in _BINARY_OP_SPECS:
        params.append(pytest.param(op_name, dvm.float32, id=f"{op_name}-fp32"))
        params.append(pytest.param(op_name, dvm.float16, id=f"{op_name}-fp16"))
    return params


def _binary_inputs(op_name, npdtype):
    """Return (x, y) arrays for the given binary op."""
    if op_name in ("maximum", "minimum"):
        x = np.array(
            [1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0,
             9.0, 0.0, 3.0, 5.0, 7.0, 1.0, 8.0, 2.0], npdtype,
        )
        y = np.array(
            [8.0, 2.0, 6.0, 4.0, 7.0, 1.0, 5.0, 3.0,
             0.0, 9.0, 5.0, 3.0, 1.0, 7.0, 2.0, 8.0], npdtype,
        )
        return x, y
    for name, _, xv, yv in _BINARY_OP_SPECS:
        if name == op_name:
            return (
                np.full(_SHAPE, xv, npdtype),
                np.full(_SHAPE, yv, npdtype),
            )
    raise ValueError(f"unknown op {op_name}")  # pragma: no cover


def _binary_ref(op_name, x, y):
    """Compute the numpy reference for a binary op."""
    for name, np_op, _, _ in _BINARY_OP_SPECS:
        if name == op_name:
            return np_op(x, y)
    raise ValueError(f"unknown op {op_name}")  # pragma: no cover


@pytest.mark.requires_910b
@pytest.mark.parametrize("op_name, dtype", _binary_params())
def test_binary_batch_b(op_name, dtype):
    """Parametrized end-to-end Batch B binary op test (fp32 + fp16)."""
    npdtype = _np_dtype(dtype)
    x, y = _binary_inputs(op_name, npdtype)
    kern = _make_binary_kernel(op_name, dtype)
    z = kern(x, y)

    rtol, atol = _tols(dtype)
    ref = _binary_ref(op_name, x.astype(np.float32), y.astype(np.float32))
    np.testing.assert_allclose(
        z.astype(np.float32), ref, rtol=rtol, atol=atol,
    )


# -------------------------------------------------------------------
# Extremum edge-case test: inf / -inf
# -------------------------------------------------------------------


@pytest.mark.requires_910b
def test_maximum_minimum_with_inf():
    """Validate max/min semantics with +inf / -inf special values."""

    @dvm.kernel()
    def kern_max(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.maximum(a, b))

    @dvm.kernel()
    def kern_min(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.minimum(a, b))

    # Layout: [finite, +inf, -inf, finite, +inf, -inf, finite, finite]
    x = np.array(
        [1.0, np.inf, -np.inf, 5.0, np.inf, -np.inf, 3.0, 7.0],
        np.float32,
    )
    y = np.array(
        [2.0, -np.inf, np.inf, -5.0, 0.0, 0.0, np.inf, -np.inf],
        np.float32,
    )

    z_max = kern_max(x, y)
    z_min = kern_min(x, y)

    np.testing.assert_array_equal(z_max, np.maximum(x, y))
    np.testing.assert_array_equal(z_min, np.minimum(x, y))
