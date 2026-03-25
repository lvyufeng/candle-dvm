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
# Batch A -- unary op hardware tests
# -------------------------------------------------------------------


@pytest.mark.requires_910b
def test_decorated_sqrt_fp32_end_to_end():
    """End-to-end sqrt on fp32 inputs."""

    @dvm.kernel()
    def my_sqrt(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.sqrt(a))

    x = np.full([32, 32], 4.0, np.float32)
    z = my_sqrt(x)
    np.testing.assert_allclose(z, np.sqrt(x), rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_decorated_exp_fp16_end_to_end():
    """End-to-end exp on fp16 inputs."""

    @dvm.kernel()
    def my_exp(k, x):
        a = k.load(x.shape, dvm.float16)
        return k.store(k.exp(a))

    x = np.full([32, 32], 1.0, np.float16)
    z = my_exp(x)
    np.testing.assert_allclose(
        z.astype(np.float32),
        np.exp(x.astype(np.float32)),
        rtol=5e-3, atol=5e-3,
    )


# -------------------------------------------------------------------
# Batch B -- expanded binary op hardware tests
# -------------------------------------------------------------------


@pytest.mark.requires_910b
def test_decorated_mul_fp32_end_to_end():
    """End-to-end mul on fp32 inputs."""

    @dvm.kernel()
    def my_mul(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.mul(a, b))

    x = np.full([32, 32], 2.0, np.float32)
    y = np.full([32, 32], 3.0, np.float32)
    z = my_mul(x, y)
    np.testing.assert_allclose(z, x * y, rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_decorated_div_fp16_end_to_end():
    """End-to-end div on fp16 inputs."""

    @dvm.kernel()
    def my_div(k, x, y):
        a = k.load(x.shape, dvm.float16)
        b = k.load(y.shape, dvm.float16)
        return k.store(k.div(a, b))

    x = np.full([32, 32], 6.0, np.float16)
    y = np.full([32, 32], 2.0, np.float16)
    z = my_div(x, y)
    np.testing.assert_allclose(
        z.astype(np.float32),
        (x / y).astype(np.float32),
        rtol=5e-3, atol=5e-3,
    )


@pytest.mark.requires_910b
def test_decorated_maximum_fp32_end_to_end():
    """End-to-end maximum on fp32 inputs."""

    @dvm.kernel()
    def my_max(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.maximum(a, b))

    x = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0], np.float32)
    y = np.array([8.0, 2.0, 6.0, 4.0, 7.0, 1.0, 5.0, 3.0], np.float32)
    z = my_max(x, y)
    np.testing.assert_allclose(z, np.maximum(x, y), rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_decorated_minimum_fp16_end_to_end():
    """End-to-end minimum on fp16 inputs."""

    @dvm.kernel()
    def my_min(k, x, y):
        a = k.load(x.shape, dvm.float16)
        b = k.load(y.shape, dvm.float16)
        return k.store(k.minimum(a, b))

    x = np.array(
        [1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0,
         9.0, 0.0, 3.0, 5.0, 7.0, 1.0, 8.0, 2.0], np.float16,
    )
    y = np.array(
        [8.0, 2.0, 6.0, 4.0, 7.0, 1.0, 5.0, 3.0,
         0.0, 9.0, 5.0, 3.0, 1.0, 7.0, 2.0, 8.0], np.float16,
    )
    z = my_min(x, y)
    np.testing.assert_allclose(
        z.astype(np.float32),
        np.minimum(x, y).astype(np.float32),
        rtol=5e-3, atol=5e-3,
    )
