# candle-dvm Batch C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Batch C tensor-scalar vector ops to `candle-dvm` (`adds`, `muls`, `divs`, `maximum` scalar, `minimum` scalar) with `fp32` and `fp16` support, and expose them only after portable and 910B end-to-end validation passes.

**Architecture:** Batch C extends the existing vector path by introducing a new `BinaryScalarOp(FlexOp)` cdef class and a dedicated `vBinaryS` ISA encode helper. The work stays inside the proven phase-1/Batch A+B architecture: ISA routing → op emit → public Kernel API → PyKernel hardware tests. Public API exposure stays strict: only completed tensor-scalar ops become callable.

**Tech Stack:** Python 3.12, Cython 3, setuptools, pytest, NumPy, Ascend 910B runtime, DVM-compatible C220 device binary

---

## Scope and boundaries

This plan covers **Batch C only**:

- tensor-scalar `add`
- tensor-scalar `mul`
- tensor-scalar `div`
- tensor-scalar `maximum`
- tensor-scalar `minimum`

for:
- `fp32`
- `fp16`

This plan explicitly does **not** cover:
- scalar-left vs tensor-right canonicalization beyond the minimal one-sided route needed for current API calls
- compare scalar ops (`equal(a, 1.0)` etc.) — Batch D
- select / workspace ops — Batch E+
- broadcast / reduce / cast / copy / bool storage changes

## File map

### Files to modify

- Modify: `candle_dvm/isa.pyx`
  - add Batch C scalar opcode routing table
  - add `encode_binary_scalar()` helper for `vBinaryS`

- Modify: `candle_dvm/isa.pxd`
  - export `encode_binary_scalar()` and any needed constants

- Modify: `candle_dvm/ops.pyx`
  - add Batch C scalar op constants
  - implement `BinaryScalarOp(FlexOp)`
  - add scalar packing helper(s)
  - keep `BinaryOp` unchanged except where API routing now chooses scalar vs tensor path

- Modify: `candle_dvm/ops.pxd`
  - export `BinaryScalarOp`

- Modify: `candle_dvm/api.pyx`
  - route `add`, `mul`, `div`, `maximum`, `minimum` to `BinaryScalarOp` when exactly one operand is scalar and the other is tensor
  - preserve the existing tensor-tensor path through `BinaryOp`

- Modify: `candle_dvm/pykernel.py`
  - no architectural change expected
  - only update if scalar execution exposes a decorator/runtime bug

- Modify: `candle_dvm/__init__.py`
  - no new public exports expected; scalar functionality should appear through existing `Kernel` methods only

- Modify: `tests/test_isa.py`
  - add Batch C scalar routing tests and `encode_binary_scalar()` tests

- Modify: `tests/test_ops.py`
  - add normalize / emit / negative-path tests for `BinaryScalarOp`

- Modify: `tests/test_api.py`
  - add public API tests for tensor-scalar calls

- Modify: `tests/test_add.py`
  - add 910B end-to-end tests for scalar paths

### Files not expected to change

- `candle_dvm/system.pyx`
- `candle_dvm/code.pyx`
- `candle_dvm/kernel.pyx`

If a bug appears there during end-to-end validation, stop and debug root cause before changing architecture.

## Upstream references to read before coding

- `/home/dndx/lvyufeng/dvm/src/ops.cc:67-80`
  - `binarys_id_list`
- `/home/dndx/lvyufeng/dvm/src/isa.h:417-440`
  - `vBinaryS::Encode`
- `/home/dndx/lvyufeng/dvm/src/ops.cc:1345-1354`
  - `BinaryScalarOp::Emit`
- current `candle_dvm/isa.pyx`
- current `candle_dvm/ops.pyx`
- current `candle_dvm/api.pyx`
- current `tests/test_ops.py`
- current `tests/test_add.py`

## Batch C design rules

### Scalar-path encoding

Use `vBinaryS` exactly:

- `pc[0] = make_simd_head(opcode, xn, 2)`
- `pc[1] = scalar_bits << 32 | compact_x(xd) << 16 | count`

### Scalar representation

For phase 2 Batch C, use these minimal scalar encodings:

- `fp32`: pack as raw IEEE754 32-bit bits in `scalar_bits`
- `fp16`: pack as raw 16-bit IEEE754 half bits in the low 16 bits of `scalar_bits`

Do not add int32 scalar support in this batch.

### API policy

Only existing methods become scalar-aware:

- `add`
- `mul`
- `div`
- `maximum`
- `minimum`

No new public method names are introduced.

### Operand policy

For this batch, support exactly one scalar operand and one tensor operand.

If both operands are tensors: use existing `BinaryOp`.
If both operands are scalars: raise `TypeError`.
If scalar is on the left: normalize by swapping into the right-hand scalar position only when the op is commutative (`add`, `mul`, `maximum`, `minimum`).
If scalar is on the left for non-commutative ops (`div`): raise `NotImplementedError` in this batch.

## Task breakdown

### Task 1: Add Batch C scalar opcode routing tables

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing routing tests**

```python
from candle_dvm import isa


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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_isa.py::test_binary_scalar_opcode_routing_fp32_and_fp16 -v
```

Expected: FAIL because the table and constants do not exist yet

- [ ] **Step 3: Add the minimal routing table**

Implement in `candle_dvm/isa.pyx`:
- scalar op constants: `BINS_ADD`, `BINS_MUL`, `BINS_DIV`, `BINS_MAX`, `BINS_MIN`
- `BINARY_SCALAR_OPCODE_TABLE`
- export any needed symbols from `isa.pxd`

Use exact upstream mappings from `dvm/src/ops.cc:67-80`.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_isa.py::test_binary_scalar_opcode_routing_fp32_and_fp16 -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add batch c scalar opcode routing"
```

### Task 2: Add `encode_binary_scalar()` to the ISA layer

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing encode test**

```python
from candle_dvm import isa


def test_encode_binary_scalar_matches_vBinaryS_layout():
    # Upstream vBinaryS::Encode uses vCompactX(xd), and upstream isa.h defines
    # vCompactX(x) as x >> 5 for xbuf addresses aligned to 32 bytes.
    words = isa.encode_binary_scalar(opcode=isa.V_ADDS, xn=0x200, xd=0x400, count=32, scalar_bits=0x3F800000)
    assert len(words) == 2
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == 0x200
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_ADDS]
    assert words[1] == (0x3F800000 << 32) | ((0x400 >> 5) << 16) | 32
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_binary_scalar_matches_vBinaryS_layout -v
```

Expected: FAIL because `encode_binary_scalar` does not exist yet

- [ ] **Step 3: Implement the helper**

Add to `candle_dvm/isa.pyx`:

```python
def encode_binary_scalar(opcode: int, xn: int, xd: int, count: int, scalar_bits: int):
    return [
        make_simd_head(opcode, xn, 2),
        (scalar_bits << 32) | ((xd >> 5) << 16) | count,
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_binary_scalar_matches_vBinaryS_layout -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add scalar binary encode helper"
```

### Task 3: Implement `BinaryScalarOp` normalize and emit

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Modify: `candle_dvm/ops.pxd`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing `BinaryScalarOp` tests**

```python
from candle_dvm.ops import NDLoad, BinaryScalarOp, DTYPE_F32, DTYPE_FP16, DTYPE_INT32, BINS_ADD, BINS_DIV
from candle_dvm.code import Code
import pytest


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
    op.emit(code, [])
    assert code.data_size == 16


def test_binary_scalar_emit_raises_for_unsupported_dtype():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_INT32)
    x.normalize(); x.xbuf = 0x200
    op = BinaryScalarOp(op_type=BINS_DIV, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400
    code = Code(capacity=4096)
    with pytest.raises(NotImplementedError):
        op.emit(code, [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py::test_binary_scalar_normalize_preserves_shape_and_dtype tests/test_ops.py::test_binary_scalar_emit_appends_two_words tests/test_ops.py::test_binary_scalar_emit_raises_for_unsupported_dtype -v
```

Expected: FAIL because `BinaryScalarOp` does not exist yet

- [ ] **Step 3: Implement `BinaryScalarOp` minimally**

Requirements:
- `BinaryScalarOp(FlexOp)`
- constructor `(op_type, src, scalar)`
- `normalize()` preserves shape and dtype from `src`
- `emit()`:
  - compute `count` like `BinaryOp`
  - look up opcode from `BINARY_SCALAR_OPCODE_TABLE[(op_type, dtype)]`
  - pack scalar bits according to dtype
  - call `encode_binary_scalar()`
  - append 2 words
  - no relocs
- unsupported `(op, dtype)` -> `NotImplementedError`

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ops.py::test_binary_scalar_normalize_preserves_shape_and_dtype tests/test_ops.py::test_binary_scalar_emit_appends_two_words tests/test_ops.py::test_binary_scalar_emit_raises_for_unsupported_dtype -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx candle_dvm/ops.pxd tests/test_ops.py
git commit -m "feat: add binary scalar op support"
```

### Task 4: Route public `Kernel` methods to scalar path when one operand is scalar

**Files:**
- Modify: `candle_dvm/api.pyx`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing public API tests**

```python
from candle_dvm import Kernel, float32
import pytest


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_add_accepts_tensor_scalar tests/test_api.py::test_kernel_mul_accepts_tensor_scalar tests/test_api.py::test_kernel_div_accepts_tensor_scalar tests/test_api.py::test_kernel_maximum_accepts_tensor_scalar tests/test_api.py::test_kernel_add_accepts_scalar_left_commutative tests/test_api.py::test_kernel_div_scalar_left_raises tests/test_api.py::test_kernel_both_scalars_raises -v
```

Expected: FAIL because API only accepts tensor-tensor today

- [ ] **Step 3: Implement minimal scalar routing**

Requirements:
- for `add`, `mul`, `div`, `maximum`, `minimum`
- if one operand is tensor and the other scalar:
  - route to `BinaryScalarOp`
- if both tensors:
  - keep existing `BinaryOp`
- if both scalars:
  - raise `TypeError`
- scalar-left support only for commutative ops in this batch (`add`, `mul`, `maximum`, `minimum`); for `div(scalar, tensor)`, raise `NotImplementedError`

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_add_accepts_tensor_scalar tests/test_api.py::test_kernel_mul_accepts_tensor_scalar tests/test_api.py::test_kernel_div_accepts_tensor_scalar tests/test_api.py::test_kernel_maximum_accepts_tensor_scalar tests/test_api.py::test_kernel_add_accepts_scalar_left_commutative tests/test_api.py::test_kernel_div_scalar_left_raises tests/test_api.py::test_kernel_both_scalars_raises -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/api.pyx tests/test_api.py
git commit -m "feat: route scalar arguments through batch c"
```

### Task 5: Add hardware tests for Batch C on 910B

**Files:**
- Modify: `tests/test_add.py`

- [ ] **Step 1: Write the failing 910B tests**

```python
import numpy as np
import pytest
import candle_dvm as dvm


@pytest.mark.requires_910b
def test_add_scalar_fp32_end_to_end():
    @dvm.kernel()
    def my_adds(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.add(a, 1.0))
    x = np.full([32, 32], 2.0, np.float32)
    z = my_adds(x)
    np.testing.assert_allclose(z, x + 1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_mul_scalar_fp16_end_to_end():
    @dvm.kernel()
    def my_muls(k, x):
        a = k.load(x.shape, dvm.float16)
        return k.store(k.mul(a, np.float16(2.0)))
    x = np.full([32, 32], 3.0, np.float16)
    z = my_muls(x)
    np.testing.assert_allclose(z.astype(np.float32), (x * np.float16(2.0)).astype(np.float32), rtol=5e-3, atol=5e-3)


@pytest.mark.requires_910b
def test_div_scalar_fp32_end_to_end():
    @dvm.kernel()
    def my_divs(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.div(a, 2.0))
    x = np.full([32, 32], 6.0, np.float32)
    z = my_divs(x)
    np.testing.assert_allclose(z, x / 2.0, rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_maximum_scalar_fp32_end_to_end():
    @dvm.kernel()
    def my_maxs(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.maximum(a, 0.0))
    x = np.linspace(-5.0, 5.0, 1024).reshape(32, 32).astype(np.float32)
    z = my_maxs(x)
    np.testing.assert_allclose(z, np.maximum(x, 0.0), rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_minimum_scalar_fp16_end_to_end():
    @dvm.kernel()
    def my_mins(k, x):
        a = k.load(x.shape, dvm.float16)
        return k.store(k.minimum(a, np.float16(1.0)))
    x = np.linspace(-2.0, 4.0, 1024).reshape(32, 32).astype(np.float16)
    z = my_mins(x)
    np.testing.assert_allclose(z.astype(np.float32), np.minimum(x, np.float16(1.0)).astype(np.float32), rtol=5e-3, atol=5e-3)

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_add.py::test_add_scalar_fp32_end_to_end tests/test_add.py::test_mul_scalar_fp16_end_to_end tests/test_add.py::test_div_scalar_fp32_end_to_end tests/test_add.py::test_maximum_scalar_fp32_end_to_end tests/test_add.py::test_minimum_scalar_fp16_end_to_end -v
```

Expected: FAIL because scalar routing and/or encoding are incomplete

- [ ] **Step 3: Make the minimal fixes needed for end-to-end Batch C**

Only fix what the tests prove is required. Do not add compare scalar or select.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_add.py::test_add_scalar_fp32_end_to_end tests/test_add.py::test_mul_scalar_fp16_end_to_end tests/test_add.py::test_div_scalar_fp32_end_to_end tests/test_add.py::test_maximum_scalar_fp32_end_to_end tests/test_add.py::test_minimum_scalar_fp16_end_to_end -v
```

Expected: PASS on 910B

- [ ] **Step 5: Commit**

```bash
git add tests/test_add.py candle_dvm/api.pyx candle_dvm/ops.pyx candle_dvm/isa.pyx
git commit -m "test: validate batch c scalar ops end to end"
```

### Task 6: Run the full Batch C verification set

**Files:**
- Modify: `tests/test_isa.py` (only if coverage gaps are discovered)
- Modify: `tests/test_ops.py` (only if coverage gaps are discovered)
- Modify: `tests/test_api.py` (only if coverage gaps are discovered)
- Modify: `tests/test_add.py` (only if hardware coverage gaps are discovered)

- [ ] **Step 1: Write the verification checklist into the task log**

Use this exact checklist:

```text
- batch c fp32 opcode routing covered
- batch c fp16 opcode routing covered
- batch c encode helper covered
- binary scalar normalize/emit covered
- public scalar API routing covered
- batch c hardware tests pass on 910B
- previous add/unary/binary tests still pass
```

- [ ] **Step 2: Run the portable tests**

Run:

```bash
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_api.py -v
```

Expected: PASS

- [ ] **Step 3: Run the hardware tests**

Run:

```bash
python -m pytest tests/test_add.py -v
```

Expected: PASS on 910B

- [ ] **Step 4: Run the full suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_isa.py tests/test_ops.py tests/test_api.py tests/test_add.py
git commit -m "test: verify batch c baseline"
```

## Notes for the implementing agent

- Do not add compare scalar ops in this batch.
- Do not add scalar-left `div` support in this batch.
- Keep scalar routing inside existing method names (`add`, `mul`, `div`, `maximum`, `minimum`).
- Match upstream `vBinaryS` exactly.
- Reuse existing `BinaryOp` count logic instead of inventing a new count rule.
- `fp16` scalar packing must be explicit and tested.

## Final verification commands

When all tasks are done, these should all pass from repo root:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_api.py -v
python -m pytest tests/test_add.py -v
python -m pytest -v
```
