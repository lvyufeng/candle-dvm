# candle-dvm Batch A+B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 2 Batch A and Batch B support to `candle-dvm`: unary vector ops (`sqrt`, `abs`, `log`, `exp`, `round`, `floor`, `ceil`, `trunc`, `isfinite`) plus binary vector ops (`sub`, `mul`, `div`, `maximum`, `minimum`) with both `fp32` and `fp16` support, and expose only the ops that are fully tested end-to-end.

**Architecture:** Extend the existing phase-1 vector path without changing the runtime or launch stack. The work lands in three layers: ISA routing/encode helpers, NDObject op classes and emit logic, then public API exposure plus PyKernel end-to-end execution. All additions stay within the 2-word vector instruction families already proven in phase 1 (`vUnary` and `vBinary`).

**Tech Stack:** Python 3.12, Cython 3, setuptools, pytest, NumPy, Ascend ACL/RT runtime, prebuilt DVM C220 device binary, 910B hardware

---

## Scope and boundaries

This plan covers only:
- Batch A unary ops backed by upstream `vUnary` with direct opcodes
- Batch B binary ops backed by upstream `vBinary`
- `fp32` and `fp16` variants only
- public API exposure only for the ops that are fully implemented and passing

This plan explicitly does **not** include:
- `reciprocal` and `logical_not` as direct unary SIMD ops (upstream unary table marks them `V_NONE` for all dtypes)
- scalar variants (`vBinaryS`) — that is Batch C
- compare/select/cast/broadcast/reduce — later batches
- any runtime/system changes

## File map

### Files to modify

- Modify: `setup.py`
  - Add any new Cython modules if needed, but likely only recompilation of existing modules is required.
  - Keep extension list explicit.

- Modify: `candle_dvm/isa.pyx`
  - Add unary opcode routing tables for `fp32`/`fp16`.
  - Extend binary opcode tables for Batch B ops.
  - Add `encode_unary()` helper wrapping the `vUnary` 2-word layout.
  - Keep `make_simd_head()` unchanged.

- Modify: `candle_dvm/isa.pxd`
  - Export any new helper signatures needed by `ops.pyx`.

- Modify: `candle_dvm/ops.pyx`
  - Add unary op enum constants.
  - Extend binary op constants and routing tables.
  - Implement `UnaryOp(FlexOp)`.
  - Extend `BinaryOp.emit()` and validation to support Batch B ops and `fp16`.
  - Preserve existing `add` behavior.

- Modify: `candle_dvm/ops.pxd`
  - Export `UnaryOp` and any new public constants/fields required by `api.pyx`.

- Modify: `candle_dvm/api.pyx`
  - Add public `Kernel` methods only for completed ops:
    - unary: `sqrt`, `abs`, `log`, `exp`, `round`, `floor`, `ceil`, `trunc`, `isfinite`
    - binary: `sub`, `mul`, `div`, `maximum`, `minimum`

- Modify: `candle_dvm/pykernel.py`
  - No major architecture change expected.
  - Ensure decorator-built graphs can call the newly exposed API methods.

- Modify: `candle_dvm/__init__.py`
  - Export new public methods only through `Kernel`/`PyKernel`; no extra free functions required.
  - Add `float16` alias if not already exported and needed by tests.

- Modify: `tests/test_isa.py`
  - Add opcode routing tests for all new `(op, dtype)` pairs in Batch A+B.
  - Add direct `encode_unary()` tests.

- Modify: `tests/test_ops.py`
  - Add normalize and emit tests for `UnaryOp` and expanded `BinaryOp`.

- Modify: `tests/test_api.py`
  - Add graph-builder tests for the new public methods.

- Modify: `tests/test_add.py`
  - Keep existing add tests unchanged.
  - Add new end-to-end hardware tests for representative unary and binary ops.

### No new files expected

This batch should not require new source files. It should fit into the existing phase-1 file structure.

## Upstream references to read before coding

- `/home/dndx/lvyufeng/dvm/src/ops.cc:35-80`
  - `unary_id_list`, `binary_id_list`, `binarys_id_list`
- `/home/dndx/lvyufeng/dvm/src/isa.h:343-460`
  - `vUnary::Encode`, `vBinary::Encode`
- `/home/dndx/lvyufeng/candle-dvm/candle_dvm/isa.pyx`
- `/home/dndx/lvyufeng/candle-dvm/candle_dvm/ops.pyx`
- `/home/dndx/lvyufeng/candle-dvm/tests/test_ops.py`
- `/home/dndx/lvyufeng/candle-dvm/tests/test_add.py`

## Batch content

### Batch A unary ops in scope

Implement exactly these direct unary ops:
- `sqrt`
- `abs`
- `log`
- `exp`
- `round`
- `floor`
- `ceil`
- `trunc`
- `isfinite`

Do **not** expose yet:
- `reciprocal`
- `logical_not`

### Batch B binary ops in scope

Implement exactly these binary ops:
- `sub`
- `mul`
- `div`
- `maximum`
- `minimum`

Keep existing:
- `add`

## Task breakdown

### Task 1: Extend ISA routing tables for Batch A+B

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing ISA routing tests**

```python
from candle_dvm import isa


def test_unary_opcode_routing_all_batch_a():
    """Every (unary_op, dtype) pair must map to the expected opcode."""
    expected = {
        (isa.UNARY_SQRT, isa.DTYPE_F32): isa.V_SQRT,
        (isa.UNARY_SQRT, isa.DTYPE_FP16): isa.V_SQRT_FP16,
        (isa.UNARY_ABS, isa.DTYPE_F32): isa.V_ABS,
        (isa.UNARY_ABS, isa.DTYPE_FP16): isa.V_ABS_FP16,
        (isa.UNARY_LOG, isa.DTYPE_F32): isa.V_LOG,
        (isa.UNARY_LOG, isa.DTYPE_FP16): isa.V_LOG_FP16,
        (isa.UNARY_EXP, isa.DTYPE_F32): isa.V_EXP,
        (isa.UNARY_EXP, isa.DTYPE_FP16): isa.V_EXP_FP16,
        (isa.UNARY_ROUND, isa.DTYPE_F32): isa.V_ROUND,
        (isa.UNARY_FLOOR, isa.DTYPE_F32): isa.V_FLOOR,
        (isa.UNARY_CEIL, isa.DTYPE_F32): isa.V_CEIL,
        (isa.UNARY_TRUNC, isa.DTYPE_F32): isa.V_TRUNC,
        # Note: round, floor, ceil, trunc have NO fp16 variant in upstream
        # dvm/src/ops.cc:44-47 (all V_NONE for kFloat16). Only fp32 is supported.
        (isa.UNARY_ISFINITE, isa.DTYPE_F32): isa.V_ISFINITE,
        (isa.UNARY_ISFINITE, isa.DTYPE_FP16): isa.V_ISFINITE_FP16,
    }
    for key, val in expected.items():
        assert isa.UNARY_OPCODE_TABLE[key] == val, f"mismatch for {key}"


def test_binary_opcode_routing_all_batch_b():
    """Every (binary_op, dtype) pair must map to the expected opcode."""
    expected = {
        (isa.BIN_ADD, isa.DTYPE_F32): isa.V_ADD,
        (isa.BIN_ADD, isa.DTYPE_FP16): isa.V_ADD_FP16,
        (isa.BIN_SUB, isa.DTYPE_F32): isa.V_SUB,
        (isa.BIN_SUB, isa.DTYPE_FP16): isa.V_SUB_FP16,
        (isa.BIN_MUL, isa.DTYPE_F32): isa.V_MUL,
        (isa.BIN_MUL, isa.DTYPE_FP16): isa.V_MUL_FP16,
        (isa.BIN_DIV, isa.DTYPE_F32): isa.V_DIV,
        (isa.BIN_DIV, isa.DTYPE_FP16): isa.V_DIV_FP16,
        (isa.BIN_MAX, isa.DTYPE_F32): isa.V_MAX,
        (isa.BIN_MAX, isa.DTYPE_FP16): isa.V_MAX_FP16,
        (isa.BIN_MIN, isa.DTYPE_F32): isa.V_MIN,
        (isa.BIN_MIN, isa.DTYPE_FP16): isa.V_MIN_FP16,
    }
    for key, val in expected.items():
        assert isa.BINARY_OPCODE_TABLE[key] == val, f"mismatch for {key}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_isa.py::test_unary_opcode_routing_all_batch_a tests/test_isa.py::test_binary_opcode_routing_all_batch_b -v
```

Expected: FAIL because the routing tables and/or constants do not exist yet

- [ ] **Step 3: Add minimal routing tables and helper declarations**

Implement in `candle_dvm/isa.pyx`:
- unary op enum constants (e.g. `UNARY_SQRT`, `UNARY_ABS`, etc.)
- binary op constants for Batch B (`BIN_SUB`, `BIN_MUL`, `BIN_DIV`, `BIN_MAX`, `BIN_MIN`)
- `UNARY_OPCODE_TABLE`
- extend `BINARY_OPCODE_TABLE`
- export any needed names from `isa.pxd`

Use the exact upstream mappings from `dvm/src/ops.cc:35-64`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_isa.py::test_unary_opcode_routing_all_batch_a tests/test_isa.py::test_binary_opcode_routing_all_batch_b -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add unary and binary opcode routing tables"
```

### Task 2: Add `encode_unary()` to the ISA layer

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing unary encode test**

```python
from candle_dvm import isa


def test_encode_unary_fp32_matches_vUnary_layout():
    words = isa.encode_unary(opcode=isa.V_SQRT, xd=0x200, xn=0x400, count=32)
    assert len(words) == 2
    # head ext field should contain xd (not xn) for vUnary
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == 0x200
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_SQRT]
    assert words[1] == (0x400 << 32) | 32
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_unary_fp32_matches_vUnary_layout -v
```

Expected: FAIL because `encode_unary` does not exist yet

- [ ] **Step 3: Implement the minimal helper**

Add to `candle_dvm/isa.pyx`:

```python
def encode_unary(opcode: int, xd: int, xn: int, count: int):
    return [make_simd_head(opcode, xd, 2), (xn << 32) | count]
```

Use the function-offset-based head field already established in phase 1.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_unary_fp32_matches_vUnary_layout -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add unary encode helper"
```

### Task 3: Implement `UnaryOp` normalize and emit

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Modify: `candle_dvm/ops.pxd`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing UnaryOp tests**

```python
from candle_dvm.ops import NDLoad, UnaryOp, DTYPE_F32, DTYPE_FP16, DTYPE_BOOL, DTYPE_INT32, UNARY_SQRT, UNARY_ISFINITE
from candle_dvm.code import Code


def test_unary_normalize_preserves_shape_and_dtype_for_sqrt():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = UnaryOp(op_type=UNARY_SQRT, src=x)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_F32


def test_unary_normalize_sets_bool_dtype_for_isfinite():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_FP16)
    op = UnaryOp(op_type=UNARY_ISFINITE, src=x)
    op.normalize()
    assert op.type_id == DTYPE_BOOL


def test_unary_emit_appends_two_words():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    x.normalize()
    x.xbuf = 0x200
    op = UnaryOp(op_type=UNARY_SQRT, src=x)
    op.normalize()
    op.xbuf = 0x400
    code = Code(capacity=4096)
    relocs = []
    op.emit(code, relocs)
    assert code.data_size == 16
    assert relocs == []


def test_unary_emit_raises_for_unsupported_dtype():
    import pytest
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_INT32)
    x.normalize()
    x.xbuf = 0x200
    op = UnaryOp(op_type=UNARY_SQRT, src=x)
    op.normalize()
    op.xbuf = 0x400
    code = Code(capacity=4096)
    with pytest.raises(NotImplementedError):
        op.emit(code, [])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py::test_unary_normalize_preserves_shape_and_dtype_for_sqrt tests/test_ops.py::test_unary_normalize_sets_bool_dtype_for_isfinite tests/test_ops.py::test_unary_emit_appends_two_words -v
```

Expected: FAIL because `UnaryOp` does not exist yet

- [ ] **Step 3: Implement the minimal `UnaryOp`**

Add to `candle_dvm/ops.pyx`:
- `UnaryOp(FlexOp)`
- `normalize()` for shape/dtype propagation
- `emit()` using `isa.encode_unary()`
- route opcodes through `UNARY_OPCODE_TABLE`

Important:
- `isfinite` returns bool dtype
- `count` should match the same contiguous-row logic already used by `BinaryOp`
- no new relocations are needed

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ops.py::test_unary_normalize_preserves_shape_and_dtype_for_sqrt tests/test_ops.py::test_unary_normalize_sets_bool_dtype_for_isfinite tests/test_ops.py::test_unary_emit_appends_two_words -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx candle_dvm/ops.pxd tests/test_ops.py
git commit -m "feat: add unary op support"
```

### Task 4: Expand `BinaryOp` for Batch B

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing expanded BinaryOp tests**

```python
from candle_dvm.ops import NDLoad, BinaryOp, DTYPE_F32, DTYPE_FP16, BIN_SUB, BIN_MUL, BIN_DIV, BIN_MAX, BIN_MIN
from candle_dvm.code import Code


def test_binary_sub_emit_uses_v_sub_fp32():
    a = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(4, 8), dtype=DTYPE_F32)
    a.normalize(); b.normalize()
    a.xbuf = 0x200; b.xbuf = 0x400
    op = BinaryOp(op_type=BIN_SUB, lhs=a, rhs=b)
    op.normalize()
    op.xbuf = 0x600
    code = Code(capacity=4096)
    op.emit(code, [])
    assert code.data_size == 16


def test_binary_mul_emit_uses_v_mul_fp16():
    a = NDLoad(io_index=0, shape=(4, 16), dtype=DTYPE_FP16)
    b = NDLoad(io_index=1, shape=(4, 16), dtype=DTYPE_FP16)
    a.normalize(); b.normalize()
    a.xbuf = 0x200; b.xbuf = 0x400
    op = BinaryOp(op_type=BIN_MUL, lhs=a, rhs=b)
    op.normalize()
    op.xbuf = 0x600
    code = Code(capacity=4096)
    op.emit(code, [])
    assert code.data_size == 16
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py::test_binary_sub_emit_uses_v_sub_fp32 tests/test_ops.py::test_binary_mul_emit_uses_v_mul_fp16 -v
```

Expected: FAIL because the routing table does not contain the new ops yet

- [ ] **Step 3: Extend `BinaryOp` minimally**

- add Batch B op constants
- extend `BINARY_OPCODE_TABLE`
- keep `BinaryOp.emit()` unchanged structurally — only the opcode lookup widens

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ops.py::test_binary_sub_emit_uses_v_sub_fp32 tests/test_ops.py::test_binary_mul_emit_uses_v_mul_fp16 -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx tests/test_ops.py
git commit -m "feat: expand binary ops for batch b"
```

### Task 5: Expose Batch A unary methods on the public Kernel API

**Files:**
- Modify: `candle_dvm/api.pyx`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing API tests for unary methods**

```python
from candle_dvm import Kernel, float32


def test_kernel_exposes_sqrt_and_log_methods():
    k = Kernel()
    x = k.load((32, 32), float32)
    y = k.sqrt(x)
    z = k.log(x)
    assert y.shape_ref == (32, 32)
    assert z.shape_ref == (32, 32)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_exposes_sqrt_and_log_methods -v
```

Expected: FAIL because `Kernel.sqrt` / `Kernel.log` do not exist yet

- [ ] **Step 3: Add only the completed unary methods**

Expose exactly these methods on `Kernel`:
- `sqrt`, `abs`, `log`, `exp`, `round`, `floor`, `ceil`, `trunc`, `isfinite`

Do not expose `reciprocal` or `logical_not` yet.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_exposes_sqrt_and_log_methods -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/api.pyx tests/test_api.py
git commit -m "feat: expose unary kernel api"
```

### Task 6: Expose Batch B binary methods on the public Kernel API

**Files:**
- Modify: `candle_dvm/api.pyx`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing API tests for Batch B binary methods**

```python
from candle_dvm import Kernel, float32


def test_kernel_exposes_batch_b_binary_methods():
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    assert k.sub(a, b).shape_ref == (32, 32)
    assert k.mul(a, b).shape_ref == (32, 32)
    assert k.div(a, b).shape_ref == (32, 32)
    assert k.maximum(a, b).shape_ref == (32, 32)
    assert k.minimum(a, b).shape_ref == (32, 32)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_exposes_batch_b_binary_methods -v
```

Expected: FAIL because the methods do not exist yet

- [ ] **Step 3: Add only the completed binary methods**

Expose exactly:
- `sub`, `mul`, `div`, `maximum`, `minimum`

Keep `add` as-is.

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_exposes_batch_b_binary_methods -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/api.pyx tests/test_api.py
git commit -m "feat: expose batch b binary kernel api"
```

### Task 7: Add public exports for `float16` and keep unsupported ops hidden

**Files:**
- Modify: `candle_dvm/__init__.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing export tests**

```python
import candle_dvm as dvm


def test_float16_is_exported():
    assert hasattr(dvm, "float16")


def test_reciprocal_is_not_exported_yet():
    assert not hasattr(dvm.Kernel, "reciprocal")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_api.py::test_float16_is_exported tests/test_api.py::test_reciprocal_is_not_exported_yet -v
```

Expected: FAIL because `float16` is not exported yet

- [ ] **Step 3: Update exports minimally**

In `__init__.py`:
- add `float16 = DTYPE_FP16`
- keep only finished ops reachable via `Kernel`
- do not export placeholders

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_api.py::test_float16_is_exported tests/test_api.py::test_reciprocal_is_not_exported_yet -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/__init__.py tests/test_api.py
git commit -m "feat: export float16 without exposing unfinished ops"
```

### Task 8: Add Batch A end-to-end hardware tests

**Files:**
- Modify: `tests/test_add.py`

- [ ] **Step 1: Write the failing hardware tests for representative unary ops**

```python
import numpy as np
import pytest
import candle_dvm as dvm


@pytest.mark.requires_910b
def test_decorated_sqrt_fp32_end_to_end():
    @dvm.kernel()
    def my_sqrt(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.sqrt(a))

    x = np.full([32, 32], 4.0, np.float32)
    z = my_sqrt(x)
    np.testing.assert_allclose(z, np.sqrt(x), rtol=1e-5, atol=1e-5)


@pytest.mark.requires_910b
def test_decorated_exp_fp16_end_to_end():
    @dvm.kernel()
    def my_exp(k, x):
        a = k.load(x.shape, dvm.float16)
        return k.store(k.exp(a))

    x = np.full([32, 32], 1.0, np.float16)
    z = my_exp(x)
    np.testing.assert_allclose(z.astype(np.float32), np.exp(x.astype(np.float32)), rtol=5e-3, atol=5e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_add.py::test_decorated_sqrt_fp32_end_to_end tests/test_add.py::test_decorated_exp_fp16_end_to_end -v
```

Expected: FAIL because the public unary API and/or emit path are incomplete

- [ ] **Step 3: Make the minimal fixes needed for unary end-to-end execution**

This step should only include the wiring already required by Tasks 1–7. Do not add extra ops.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_add.py::test_decorated_sqrt_fp32_end_to_end tests/test_add.py::test_decorated_exp_fp16_end_to_end -v
```

Expected: PASS on 910B

- [ ] **Step 5: Commit**

```bash
git add tests/test_add.py candle_dvm/api.pyx candle_dvm/ops.pyx candle_dvm/isa.pyx
git commit -m "test: validate unary ops end to end"
```

### Task 9: Add Batch B end-to-end hardware tests

**Files:**
- Modify: `tests/test_add.py`

- [ ] **Step 1: Write the failing hardware tests for representative binary ops**

```python
import numpy as np
import pytest
import candle_dvm as dvm


@pytest.mark.requires_910b
def test_decorated_mul_fp32_end_to_end():
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
    @dvm.kernel()
    def my_div(k, x, y):
        a = k.load(x.shape, dvm.float16)
        b = k.load(y.shape, dvm.float16)
        return k.store(k.div(a, b))

    x = np.full([32, 32], 6.0, np.float16)
    y = np.full([32, 32], 2.0, np.float16)
    z = my_div(x, y)
    np.testing.assert_allclose(z.astype(np.float32), (x / y).astype(np.float32), rtol=5e-3, atol=5e-3)


@pytest.mark.requires_910b
def test_decorated_maximum_fp32_end_to_end():
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
    @dvm.kernel()
    def my_min(k, x, y):
        a = k.load(x.shape, dvm.float16)
        b = k.load(y.shape, dvm.float16)
        return k.store(k.minimum(a, b))

    x = np.array([1.0, 5.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0], np.float16)
    y = np.array([8.0, 2.0, 6.0, 4.0, 7.0, 1.0, 5.0, 3.0], np.float16)
    z = my_min(x, y)
    np.testing.assert_allclose(z.astype(np.float32), np.minimum(x, y).astype(np.float32), rtol=5e-3, atol=5e-3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_add.py::test_decorated_mul_fp32_end_to_end tests/test_add.py::test_decorated_div_fp16_end_to_end tests/test_add.py::test_decorated_maximum_fp32_end_to_end tests/test_add.py::test_decorated_minimum_fp16_end_to_end -v
```

Expected: FAIL because the public binary API and/or dtype routing are incomplete

- [ ] **Step 3: Make the minimal fixes needed for binary end-to-end execution**

This step should only include the wiring already required by Tasks 1–9. Do not add scalar paths.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_add.py::test_decorated_mul_fp32_end_to_end tests/test_add.py::test_decorated_div_fp16_end_to_end tests/test_add.py::test_decorated_maximum_fp32_end_to_end tests/test_add.py::test_decorated_minimum_fp16_end_to_end -v
```

Expected: PASS on 910B

- [ ] **Step 5: Commit**

```bash
git add tests/test_add.py candle_dvm/api.pyx candle_dvm/ops.pyx candle_dvm/isa.pyx
git commit -m "test: validate batch b ops end to end"
```

### Task 10: Run the full Batch A+B verification set

**Files:**
- Modify: `tests/test_isa.py` (only if coverage gaps are discovered)
- Modify: `tests/test_ops.py` (only if coverage gaps are discovered)
- Modify: `tests/test_api.py` (only if coverage gaps are discovered)
- Modify: `tests/test_add.py` (only if hardware coverage gaps are discovered)

- [ ] **Step 1: Write the verification checklist into the task log**

Use this exact checklist:

```text
- unary fp32 opcode routing covered
- unary fp16 opcode routing covered
- batch b fp32 opcode routing covered
- batch b fp16 opcode routing covered
- public API exposes only completed ops
- unary end-to-end tests pass on 910B
- batch b arithmetic tests pass on 910B
- batch b extremum tests pass on 910B
- existing add path still passes
```

- [ ] **Step 2: Run portable tests**

Run:

```bash
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_api.py -v
```

Expected: PASS

- [ ] **Step 3: Run hardware tests for Batch A+B plus add regression**

Run:

```bash
python -m pytest tests/test_add.py -v
```

Expected: PASS on 910B, including old add tests and new unary/binary tests

- [ ] **Step 4: Run the full suite**

Run:

```bash
python -m pytest -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_isa.py tests/test_ops.py tests/test_api.py tests/test_add.py
git commit -m "test: verify batch a and b baseline"
```

## Notes for the implementing agent

- Do not expose `reciprocal` or `logical_not` in this plan.
- Do not add scalar variants in this plan.
- Do not add compare/select/cast/broadcast/reduce in this plan.
- Match upstream opcode routing tables exactly from `dvm/src/ops.cc`.
- Keep `vUnary` and `vBinary` as separate encode helpers; do not merge them.
- Use relaxed tolerance only for `fp16` hardware tests.
- Keep the public API strict: only expose methods that pass the full two-layer verification.

## Final verification commands

When all tasks are done, these should all pass from repo root:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_api.py -v
python -m pytest tests/test_add.py -v
python -m pytest -v
```
