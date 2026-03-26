# candle-dvm Batch D Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Batch D compare support to `candle-dvm`: tensor-tensor and tensor-scalar compare ops (`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`) with scalar-left rewrite, workspace-aware kernel allocation, bool output propagation, and 910B end-to-end validation.

**Architecture:** Batch D extends the existing Phase 2 vector path by introducing compare-specific ISA helpers (`vCompare`, `vCompareS`), new op classes (`CompareOp`, `CompareScalarOp`), and the first workspace-aware xbuf allocation protocol in `kernel.pyx`. Public compare APIs land only after portable encode/emit tests and 910B end-to-end bool-output tests are both passing.

**Tech Stack:** Python 3.12, Cython 3, setuptools, pytest, NumPy, Ascend 910B runtime, DVM-compatible C220 device binary

---

## Scope and boundaries

This plan covers **Batch D only**:

- tensor-tensor compare ops:
  - `equal`
  - `not_equal`
  - `greater`
  - `greater_equal`
  - `less`
  - `less_equal`
- tensor-scalar compare ops for the same six semantics
- scalar-left compare via API rewrite
- bool output propagation and end-to-end verification
- minimal workspace-aware kernel xbuf allocation

This plan explicitly does **not** cover:

- select (`vSelect`) — Batch E
- cast / copy — Batch F
- broadcast / reduce
- shape broadcasting for compare operands
- scalar-scalar compare execution (must raise `TypeError`)
- workspace reuse / liveness analysis / allocator redesign
- bool storage redesign beyond what is needed to validate compare output end-to-end on current hardware path

## File map

### Files to modify

- Modify: `candle_dvm/isa.pyx`
  - add compare semantic type constants (`CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, `CMP_LE`)
  - add dtype routing tables for tensor compare and scalar compare
  - add `encode_compare()` for `vCompare`
  - add `encode_compare_scalar()` for `vCompareS`

- Modify: `candle_dvm/isa.pxd`
  - export `encode_compare()` and `encode_compare_scalar()`
  - document any new compare-related symbols needed by Cython callers

- Modify: `candle_dvm/ops.pyx`
  - add compare type constants and re-exports
  - add default `workspace_slots()` protocol on `FlexOp`
  - add `CompareOp(FlexOp)`
  - add `CompareScalarOp(FlexOp)`
  - add scalar bit packing reuse for compare-scalar

- Modify: `candle_dvm/ops.pxd`
  - export `workspace_slots()` on `FlexOp`
  - export `CompareOp` and `CompareScalarOp`

- Modify: `candle_dvm/kernel.pyx`
  - extend xbuf assignment to allocate workspace slots monotonically after result slots
  - keep the current codegen pipeline otherwise unchanged

- Modify: `candle_dvm/api.pyx`
  - add compare API methods (`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`)
  - add `_compare_dispatch` helper for tensor/tensor, tensor/scalar, scalar/tensor normalization
  - implement scalar-left rewrite rules for ordered compares

- Modify: `tests/test_isa.py`
  - add compare constant tests, routing-table tests, and exact encode-layout tests

- Modify: `tests/test_ops.py`
  - add normalize / emit / workspace / error-path tests for `CompareOp` and `CompareScalarOp`

- Modify: `tests/test_kernel.py`
  - add workspace-aware xbuf allocation tests at kernel level

- Modify: `tests/test_api.py`
  - add public compare API tests, scalar-left rewrite tests, and error tests

- Modify: `tests/test_add.py`
  - add 910B end-to-end compare tests for tensor-tensor, tensor-scalar, and scalar-left rewritten paths

### Files not expected to change

- `candle_dvm/system.pyx`
- `candle_dvm/code.pyx`
- `candle_dvm/pykernel.py`
- `candle_dvm/__init__.py`

If bool-output hardware validation exposes a root-cause bug in one of these files, stop and debug before broadening scope.

## Upstream references to read before coding

- `/home/dndx/lvyufeng/dvm/src/ops.cc`
  - compare opcode tables (`binary_id_list`, `binarys_id_list`, compare-related op emission)
- `/home/dndx/lvyufeng/dvm/src/isa.h`
  - `vCompare::Encode`
  - `vCompareS::Encode`
- `/home/dndx/lvyufeng/dvm/include/dvm.h`
  - compare semantic enum / dtype constants
- current `candle_dvm/isa.pyx`
- current `candle_dvm/ops.pyx`
- current `candle_dvm/kernel.pyx`
- current `candle_dvm/api.pyx`
- current `tests/test_kernel.py`
- current `tests/test_add.py`

## Batch D design rules

### Compare semantic type separation

Keep compare **semantic type** separate from compare **dtype opcode routing**:

- semantic type encodes EQ / NE / GT / GE / LT / LE
- dtype routing chooses `V_CMP` vs `V_CMP_FP16` and `V_CMPS` vs `V_CMPS_FP16`

Do not collapse both concerns into one generic table.

### `vCompare` encoding

Use `vCompare` exactly:

- `pc[0] = make_simd_head(opcode, (cmp_type << 18) | xn, 2)`
- `pc[1] = count << 49 | compact_x(ws) << 36 | xd << 18 | xm`

Important:
- `count` is 15 bits
- `ws` uses `compact_x(ws)`

### `vCompareS` encoding

Use `vCompareS` exactly:

- `pc[0] = make_simd_head(opcode, (cmp_type << 18) | xn, 3)`
- `pc[1] = count << 48 | ws << 18 | xd`
- `pc[2] = scalar_bits`

Important:
- instruction size is 3 words
- `ws` is a full 18-bit xbuf value here, not compacted
- scalar packing matches Batch C

### Output dtype policy

Compare outputs are always `DTYPE_BOOL`.

This must be true in:
- `normalize()`
- public API graph-building behavior
- portable tests
- hardware end-to-end validation

### Operand policy

For this batch:
- tensor-tensor compare → `CompareOp`
- tensor-scalar compare → `CompareScalarOp`
- scalar-left compare → rewrite in API layer to tensor-scalar form
- scalar-scalar compare → `TypeError`

Scalar-left rewrite rules:
- `equal(s, x)` → `equal(x, s)`
- `not_equal(s, x)` → `not_equal(x, s)`
- `greater(s, x)` → `less(x, s)`
- `greater_equal(s, x)` → `less_equal(x, s)`
- `less(s, x)` → `greater(x, s)`
- `less_equal(s, x)` → `greater_equal(x, s)`

### Workspace policy

This is the first workspace-using batch.

Protocol:
- `FlexOp.workspace_slots()` default returns `0`
- `CompareOp.workspace_slots()` returns `1`
- `CompareScalarOp.workspace_slots()` returns `1`
- kernel xbuf assignment allocates those extra slots after the result slot

No reuse. No liveness tracking.

## Task breakdown

### Task 1: Add compare semantic constants and routing tables

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing compare constant and routing tests**

```python
from candle_dvm import isa


def test_compare_type_constants_match_upstream():
    assert isa.CMP_EQ == 0
    assert isa.CMP_NE == 1
    assert isa.CMP_GT == 2
    assert isa.CMP_GE == 3
    assert isa.CMP_LT == 4
    assert isa.CMP_LE == 5


def test_compare_opcode_routing_fp32_and_fp16():
    expected = {
        isa.DTYPE_F32: isa.V_CMP,
        isa.DTYPE_FP16: isa.V_CMP_FP16,
    }
    for dtype, opcode in expected.items():
        assert isa.COMPARE_OPCODE_TABLE[dtype] == opcode


def test_compare_scalar_opcode_routing_fp32_and_fp16():
    expected = {
        isa.DTYPE_F32: isa.V_CMPS,
        isa.DTYPE_FP16: isa.V_CMPS_FP16,
    }
    for dtype, opcode in expected.items():
        assert isa.COMPARE_SCALAR_OPCODE_TABLE[dtype] == opcode
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_isa.py::test_compare_type_constants_match_upstream tests/test_isa.py::test_compare_opcode_routing_fp32_and_fp16 tests/test_isa.py::test_compare_scalar_opcode_routing_fp32_and_fp16 -v
```

Expected: FAIL because compare constants and routing tables do not exist yet

- [ ] **Step 3: Add minimal compare constants and routing tables**

Implement in `candle_dvm/isa.pyx`:
- `CMP_EQ`, `CMP_NE`, `CMP_GT`, `CMP_GE`, `CMP_LT`, `CMP_LE`
- `COMPARE_OPCODE_TABLE`
- `COMPARE_SCALAR_OPCODE_TABLE`
- export any needed names from `isa.pxd`

Use exact upstream mappings from `dvm/src/ops.cc` and `dvm/include/dvm.h`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_isa.py::test_compare_type_constants_match_upstream tests/test_isa.py::test_compare_opcode_routing_fp32_and_fp16 tests/test_isa.py::test_compare_scalar_opcode_routing_fp32_and_fp16 -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add compare constants and opcode routing"
```

### Task 2: Add `encode_compare()` for `vCompare`

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing `vCompare` encode test**

```python
from candle_dvm import isa


def test_encode_compare_matches_vCompare_layout():
    words = isa.encode_compare(
        opcode=isa.V_CMP,
        cmp_type=isa.CMP_GT,
        xn=0x200,
        xm=0x400,
        xd=0x600,
        ws=0x800,
        count=32,
    )
    assert len(words) == 2
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == ((isa.CMP_GT << 18) | 0x200)
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_CMP]
    assert words[1] == (32 << 49) | ((0x800 >> 5) << 36) | (0x600 << 18) | 0x400
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_compare_matches_vCompare_layout -v
```

Expected: FAIL because `encode_compare` does not exist yet

- [ ] **Step 3: Implement `encode_compare()` minimally**

Add to `candle_dvm/isa.pyx`:

```python
def encode_compare(opcode: int, cmp_type: int, xn: int, xm: int, xd: int, ws: int, count: int):
    return [
        make_simd_head(opcode, (cmp_type << 18) | xn, 2),
        (count << 49) | ((ws >> 5) << 36) | (xd << 18) | xm,
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_compare_matches_vCompare_layout -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add vCompare encode helper"
```

### Task 3: Add `encode_compare_scalar()` for `vCompareS`

**Files:**
- Modify: `candle_dvm/isa.pyx`
- Modify: `candle_dvm/isa.pxd`
- Test: `tests/test_isa.py`

- [ ] **Step 1: Write the failing `vCompareS` encode test**

```python
from candle_dvm import isa


def test_encode_compare_scalar_matches_vCompareS_layout():
    words = isa.encode_compare_scalar(
        opcode=isa.V_CMPS,
        cmp_type=isa.CMP_LE,
        xn=0x200,
        xd=0x400,
        ws=0x600,
        count=32,
        scalar_bits=0x3F800000,
    )
    assert len(words) == 3
    ext_field = (words[0] >> isa.V_HEAD_EXT_OFFSET) & 0x3FFFFFF
    assert ext_field == ((isa.CMP_LE << 18) | 0x200)
    assert ((words[0] >> isa.V_HEAD_ID_OFFSET) & 0xFFFF) == isa.SIMD_FUNC_OFFSET[isa.V_CMPS]
    assert words[1] == (32 << 48) | (0x600 << 18) | 0x400
    assert words[2] == 0x3F800000
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_compare_scalar_matches_vCompareS_layout -v
```

Expected: FAIL because `encode_compare_scalar` does not exist yet

- [ ] **Step 3: Implement `encode_compare_scalar()` minimally**

Add to `candle_dvm/isa.pyx`:

```python
def encode_compare_scalar(opcode: int, cmp_type: int, xn: int, xd: int, ws: int, count: int, scalar_bits: int):
    return [
        make_simd_head(opcode, (cmp_type << 18) | xn, 3),
        (count << 48) | (ws << 18) | xd,
        scalar_bits,
    ]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python -m pytest tests/test_isa.py::test_encode_compare_scalar_matches_vCompareS_layout -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/isa.pyx candle_dvm/isa.pxd tests/test_isa.py
git commit -m "feat: add vCompareS encode helper"
```

### Task 4: Add workspace protocol to `FlexOp` and kernel allocation

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Modify: `candle_dvm/ops.pxd`
- Modify: `candle_dvm/kernel.pyx`
- Test: `tests/test_kernel.py`

- [ ] **Step 1: Write the failing kernel workspace allocation tests**

```python
from candle_dvm.kernel import VKernelS
from candle_dvm.ops import NDLoad, NDStore, DTYPE_F32, DTYPE_BOOL, FlexOp


class DummyWorkspaceOp(FlexOp):
    def __init__(self, src):
        super().__init__(99, DTYPE_BOOL, src.shape_ref, src, None)
        self._workspace_xbuf = 0

    def normalize(self):
        self.shape_ref = self.lhs.shape_ref
        self.type_id = DTYPE_BOOL
        self.normalized = True

    def workspace_slots(self):
        return 1

    def emit(self, code, relocs):
        pass


def test_flexop_workspace_slots_default_zero():
    op = FlexOp(1, DTYPE_F32, (4, 8))
    assert op.workspace_slots() == 0


def test_kernel_allocates_workspace_slot_after_result_slot():
    k = VKernelS()
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    c = DummyWorkspaceOp(a)
    d = NDStore(io_index=0, src=c)
    for obj in [a, c, d]:
        k.append(obj)
    k.normalize()
    k.codegen()
    assert c.xbuf > a.xbuf
    assert c._workspace_xbuf > c.xbuf
    assert c._workspace_xbuf % 32 == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_kernel.py::test_flexop_workspace_slots_default_zero tests/test_kernel.py::test_kernel_allocates_workspace_slot_after_result_slot -v
```

Expected: FAIL because `workspace_slots()` and workspace allocation do not exist yet

- [ ] **Step 3: Implement minimal workspace protocol and allocation**

Requirements:
- `FlexOp.workspace_slots()` default returns `0`
- kernel xbuf assignment allocates extra slots after result slot when `workspace_slots() > 0`
- compare ops will later store workspace xbuf explicitly; for this task only the allocation protocol is needed

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_kernel.py::test_flexop_workspace_slots_default_zero tests/test_kernel.py::test_kernel_allocates_workspace_slot_after_result_slot -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx candle_dvm/ops.pxd candle_dvm/kernel.pyx tests/test_kernel.py
git commit -m "feat: add workspace slot protocol and kernel allocation"
```

### Task 5: Implement `CompareOp` for tensor-tensor compare

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Modify: `candle_dvm/ops.pxd`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing `CompareOp` tests**

```python
import pytest
from candle_dvm.code import Code
from candle_dvm.ops import NDLoad, CompareOp, DTYPE_F32, DTYPE_BOOL, CMP_GT


def test_compare_op_normalize_sets_bool_dtype():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_BOOL


def test_compare_op_workspace_slots_is_one():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    assert op.workspace_slots() == 1


def test_compare_op_emit_appends_two_words():
    a = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(4, 8), dtype=DTYPE_F32)
    a.normalize(); b.normalize()
    a.xbuf = 0x200; b.xbuf = 0x400
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    op.normalize(); op.xbuf = 0x600; op.workspace_xbuf = 0x800
    code = Code(capacity=4096)
    try:
        op.emit(code, [])
        assert code.size == 16
    finally:
        code.free()


def test_compare_op_shape_mismatch_raises():
    a = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    b = NDLoad(io_index=1, shape=(16, 16), dtype=DTYPE_F32)
    op = CompareOp(cmp_type=CMP_GT, lhs=a, rhs=b)
    with pytest.raises(ValueError):
        op.normalize()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py::test_compare_op_normalize_sets_bool_dtype tests/test_ops.py::test_compare_op_workspace_slots_is_one tests/test_ops.py::test_compare_op_emit_appends_two_words tests/test_ops.py::test_compare_op_shape_mismatch_raises -v
```

Expected: FAIL because `CompareOp` does not exist yet

- [ ] **Step 3: Implement `CompareOp` minimally**

Requirements:
- constructor `(cmp_type, lhs, rhs)`
- bool output dtype
- `workspace_slots() == 1`
- `emit()` uses `COMPARE_OPCODE_TABLE` + `encode_compare()`
- store workspace xbuf on the op

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ops.py::test_compare_op_normalize_sets_bool_dtype tests/test_ops.py::test_compare_op_workspace_slots_is_one tests/test_ops.py::test_compare_op_emit_appends_two_words tests/test_ops.py::test_compare_op_shape_mismatch_raises -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx candle_dvm/ops.pxd tests/test_ops.py
git commit -m "feat: add tensor compare op"
```

### Task 6: Implement `CompareScalarOp` for tensor-scalar compare

**Files:**
- Modify: `candle_dvm/ops.pyx`
- Modify: `candle_dvm/ops.pxd`
- Test: `tests/test_ops.py`

- [ ] **Step 1: Write the failing `CompareScalarOp` tests**

```python
import pytest
from candle_dvm.code import Code
from candle_dvm.ops import NDLoad, CompareScalarOp, DTYPE_F32, DTYPE_BOOL, DTYPE_INT32, CMP_LE


def test_compare_scalar_op_normalize_sets_bool_dtype():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize()
    assert op.shape_ref == (32, 32)
    assert op.type_id == DTYPE_BOOL


def test_compare_scalar_op_workspace_slots_is_one():
    x = NDLoad(io_index=0, shape=(32, 32), dtype=DTYPE_F32)
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    assert op.workspace_slots() == 1


def test_compare_scalar_op_emit_appends_three_words():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_F32)
    x.normalize(); x.xbuf = 0x200
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400; op.workspace_xbuf = 0x600
    code = Code(capacity=4096)
    try:
        op.emit(code, [])
        assert code.size == 24
    finally:
        code.free()


def test_compare_scalar_op_unsupported_dtype_raises():
    x = NDLoad(io_index=0, shape=(4, 8), dtype=DTYPE_INT32)
    x.normalize(); x.xbuf = 0x200
    op = CompareScalarOp(cmp_type=CMP_LE, src=x, scalar=1.0)
    op.normalize(); op.xbuf = 0x400; op.workspace_xbuf = 0x600
    code = Code(capacity=4096)
    try:
        with pytest.raises(NotImplementedError):
            op.emit(code, [])
    finally:
        code.free()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py::test_compare_scalar_op_normalize_sets_bool_dtype tests/test_ops.py::test_compare_scalar_op_workspace_slots_is_one tests/test_ops.py::test_compare_scalar_op_emit_appends_three_words tests/test_ops.py::test_compare_scalar_op_unsupported_dtype_raises -v
```

Expected: FAIL because `CompareScalarOp` does not exist yet

- [ ] **Step 3: Implement `CompareScalarOp` minimally**

Requirements:
- constructor `(cmp_type, src, scalar)`
- bool output dtype
- `workspace_slots() == 1`
- scalar packing matches Batch C
- `emit()` uses `COMPARE_SCALAR_OPCODE_TABLE` + `encode_compare_scalar()`

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_ops.py::test_compare_scalar_op_normalize_sets_bool_dtype tests/test_ops.py::test_compare_scalar_op_workspace_slots_is_one tests/test_ops.py::test_compare_scalar_op_emit_appends_three_words tests/test_ops.py::test_compare_scalar_op_unsupported_dtype_raises -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/ops.pyx candle_dvm/ops.pxd tests/test_ops.py
git commit -m "feat: add tensor-scalar compare op"
```

### Task 7: Expose compare methods and scalar-left rewrite in public API

**Files:**
- Modify: `candle_dvm/api.pyx`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing public API compare tests**

```python
import pytest
from candle_dvm import Kernel, float32
from candle_dvm.ops import DTYPE_BOOL


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_equal_tensor_tensor_returns_bool_node tests/test_api.py::test_kernel_greater_tensor_scalar_returns_bool_node tests/test_api.py::test_kernel_less_scalar_left_rewrites tests/test_api.py::test_kernel_compare_both_scalars_raises_type_error -v
```

Expected: FAIL because compare methods do not exist yet

- [ ] **Step 3: Implement minimal compare API dispatch**

Requirements:
- add `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`
- add internal `_compare_dispatch` helper
- scalar-left rewrite rules exactly as in the spec
- both scalars raise `TypeError`
- tensor-tensor uses `CompareOp`
- tensor-scalar uses `CompareScalarOp`

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_api.py::test_kernel_equal_tensor_tensor_returns_bool_node tests/test_api.py::test_kernel_greater_tensor_scalar_returns_bool_node tests/test_api.py::test_kernel_less_scalar_left_rewrites tests/test_api.py::test_kernel_compare_both_scalars_raises_type_error -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add candle_dvm/api.pyx tests/test_api.py
git commit -m "feat: expose compare kernel api"
```

### Task 8: Add 910B end-to-end compare tests

**Files:**
- Modify: `tests/test_add.py`

- [ ] **Step 1: Write the failing hardware compare tests**

```python
import numpy as np
import pytest
import candle_dvm as dvm


@pytest.mark.requires_910b
def test_equal_tensor_tensor_fp32_end_to_end():
    @dvm.kernel()
    def my_equal(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.equal(a, b))
    x = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    y = np.array([[1.0, 0.0], [3.0, 5.0]], np.float32)
    z = my_equal(x, y)
    np.testing.assert_array_equal(z, x == y)


@pytest.mark.requires_910b
def test_greater_tensor_scalar_fp16_end_to_end():
    @dvm.kernel()
    def my_greater(k, x):
        a = k.load(x.shape, dvm.float16)
        return k.store(k.greater(a, np.float16(1.0)))
    x = np.array([[0.5, 1.0], [1.5, 2.0]], np.float16)
    z = my_greater(x)
    np.testing.assert_array_equal(z, x > np.float16(1.0))


@pytest.mark.requires_910b
def test_less_scalar_left_fp32_end_to_end():
    @dvm.kernel()
    def my_less(k, x):
        a = k.load(x.shape, dvm.float32)
        return k.store(k.less(1.0, a))
    x = np.array([[0.5, 1.0], [1.5, 2.0]], np.float32)
    z = my_less(x)
    np.testing.assert_array_equal(z, 1.0 < x)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_add.py::test_equal_tensor_tensor_fp32_end_to_end tests/test_add.py::test_greater_tensor_scalar_fp16_end_to_end tests/test_add.py::test_less_scalar_left_fp32_end_to_end -v
```

Expected: FAIL because compare op path and/or bool-output E2E are incomplete

- [ ] **Step 3: Make the minimal fixes required for end-to-end compare execution**

Only fix what the failing tests prove is required. Do not add select or other workspace ops.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python -m pytest tests/test_add.py::test_equal_tensor_tensor_fp32_end_to_end tests/test_add.py::test_greater_tensor_scalar_fp16_end_to_end tests/test_add.py::test_less_scalar_left_fp32_end_to_end -v
```

Expected: PASS on 910B

- [ ] **Step 5: Commit**

```bash
git add tests/test_add.py candle_dvm/api.pyx candle_dvm/ops.pyx candle_dvm/kernel.pyx candle_dvm/isa.pyx
git commit -m "test: validate compare ops end to end"
```

### Task 9: Run the full Batch D verification set

**Files:**
- Modify: `tests/test_isa.py` (only if coverage gaps are discovered)
- Modify: `tests/test_ops.py` (only if coverage gaps are discovered)
- Modify: `tests/test_kernel.py` (only if coverage gaps are discovered)
- Modify: `tests/test_api.py` (only if coverage gaps are discovered)
- Modify: `tests/test_add.py` (only if hardware coverage gaps are discovered)

- [ ] **Step 1: Write the verification checklist into the task log**

Use this exact checklist:

```text
- compare type constants covered
- compare fp32 opcode routing covered
- compare fp16 opcode routing covered
- compare-scalar fp32 opcode routing covered
- compare-scalar fp16 opcode routing covered
- vCompare encode helper covered
- vCompareS encode helper covered
- workspace slot protocol covered
- kernel workspace allocation covered
- compare bool dtype propagation covered
- public compare API routing covered
- scalar-left rewrite covered
- compare hardware tests pass on 910B
- previous add/unary/binary/scalar tests still pass
```

- [ ] **Step 2: Run portable tests**

Run:

```bash
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_kernel.py tests/test_api.py -v
```

Expected: PASS

- [ ] **Step 3: Run hardware tests**

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
git add tests/test_isa.py tests/test_ops.py tests/test_kernel.py tests/test_api.py tests/test_add.py
git commit -m "test: verify batch d compare baseline"
```

## Notes for the implementing agent

- Keep compare semantic type and dtype opcode routing separate.
- Match upstream `vCompare` and `vCompareS` exactly.
- `vCompare` uses 15-bit count; `vCompareS` uses 16-bit count.
- `CompareOp` and `CompareScalarOp` both require exactly one workspace slot.
- Do not add select or any other workspace-using op in this batch.
- Do not add broadcasting semantics.
- Bool output must be fully validated before public compare API is considered complete.
- Reuse the existing scalar packing strategy from Batch C; do not invent a new one.

## Final verification commands

When all tasks are done, these should all pass from repo root:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_isa.py tests/test_ops.py tests/test_kernel.py tests/test_api.py -v
python -m pytest tests/test_add.py -v
python -m pytest -v
```
