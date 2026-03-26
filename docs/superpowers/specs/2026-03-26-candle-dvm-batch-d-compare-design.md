# candle-dvm Batch D: Compare Ops Design

Date: 2026-03-26
Status: Draft

## Summary

Batch D extends `candle-dvm` with element-wise compare ops (`equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`) for both tensor-tensor and tensor-scalar operands. It also introduces the first workspace-using op classes and establishes the workspace allocation protocol that Batch E (select) will reuse.

All six compare methods are exposed in the public API only after portable bytecode tests and 910B end-to-end tests both pass.

## Goals

### Primary goals

- Add tensor-tensor compare (`CompareOp`) backed by `vCompare`.
- Add tensor-scalar compare (`CompareScalarOp`) backed by `vCompareS`.
- Add scalar-left rewrite for all six compare ops.
- Bool output (`DTYPE_BOOL`) fully validated end-to-end on 910B.
- Establish `workspace_slots()` protocol on `FlexOp` for future ops.

### Non-goals for this batch

- Select / workspace-combined select ops (Batch E).
- Reduce, broadcast, cast, copy.
- Bool storage format redesign.
- Liveness-based workspace reuse.
- Workspace pool or allocator abstraction.
- Shape broadcasting for compare operands.

## Scope and boundaries

Batch D includes:

- Six tensor-tensor compare ops: `equal`, `not_equal`, `greater`, `greater_equal`, `less`, `less_equal`
- Six tensor-scalar compare ops (same API names)
- Scalar-left rewrite at the API layer
- Bool output end-to-end: portable dtype propagation + 910B hardware correctness
- Minimum workspace allocation extension to kernel xbuf assignment

Batch D does not include:

- Any op beyond the six compare semantics listed
- Scalar-scalar compare (TypeError)
- Shape-mismatched compare operands (ValueError)
- Workspace reuse or liveness analysis

## Architecture

Batch D stays entirely within the existing Phase 1/2 layered architecture:

```
API (api.pyx)
  ↓ scalar-left rewrite + dispatch
Ops (ops.pyx)
  CompareOp / CompareScalarOp
  ↓ emit()
ISA (isa.pyx)
  encode_compare() / encode_compare_scalar()
  ↓ dtype routing table
  vCompare / vCompareS opcodes
Kernel (kernel.pyx)
  workspace-aware xbuf assignment
```

## ISA layer

### Compare type constants

Six compare semantic type constants (separate from dtype and opcode):

- `CMP_EQ`
- `CMP_NE`
- `CMP_GT`
- `CMP_GE`
- `CMP_LT`
- `CMP_LE`

These represent the comparison semantics encoded in the instruction's `type` field. They are distinct from the opcode (which selects the SIMD handler by dtype).

### Opcode routing tables

Two new routing tables:

**`COMPARE_OPCODE_TABLE`** — maps `dtype -> vSimdInsnID`:

- `DTYPE_F32  -> V_CMP`
- `DTYPE_FP16 -> V_CMP_FP16`

**`COMPARE_SCALAR_OPCODE_TABLE`** — maps `dtype -> vSimdInsnID`:

- `DTYPE_F32  -> V_CMPS`
- `DTYPE_FP16 -> V_CMPS_FP16`

The compare semantic type is NOT part of the routing key; it is a separate field encoded into the instruction payload.

### `encode_compare()` — `vCompare` layout

Matches upstream `vCompare::Encode` in `isa.h`:

```
pc[0] = make_simd_head(opcode, (cmp_type << 18) | xn, 2)
pc[1] = count << 49 | compact_x(ws) << 36 | xd << 18 | xm
```

Key differences from `vBinary`:
- `count` is **15 bits** (not 16)
- `xd` and `xm` are full 18-bit addresses in payload
- workspace `ws` is packed via `compact_x` (`ws >> 5`)
- `cmp_type` goes in the ext field alongside `xn`

### `encode_compare_scalar()` — `vCompareS` layout

Matches upstream `vCompareS::Encode`:

```
pc[0] = make_simd_head(opcode, (cmp_type << 18) | xn, 3)
pc[1] = count << 48 | ws << 18 | xd
pc[2] = scalar_bits
```

Key differences from `vCompare`:
- Instruction size is **3 words** (not 2)
- `ws` is a full 18-bit address (not compact)
- scalar goes in `pc[2]` as raw bits

### Scalar bit packing

Same as Batch C:

- `DTYPE_F32` → raw IEEE754 32-bit bits
- `DTYPE_FP16` → raw 16-bit half bits in low 16 bits of `scalar_bits`
- Unsupported dtype → `NotImplementedError`

## Ops layer

### `FlexOp.workspace_slots()` protocol

`FlexOp` gets a new `workspace_slots()` method:

```python
def workspace_slots(self):
    return 0
```

This is the default. Subclasses override it to declare their workspace needs.

This is the minimal protocol needed for Batch D and Batch E (select). No further abstraction is introduced now.

### `CompareOp(FlexOp)`

For tensor-tensor compare.

- Constructor: `(cmp_type, lhs, rhs)`
- `normalize()`: validates lhs/rhs shapes match; sets output shape = input shape; sets output `type_id = DTYPE_BOOL`
- `workspace_slots()`: returns 1
- `emit(code, relocs)`: emits one `vCompare` 2-word instruction using `encode_compare()`

Holds: `cmp_type`, `lhs`, `rhs`, result `xbuf`, workspace `xbuf`.

### `CompareScalarOp(FlexOp)`

For tensor-scalar compare.

- Constructor: `(cmp_type, src, scalar)`
- `normalize()`: sets output shape = `src.shape_ref`; sets output `type_id = DTYPE_BOOL`
- `workspace_slots()`: returns 1
- `emit(code, relocs)`: emits one `vCompareS` 3-word instruction using `encode_compare_scalar()`

Holds: `cmp_type`, `src`, `scalar`, result `xbuf`, workspace `xbuf`.

## Kernel layer

### Workspace-aware xbuf assignment

Current monotonic xbuf assignment allocates one slot per op. Batch D extends this to:

1. Allocate result `xbuf` for the op (as before)
2. Query `op.workspace_slots()`
3. Allocate that many additional xbuf slots for workspace
4. Assign workspace xbufs to the op

This remains monotonic and sequential. No reuse. No liveness analysis. The change is minimal: add a loop that allocates extra slots when `workspace_slots() > 0`.

## API layer

### Six public compare methods

Add to `Kernel`:

- `equal(a, b)`
- `not_equal(a, b)`
- `greater(a, b)`
- `greater_equal(a, b)`
- `less(a, b)`
- `less_equal(a, b)`

All route through a `_compare_dispatch` helper (analogous to `_binary_dispatch` from Batch C).

### Scalar-left rewrite rule

For symmetric ops (EQ, NE), scalar-left is equivalent to tensor-scalar:
- `equal(s, x)` → `equal(x, s)`
- `not_equal(s, x)` → `not_equal(x, s)`

For ordered ops, scalar-left flips the direction:
- `greater(s, x)` → `less(x, s)`
- `greater_equal(s, x)` → `less_equal(x, s)`
- `less(s, x)` → `greater(x, s)`
- `less_equal(s, x)` → `greater_equal(x, s)`

This rewrite happens fully at the API layer. Internal op classes only see tensor-scalar inputs.

### Error contract

- Both scalars → `TypeError`
- Tensor shape mismatch → `ValueError` (in `CompareOp.normalize()`)
- Unsupported dtype/op → `NotImplementedError`

## File map

### Files to modify

- `candle_dvm/isa.pyx`
  - Add `CMP_*` type constants
  - Add `COMPARE_OPCODE_TABLE` and `COMPARE_SCALAR_OPCODE_TABLE`
  - Add `encode_compare()` and `encode_compare_scalar()`

- `candle_dvm/isa.pxd`
  - Export `encode_compare()` and `encode_compare_scalar()`
  - Document new constants

- `candle_dvm/ops.pyx`
  - Add `workspace_slots()` default to `FlexOp`
  - Add `CompareOp` and `CompareScalarOp`
  - Add `CMP_*` re-exports

- `candle_dvm/ops.pxd`
  - Declare `FlexOp.workspace_slots()`
  - Declare `CompareOp` and `CompareScalarOp`

- `candle_dvm/kernel.pyx`
  - Extend xbuf assignment to be workspace-aware

- `candle_dvm/api.pyx`
  - Add six compare methods
  - Add `_compare_dispatch` internal helper
  - Add scalar-left rewrite

- `tests/test_isa.py`
  - CMP_* constants, routing tables, encode helpers bit-layout tests

- `tests/test_ops.py`
  - CompareOp / CompareScalarOp normalize, emit, workspace, bool dtype

- `tests/test_api.py`
  - Six compare API methods, scalar-left rewrite, error paths

- `tests/test_add.py`
  - 910B end-to-end compare tests (fp32 + fp16, tensor-tensor + tensor-scalar)

## Implementation order

1. CMP_* constants + routing tables + ISA tests
2. `encode_compare()` + ISA encode tests
3. `encode_compare_scalar()` + ISA encode tests
4. `workspace_slots()` protocol on `FlexOp`
5. Workspace-aware xbuf assignment in `kernel.pyx`
6. `CompareOp` + portable tests
7. `CompareScalarOp` + portable tests
8. API compare methods + scalar-left rewrite + API tests
9. 910B end-to-end verification
10. Full suite verification + final commit

## Testing strategy

### ISA portable tests

- `CMP_*` constant values match upstream `CompareType` enum
- `COMPARE_OPCODE_TABLE` f32/fp16 routing
- `COMPARE_SCALAR_OPCODE_TABLE` f32/fp16 routing
- `encode_compare()` 2-word layout: ext field, count (15-bit), ws placement, xd/xm
- `encode_compare_scalar()` 3-word layout: ext field, count (16-bit), ws, xd, scalar word
- compare type field encoding in ext

### Ops / API portable tests

- `CompareOp.normalize()` propagates bool dtype
- `CompareScalarOp.normalize()` propagates bool dtype
- `workspace_slots()` returns 1 for both compare op classes
- `CompareOp.emit()` emits 2 words, workspace field correct
- `CompareScalarOp.emit()` emits 3 words, workspace and scalar fields correct
- All six compare API methods route correctly
- Scalar-left rewrite for symmetric and ordered ops
- Both-scalars raises TypeError
- Shape mismatch raises ValueError
- Unsupported dtype raises NotImplementedError

### 910B end-to-end tests

- tensor-tensor: equal, greater, less for fp32 and fp16 vs NumPy
- tensor-scalar: equal, greater for fp32 and fp16 vs NumPy
- scalar-left: less(1.0, x)