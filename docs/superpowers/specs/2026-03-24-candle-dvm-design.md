# candle-dvm Design

Date: 2026-03-24
Status: Draft approved for implementation planning

## Summary

`candle-dvm` is a new standalone package that reimplements the DVM host runtime in Cython while remaining compatible with the existing DVM device-side VM model. The first milestone targets Ascend 910B and proves the approach by running an `01_add.py`-level example end-to-end without depending on the original DVM `_dvm_py.so` Python extension.

The project will:

- reimplement DVM host-side graph building, code generation, and runtime launch logic in Cython
- reuse precompiled Ascend device VM binaries and Ascend runtime libraries
- preserve compatibility with the existing DVM bytecode and launch model
- start as a standalone runtime package, not a Candle backend integration
- prioritize architectural fidelity and layer-by-layer translation over a minimal hacky prototype

## Goals

### Primary goals

- Build a standalone `candle-dvm` package with a Cython host runtime.
- Stay compatible with the current DVM device-side VM and bytecode model.
- Run on the current 910B environment where DVM already works.
- Achieve an end-to-end `add` demo equivalent to DVM's `examples/01_add.py`.
- Avoid introducing new C++ code in `candle-dvm`.

### Secondary goals

- Preserve a structure that can later expand to 910A and 310B.
- Preserve a structure that can later integrate with Candle as a supplemental backend.
- Keep the code organized so future support for vector, cube, mix, split, and eager modes can be added incrementally.

### Non-goals for phase 1

- Replacing the device-side `.cce` VM kernels.
- Replacing Ascend runtime dependencies.
- Supporting all DVM ops or all kernel modes.
- Implementing zero-copy Candle integration.
- Reproducing DVM's full optimization/pass pipeline in phase 1.

## Architecture overview

The system is organized as a layered translation of DVM host runtime responsibilities.

### Layer 0: External runtime boundary

This layer is not reimplemented. It consists of:

- Ascend runtime shared libraries such as `libruntime.so` and `libascendcl.so`
- precompiled DVM-compatible device binaries, initially `g_vkernel_c220.bin` and later `g_vkernel_c310.bin`

This boundary remains the execution substrate for `candle-dvm`.

### Layer 1: System / runtime layer

This layer handles:

- device selection
- SoC detection
- architecture mapping such as 910B -> C220
- runtime symbol loading
- binary registration and function resolution
- stream creation/destruction
- kernel launch bridging

This layer will be implemented in Cython using `cdef extern` declarations for ACL APIs plus runtime dynamic loading for optional symbols.

### Layer 2: ISA / code layer

This layer reproduces the DVM bytecode format and host-side code buffer layout.

It is responsible for:

- opcode and bitfield definitions
- instruction encoding
- program entry/header encoding
- code buffer growth and memory ownership
- relocations for inputs, outputs, workspace, and future scalar/shape references
- launch handoff to the system layer

Phase 1 only requires the vector path, but the code structure will preserve room for cube and mix support.

### Layer 3: Ops / pass layer

This layer represents graph objects in a Cython NDObject hierarchy.

It is responsible for:

- graph object types such as load, store, and elementwise binary ops
- shape/dtype propagation
- code emission into the `Code` buffer
- future extension to more ops and optimization passes

Phase 1 will implement the full structural skeleton of the NDObject system, but only fully realize the subset needed for `add`.

### Layer 4: Kernel layer

This layer orchestrates:

- graph assembly
- normalize
- pass execution
- code generation
- launch

Phase 1 focuses on `VKernelS`, the static vector kernel path.

Other kernel modes such as dynamic vector, cube, mix, parallel, sequence, split, and eager will exist only as architectural placeholders in this phase.

### Layer 5: Public API layer

This layer exposes a Python-facing API analogous to DVM's current Python experience:

- a `Kernel` graph builder API
- a `PyKernel` helper
- a `@kernel` decorator
- dtype aliases such as `float32`

Phase 1 only needs enough public API surface to build and run the `add` example.

## Repository structure

The package should be organized as follows:

```text
candle-dvm/
├── setup.py
├── pyproject.toml
├── candle_dvm/
│   ├── __init__.py
│   ├── _acl.pxd
│   ├── _rt.pxd
│   ├── system.pxd
│   ├── system.pyx
│   ├── device_bin.pyx
│   ├── isa.pxd
│   ├── isa.pyx
│   ├── code.pxd
│   ├── code.pyx
│   ├── ops.pxd
│   ├── ops.pyx
│   ├── pass_.pyx
│   ├── kernel.pxd
│   ├── kernel.pyx
│   ├── xkernel.pyx
│   ├── tuning.pyx
│   ├── api.pyx
│   ├── pykernel.py
│   └── data/
│       ├── g_vkernel_c220.bin
│       └── README.md
├── tests/
│   ├── test_isa.py
│   ├── test_code.py
│   ├── test_system.py
│   ├── test_ops.py
│   ├── test_kernel.py
│   └── test_add.py
└── examples/
    └── 01_add.py
```

The `.pxd` files exist to allow direct Cython-to-Cython `cimport` usage between modules without paying Python object dispatch costs across internal layers.

## Detailed design

## Layer 1: System / runtime

### Responsibilities

The `System` layer must:

- detect the active SoC
- map SoC to the DVM architecture family
- choose the correct embedded binary payload
- load runtime symbols from Ascend libraries
- register the device binary
- resolve function handles such as vector and cube VM entrypoints
- create/destroy streams
- launch generated `Code`
- allocate and free device buffers used by the phase-1 numpy H2D/D2H path

### Core class

`system.pyx` will define a `System` Cython class with fields for:

- architecture metadata
- device/runtime metadata
- dynamic library handles
- runtime function pointers for RT mode and ACL mode
- binary handles
- resolved function handles
- optional temporary device workspace

A process-wide `g_system` singleton will mirror the original DVM structure and keep the phase 1 port simple.

### Runtime loading strategy

The system must attempt two launch paths:

1. RT path via `libruntime.so`
2. ACL path via `libascendcl.so`

The implementation should prefer the runtime path if available and complete, then fall back to the ACL path. This preserves compatibility with different CANN/runtime environments and mirrors DVM behavior.

The RT path is considered usable when all four of these symbols resolve from `libruntime.so`:

- `rtKernelLaunch`
- `rtGetC2cCtrlAddr`
- `rtDevBinaryRegister`
- `rtFunctionRegister`

If any one of those symbols fails to resolve, the RT path is abandoned and the ACL path is attempted. The ACL path requires all four of these symbols from `libascendcl.so`:

- `aclrtBinaryLoadFromData`
- `aclrtBinaryGetFunction`
- `aclrtLaunchKernelWithHostArgs`
- `aclrtGetHardwareSyncAddr`

If neither path succeeds, initialization raises a clear runtime-loading error.

### Binary loading

Device binary artifacts will be packaged as package data and loaded at runtime from `candle_dvm/data/`.

The first milestone only needs the binary for the architecture used by 910B (C220). The `data/` directory will include a `README.md` that documents:

- the expected file name and binary type for `g_vkernel_c220.bin`
- that the file is the precompiled DVM device-VM binary for the C220 architecture
- which upstream DVM sources it originates from (`vm_aiv.cce`, `vm_aic.cce`, and associated build output)
- how to obtain it from a known-good DVM build or regenerate it with the Ascend toolchain
- that milestone 1 validates only the C220 binary on 910B

The `g_vkernel_c310.bin` file is not included in phase 1; it will be added when C310 targets are supported. The code should still preserve the branching structure required for later C310-based targets.

### SoC mapping

The `System` layer will maintain a Python/Cython mapping from SoC names to architecture families.

Phase 1 requires 910B -> C220. The structure must also leave room for 910A and 310B mappings later.

### `ffts_addr` patch responsibility

The `System` layer is responsible for patching the first 64-bit word of the code buffer (`ffts_addr`) immediately before launch when the active launch path and target require it.

Concretely:

- on the RT path, `system.pyx` will call the resolved `rtGetC2cCtrlAddr` equivalent before `rtKernelLaunch`
- on the ACL path, `system.pyx` will call the resolved `aclrtGetHardwareSyncAddr` equivalent before `aclrtLaunchKernelWithHostArgs`
- the patched value is written to `code.data[0:8]` before dispatching the kernel launch API

This responsibility lives in Layer 1 rather than Layer 4 because it depends on runtime-specific symbol resolution.

## Layer 2: ISA / code

### ISA strategy

The Cython ISA layer will not mimic C++ structs literally. Instead it will encode instructions using:

- Cython enums
- inline packing helpers returning `uint64_t`
- symbolic constants describing bit offsets and masks

This keeps the implementation precise while staying idiomatic for Cython.

Phase 1 requires at minimum the following opcodes from DVM's ISA (values from `isa.h`):

Access instructions (`vAccInsnID`):

- `V_LOAD = 0`
- `V_STORE = 10`

SIMD instructions (`vSimdInsnID`):

- `V_ADD = 18` (fp32 binary add)

These numeric values and their bitfield encoding layouts must be verified against the original `isa.h` during implementation. The ISA layer should define these as named constants and provide encoding functions that produce bit-exact compatible instruction words.

### Code buffer model

The `Code` object will own a raw memory buffer. Each word is 64 bits (8 bytes). The layout is:

- bytes 0-7: `uint64_t ffts_addr` (hardware sync address, patched at launch time)
- bytes 8-15: `uint64_t entry` (program entry descriptor encoding kernel type, block dim, code offset/size, and flags)
- bytes 16+: encoded bytecode, visit programs, cube payloads, and metadata tables
- terminated by a `uint64_t` zero word

The total header size is 16 bytes (2 words). Phase 1 only needs vector instructions and a terminating zero instruction.

### Entry word layout

The `entry` word must be bit-compatible with DVM's `V_ENTRY_*` layout from `isa.h`.

Phase-1 implementation needs these fields:

- bits `[0:2]`: entry type, with `V_ENTRY_TYPE_V = 0`
- bit `3`: `V_ENTRY_FLAG_CUBE_MIX` (unused in phase 1)
- bit `4`: `V_ENTRY_FLAG_EXTERN_CODE` (unused in phase 1)
- bits `[8:19]`: code size in 64-bit words, via `V_ENTRY_CODE_SIZE_OFFSET = 8` and `V_ENTRY_CODE_SIZE_BITS = 12`
- bits `[32:39]`: vector tile tail, via `V_ENTRY_V_TILE_TAIL_OFFSET = 32` and `V_ENTRY_V_TILE_TAIL_BITS = 8`
- bits `[40:63]`: vector tile body, via `V_ENTRY_V_TILE_BODY_OFFSET = 40` and `V_ENTRY_V_TILE_BODY_BITS = 24`

For phase 1, the entry word must be generated through the same formula as DVM `Code::GenEntryV(tile_num, block_dim, data_size)`:

- `block_tile = ceil(tile_num / block_dim)`
- `block_tail = block_dim * block_tile - tile_num`
- `entry = V_ENTRY_TYPE_V | ((data_size / 8) << 8) | (block_tail << 32) | (block_tile << 40)`

The implementation should treat this formula as normative for milestone 1.

### Relocation model

Relocation entries will be represented explicitly and stored alongside the code buffer.

Each relocation will describe:

- which encoded address field to patch: a pointer to a `uint64_t` location inside the code buffer
- whether it refers to an input, output, workspace, or future scalar/shape object
- the logical index and optional offset

At launch time, `bind_relocs()` writes the device pointer value directly into the `uint64_t` slot identified by each relocation entry. No masking or bitfield manipulation is needed for address relocs; the entire 64-bit word is overwritten with the device address. This matches the original DVM relocation semantics.

This preserves compatibility with the original DVM execution model and avoids later redesign when dynamic shape support arrives.

### Target support in phase 1

Phase 1 will only support:

- vector target

The public structure of the `Code` object should still reserve the `target` field and related branching for future cube and mix support.

## Layer 3: Ops / pass

### NDObject hierarchy

The NDObject system will be represented with `cdef class` inheritance.

Phase 1 will fully implement only:

- `NDObject`
- `NDAccess`
- `NDLoad`
- `NDStore`
- `FlexOp`
- `BinaryOp` with `add`

Other object families will be scaffolded later without blocking the architecture.

### Shape and dtype representation

Phase 1 will not replicate DVM's full shape-domain machinery. Instead:

- shape is represented as a Python tuple of ints
- dtype is represented as a compact enum
- helpers compute `numel` and perform exact-shape validation

This is sufficient for the first `add` milestone and avoids premature complexity.

### Normalize behavior

For phase 1:

- `NDLoad` validates input metadata and exposes output shape/dtype
- `BinaryOp(add)` checks exact shape compatibility and dtype compatibility
- `NDStore` forwards source metadata to output metadata

### Emit behavior

Each implemented op appends instructions to the `Code` buffer and registers relocation entries when needed.

For the first milestone:

- `NDLoad` emits vector-load-related data and records an input relocation
- `BinaryOp(add)` emits a vector binary add instruction
- `NDStore` emits store-related data and records an output relocation

### Pass layer

The pass module will exist in phase 1 but behave as a no-op pass manager.

This preserves the eventual architecture while keeping the first milestone small. Once `add` runs successfully, the pass layer can absorb real optimizations incrementally.

### Memory management

The original DVM object pool will not be ported in phase 1.

Instead:

- Cython `cdef class` instances will be owned by Python/Cython object graphs
- kernel object lists will hold strong references

This reduces porting risk and keeps correctness ahead of micro-optimization. Object pooling can be revisited only if profiling shows it is necessary.

## Placeholder modules in phase 1

Two modules are present in the repository layout only to preserve future structure:

- `xkernel.pyx`: placeholder for cube, mix, parallel, sequence, split, and eager kernel orchestration. It is an empty stub or raises `NotImplementedError` in phase 1.
- `tuning.pyx`: placeholder for cube tuning and online/lazy tuning logic. It is an empty stub in phase 1.

They are intentionally included so future phases can expand the package without reshaping the module tree.

## Layer 4: Kernel

### Kernel hierarchy

Phase 1 kernel types:

- `VKernel` base
- `VectorKernel` intermediate base
- `VKernelS` fully implemented
- `VKernelD` placeholder

Other kernel modes such as cube, mix, parallel, sequence, split, and eager will be represented only as placeholders or stubs in phase 1.

### Code generation pipeline

`VKernelS` will follow this order:

1. gather build-time ops
2. normalize each op
3. run the no-op pass manager
4. initialize the `Code` buffer
5. reserve head words for `ffts_addr` and `entry`
6. emit each normalized op into the code buffer
7. append terminating zero instruction
8. finalize program entry metadata
9. set block dimension and vector target

### Block dimension in phase 1

Phase 1 does not hard-code `block_dim = 1`.

Instead, the first implementation must derive the exact `block_dim` for the `add` path by tracing a known-good original DVM run of the same example on 910B and mirroring the resulting launch configuration and `GenEntryV(tile_num, block_dim, data_size)` inputs. Until that trace is captured, `block_dim` remains an implementation-time parameter rather than a frozen constant in the spec.

This is intentionally conservative because a wrong block dimension can silently produce incorrect execution. A later phase can replace the traced value with a documented derived policy based on shape, architecture, and vector tiling requirements.

### Temporary buffer allocation

There will be no liveness-based reuse initially. The design should keep the liveness analysis hook in place so the slot allocator can later be replaced with a smarter reuse policy.

### Launch behavior

At launch time the kernel layer will:

- prepare input/output bindings
- ask the `Code` object to patch relocations
- delegate `ffts_addr` patching responsibility to the system layer
- forward the final code object to the system layer for execution

## Layer 5: Public API

### Public `Kernel` API

The `Kernel` API exposed from `api.pyx` will provide graph-building methods corresponding to the first milestone:

- `load`
- `store`
- `add`

Additional methods can exist as placeholders if that helps preserve the eventual public surface.

### `PyKernel` and decorator

The public Python experience should mirror the current DVM style closely enough to support a familiar example shape.

The decorator flow is:

- build symbolic graph by calling the user function once with symbolic placeholders
- on invocation, bind real numpy inputs
- allocate device buffers
- perform host-to-device copies
- codegen if needed
- launch
- copy output back to host numpy arrays
- return results

### Data movement in phase 1

The first milestone will use explicit host/device transfers.

Device memory allocation and deallocation will be handled by `system.pyx` using `aclrtMalloc` / `aclrtFree` (or the RT-path equivalents). The `PyKernel` or `Kernel.run()` flow is responsible for calling these at the appropriate lifecycle points.

The data movement sequence is:

- numpy input -> `aclrtMalloc` device buffer + `aclrtMemcpy` H2D
- launch using patched device addresses
- `aclrtMemcpy` D2H -> numpy return value
- `aclrtFree` device buffers after output is copied back

Zero-copy is not part of phase 1.

## Testing strategy

Testing will proceed bottom-up.

### Portable unit tests

These tests do not require Ascend hardware:

- `test_isa.py`: validate instruction encoding against expected bit patterns
- `test_code.py`: validate buffer growth, head layout, entry-word encoding, and reloc patching
- `test_ops.py`: validate shape/dtype propagation and verify the exact opcode field values of emitted instructions for load/add/store graphs, not just instruction counts
- `test_kernel.py`: validate normalize/codegen flow for a simple vector kernel using mocked launch boundaries

### Hardware-dependent integration tests

These tests require the target 910B machine and a working Ascend runtime environment:

- `test_system.py`: validate runtime loading and basic binary/function resolution on the target machine
- `test_add.py`: reproduce a DVM-style `01_add.py` path end-to-end using the new Cython host runtime

### Acceptance criteria for milestone 1

Milestone 1 is complete only when:

- the original DVM `_dvm_py.so` is not imported or linked into the running process
- a `@kernel`-decorated `add` example runs entirely through `candle-dvm`
- execution happens successfully on the current 910B environment
- output matches numpy addition within expected floating-point tolerance

`test_add.py` must assert process-level isolation by checking that importing and running `candle-dvm` does not import `dvm` or `_dvm_py`, for example by inspecting `sys.modules` before and after execution.

## Error handling

Phase 1 should prefer explicit, early failures over recovery logic.

Examples:

- unsupported SoC -> raise a clear initialization error
- runtime symbol missing -> raise a clear runtime-loading error
- unsupported op or kernel mode -> raise `NotImplementedError`
- shape mismatch -> raise `ValueError`
- launch/runtime failure -> raise a runtime-specific exception with the failing API name and status code

This keeps debugging straightforward during the translation phase.

## Expansion path after milestone 1

Once the `add` milestone is stable, the next likely extensions are:

1. more elementwise ops in the same vector path
2. fuller shape handling
3. real pass implementations
4. dynamic vector kernels
5. cube and mix kernels
6. 910A and 310B target adaptation
7. eventual Candle backend integration

The current design is intended to make these expansions additive rather than forcing a rewrite.

## Recommendation

Proceed with a strict layer-by-layer translation, but keep milestone 1 narrow:

- fully establish the layered architecture now
- fully implement only the vector static path needed for `add`
- preserve placeholders and module boundaries for later expansion

This gives `candle-dvm` a durable architecture without delaying the first proof point.
