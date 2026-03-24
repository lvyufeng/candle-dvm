# candle-dvm Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone `candle-dvm` package that reimplements the DVM host runtime in Cython and runs an end-to-end `add` example on 910B without importing upstream `dvm` / `_dvm_py.so`.

**Architecture:** Recreate DVM host responsibilities layer by layer: package/build scaffolding, reference-trace capture, binary packaging, ISA/Code encoding, runtime loading, NDObject graph construction, `VKernelS` codegen, then the Python-facing `Kernel` / `PyKernel` API. Keep phase 1 narrow: only the static vector path for `load + add + store`, explicit H2D/D2H copies, and hardware validation on 910B.

**Tech Stack:** Python 3.11+, Cython 3, setuptools, pytest, NumPy, Ascend ACL/RT runtime libraries, prebuilt DVM C220 device binary

---

## Before You Start

- Work from repo root: `/home/dndx/lvyufeng/candle-dvm`
- Read the approved spec first: `docs/superpowers/specs/2026-03-24-candle-dvm-design.md`
- Reference upstream DVM files while implementing:
  - `/home/dndx/lvyufeng/dvm/examples/01_add.py`
  - `/home/dndx/lvyufeng/dvm/src/isa.h`
  - `/home/dndx/lvyufeng/dvm/src/code.h`
  - `/home/dndx/lvyufeng/dvm/src/system.cc`
  - `/home/dndx/lvyufeng/dvm/python/dvm/_dvm_py.pyi`
  - `/home/dndx/lvyufeng/dvm/prebuild/g_vkernel_c220_bin`
- Use @superpowers:test-driven-development for every task.
- Task 8 must not start until Task 2 has committed `tests/fixtures/upstream_add_trace.txt`; Task 8 reads that fixture as the source of truth for `block_dim`.
- After each task, run only the tests listed in that task, then commit immediately.
- Do not import upstream `dvm` anywhere in `candle_dvm/`; only the trace-capture script may import it.

## File map

### Files to create

- `pyproject.toml` — project metadata, dependencies, pytest config
- `setup.py` — Cython extension definitions and package data registration
- `candle_dvm/__init__.py` — public exports and dtype aliases
- `candle_dvm/_acl.pxd` — ACL runtime declarations
- `candle_dvm/_rt.pxd` — RT runtime declarations and dynamic symbol typedefs
- `candle_dvm/device_bin.pyx` — package-data loader for `g_vkernel_c220.bin`
- `candle_dvm/isa.pxd` — ISA constants and function signatures
- `candle_dvm/isa.pyx` — bitfield constants and encode helpers
- `candle_dvm/code.pxd` — `RelocAddr` / `Code` declarations
- `candle_dvm/code.pyx` — code buffer, entry generation, reloc binding, debug header
- `candle_dvm/system.pxd` — `System` declarations
- `candle_dvm/system.pyx` — SoC mapping, runtime loading, binary registration, launch, device memory helpers
- `candle_dvm/ops.pxd` — NDObject hierarchy declarations
- `candle_dvm/ops.pyx` — `NDObject`, `NDAccess`, `NDLoad`, `NDStore`, `FlexOp`, `BinaryOp(add)`
- `candle_dvm/pass_.pyx` — no-op pass manager
- `candle_dvm/kernel.pxd` — kernel declarations
- `candle_dvm/kernel.pyx` — `VKernel`, `VectorKernel`, `VKernelS`, phase-1 block-dim constant, codegen path
- `candle_dvm/xkernel.pyx` — phase-1 `NotImplementedError` placeholders
- `candle_dvm/tuning.pyx` — phase-1 stubs
- `candle_dvm/api.pyx` — public `Kernel` graph builder API
- `candle_dvm/pykernel.py` — `PyKernel` and `@kernel` decorator
- `candle_dvm/data/README.md` — binary provenance and refresh instructions
- `candle_dvm/data/g_vkernel_c220.bin` — copied from upstream DVM prebuild output
- `scripts/capture_upstream_add_trace.py` — captures upstream `das()` output for the canonical `add` example
- `tests/conftest.py` — shared test helpers, 910B skip markers, fixture readers
- `tests/test_import.py` — package import smoke test
- `tests/test_trace_fixture.py` — validates upstream trace capture and parser
- `tests/test_binary_resources.py` — validates packaged binary resources
- `tests/test_isa.py` — ISA encode tests
- `tests/test_code.py` — code buffer / entry / reloc tests
- `tests/test_system.py` — hardware-dependent runtime loader test
- `tests/test_ops.py` — NDObject normalize/emit tests
- `tests/test_kernel.py` — `VKernelS` codegen tests
- `tests/test_api.py` — public `Kernel` graph API tests
- `tests/test_add.py` — hardware-dependent end-to-end decorator test
- `tests/fixtures/upstream_add_trace.txt` — committed output from upstream DVM `das()`
- `tests/fixtures/README.md` — explains how the fixture was generated
- `examples/01_add.py` — package-local phase-1 demo

### Files that will be modified repeatedly

- `setup.py`
- `candle_dvm/__init__.py`
- `tests/conftest.py`

## Task order

The order matters. Do not skip ahead. Later tasks depend on concrete artifacts from earlier ones.

---

### Task 1: Bootstrap the package and editable build

**Files:**
- Create: `pyproject.toml`
- Create: `setup.py`
- Create: `candle_dvm/__init__.py`
- Create: `tests/test_import.py`

- [ ] **Step 1: Write the failing import smoke test**

```python
# tests/test_import.py

def test_imports_package():
    import candle_dvm
    assert hasattr(candle_dvm, "__file__")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_import.py::test_imports_package -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'candle_dvm'`

- [ ] **Step 3: Add the minimal package/build files**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68.0", "wheel", "cython>=3.0"]
build-backend = "setuptools.build_meta"

[project]
name = "candle-dvm"
version = "0.1.0"
description = "Cython host runtime for DVM-compatible Ascend execution"
readme = "README.md"
requires-python = ">=3.11"
dependencies = ["numpy>=1.24,<2"]

[project.optional-dependencies]
test = ["pytest>=7.2,<9"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"
```

```python
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize([], compiler_directives={"language_level": "3"})

setup(
    packages=["candle_dvm"],
    ext_modules=ext_modules,
)
```

```python
# candle_dvm/__init__.py
__all__ = []
```

- [ ] **Step 4: Install editable package and rerun the test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_import.py::test_imports_package -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml setup.py candle_dvm/__init__.py tests/test_import.py
git commit -m "build: bootstrap candle-dvm package"
```

---

### Task 2: Capture the upstream DVM `add` trace

**Files:**
- Create: `tests/conftest.py`
- Create: `scripts/capture_upstream_add_trace.py`
- Create: `tests/fixtures/README.md`
- Create: `tests/test_trace_fixture.py`
- Create: `tests/fixtures/upstream_add_trace.txt`

- [ ] **Step 1: Write the failing fixture validation test**

```python
# tests/test_trace_fixture.py
from pathlib import Path
import re


def test_upstream_add_trace_fixture_exists_and_has_header():
    text = Path("tests/fixtures/upstream_add_trace.txt").read_text()
    first = text.splitlines()[0]
    assert first.startswith("// target=")
    assert re.search(r"block_dim=\d+", first)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_trace_fixture.py::test_upstream_add_trace_fixture_exists_and_has_header -v
```

Expected: FAIL with `FileNotFoundError`

- [ ] **Step 3: Add the capture script and fixture docs**

```python
# scripts/capture_upstream_add_trace.py
from pathlib import Path
import numpy as np
import dvm


@dvm.kernel
def upstream_add(k, x, y):
    a = k.load(x, dvm.float32)
    b = k.load(y, dvm.float32)
    c = k.add(a, b)
    return k.store(c)


def main() -> None:
    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    upstream_add.codegen()
    text = upstream_add.das()
    out = Path("tests/fixtures/upstream_add_trace.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

```markdown
# tests/fixtures/README.md

`upstream_add_trace.txt` is the canonical DVM `das()` output for the phase-1 `add` example on the known-good 910B environment. Regenerate it with:

PYTHONPATH=/home/dndx/lvyufeng/dvm/python python scripts/capture_upstream_add_trace.py
```

- [ ] **Step 4: Generate the fixture and rerun the test**

Run:

```bash
PYTHONPATH=/home/dndx/lvyufeng/dvm/python python scripts/capture_upstream_add_trace.py && python -m pytest tests/test_trace_fixture.py::test_upstream_add_trace_fixture_exists_and_has_header -v
```

Expected:
- script prints `wrote tests/fixtures/upstream_add_trace.txt`
- pytest PASS

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py scripts/capture_upstream_add_trace.py tests/fixtures/README.md tests/fixtures/upstream_add_trace.txt tests/test_trace_fixture.py
git commit -m "test: capture upstream add trace fixture"
```

---

### Task 3: Package the C220 device binary and resource loader

**Files:**
- Modify: `setup.py`
- Create: `candle_dvm/device_bin.pyx`
- Create: `candle_dvm/data/README.md`
- Create: `candle_dvm/data/g_vkernel_c220.bin`
- Create: `tests/test_binary_resources.py`

- [ ] **Step 1: Write the failing binary resource test**

```python
# tests/test_binary_resources.py
from candle_dvm.device_bin import load_c220_binary


def test_c220_binary_is_packaged_and_nonempty():
    data = load_c220_binary()
    assert isinstance(data, bytes)
    assert len(data) > 0
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_binary_resources.py::test_c220_binary_is_packaged_and_nonempty -v
```

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `candle_dvm.device_bin`

- [ ] **Step 3: Add the loader and package data**

```python
# candle_dvm/device_bin.pyx
from importlib import resources


def load_c220_binary() -> bytes:
    return resources.files("candle_dvm.data").joinpath("g_vkernel_c220.bin").read_bytes()
```

```python
# setup.py (update)
setup(
    packages=["candle_dvm", "candle_dvm.data"],
    package_data={"candle_dvm.data": ["g_vkernel_c220.bin", "README.md"]},
    ext_modules=cythonize([
        Extension("candle_dvm.device_bin", ["candle_dvm/device_bin.pyx"]),
    ], compiler_directives={"language_level": "3"}),
)
```

```markdown
# candle_dvm/data/README.md

`g_vkernel_c220.bin` is copied from `/home/dndx/lvyufeng/dvm/prebuild/g_vkernel_c220_bin` for phase 1 validation on 910B. Refresh it by copying from a known-good upstream DVM build.
```

Copy the binary exactly once:

```bash
cp /home/dndx/lvyufeng/dvm/prebuild/g_vkernel_c220_bin candle_dvm/data/g_vkernel_c220.bin
```

- [ ] **Step 4: Reinstall editable package and rerun the test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_binary_resources.py::test_c220_binary_is_packaged_and_nonempty -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/device_bin.pyx candle_dvm/data/README.md candle_dvm/data/g_vkernel_c220.bin tests/test_binary_resources.py
git commit -m "build: package c220 device binary"
```

---

### Task 4: Implement ISA constants and encode helpers

**Files:**
- Create: `candle_dvm/isa.pxd`
- Create: `candle_dvm/isa.pyx`
- Create: `tests/test_isa.py`

- [ ] **Step 1: Write the failing ISA tests**

```python
# tests/test_isa.py
from candle_dvm import isa


def test_access_opcode_constants_match_upstream():
    assert isa.V_LOAD == 0
    assert isa.V_STORE == 10


def test_simd_opcode_constants_match_upstream():
    assert isa.V_ADD == 18


def test_make_acc_head_packs_fields():
    head = isa.make_acc_head(isa.V_LOAD, 3, 2)
    expected = (3 << isa.V_M_HEAD_EXT_OFFSET) | (2 << isa.V_M_HEAD_SIZE_OFFSET)
    assert head == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_isa.py -v
```

Expected: FAIL because `candle_dvm.isa` does not exist

- [ ] **Step 3: Implement the minimal ISA layer**

```cython
# candle_dvm/isa.pyx
from libc.stdint cimport uint64_t

cdef public int V_LOAD = 0
cdef public int V_STORE = 10
cdef public int V_ADD = 18

cdef public int V_HEAD_SIMD_FLAG_OFFSET = 0
cdef public int V_HEAD_ID_OFFSET = 48
cdef public int V_HEAD_EXT_OFFSET = 22
cdef public int V_HEAD_SIZE_OFFSET = 7
cdef public int V_M_HEAD_EXT_OFFSET = 14
cdef public int V_M_HEAD_SIZE_OFFSET = 4
cdef public int V_ENTRY_TYPE_V = 0
cdef public int V_ENTRY_CODE_SIZE_OFFSET = 8
cdef public int V_ENTRY_V_TILE_TAIL_OFFSET = 32
cdef public int V_ENTRY_V_TILE_BODY_OFFSET = 40

cpdef uint64_t make_acc_head(uint64_t opcode, uint64_t ext, uint64_t size):
    return (ext << V_M_HEAD_EXT_OFFSET) | (size << V_M_HEAD_SIZE_OFFSET)

cpdef uint64_t make_simd_head(uint64_t opcode, uint64_t ext, uint64_t size):
    return (ext << V_HEAD_EXT_OFFSET) | (size << V_HEAD_SIZE_OFFSET) | (1 << V_HEAD_SIMD_FLAG_OFFSET)
```

- [ ] **Step 4: Reinstall and rerun the tests**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_isa.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/isa.pxd candle_dvm/isa.pyx tests/test_isa.py
git commit -m "feat: add phase-1 isa helpers"
```

---

### Task 5: Implement `Code`, `RelocAddr`, and entry-word generation

**Files:**
- Create: `candle_dvm/code.pxd`
- Create: `candle_dvm/code.pyx`
- Create: `tests/test_code.py`

- [ ] **Step 1: Write the failing code-buffer tests**

```python
# tests/test_code.py
from candle_dvm.code import Code, RelocAddr


def test_gen_entry_v_matches_spec_formula():
    entry = Code.gen_entry_v(tile_num=8, block_dim=4, data_size=32)
    assert entry & 0x7 == 0  # V_ENTRY_TYPE_V


def test_bind_relocs_overwrites_u64_slots():
    code = Code(capacity=64)
    slot = code.append_u64(0)
    reloc = RelocAddr(slot)
    code.relocs.append(reloc)
    code.bind_relocs([0x1234])
    assert code.read_u64_at(slot) == 0x1234
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_code.py -v
```

Expected: FAIL because `candle_dvm.code` does not exist

- [ ] **Step 3: Implement the minimal `Code` layer**

```cython
# candle_dvm/code.pyx
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memset
cimport candle_dvm.isa as isa

cdef class RelocAddr:
    def __cinit__(self, size_t offset=0):
        self.offset = offset

cdef class Code:
    def __cinit__(self, size_t capacity=4096):
        self.capacity = capacity
        self.data = <unsigned char*>malloc(capacity)
        memset(self.data, 0, capacity)
        self.data_size = 0
        self.relocs = []

    @staticmethod
    cpdef uint64_t gen_entry_v(uint64_t tile_num, uint64_t block_dim, uint64_t data_size):
        cdef uint64_t block_tile = (tile_num + block_dim - 1) // block_dim
        cdef uint64_t block_tail = block_dim * block_tile - tile_num
        return isa.V_ENTRY_TYPE_V | ((data_size // 8) << isa.V_ENTRY_CODE_SIZE_OFFSET) | (block_tail << isa.V_ENTRY_V_TILE_TAIL_OFFSET) | (block_tile << isa.V_ENTRY_V_TILE_BODY_OFFSET)
```

Add `append_u64`, `bind_relocs`, `read_u64_at`, and a `debug_header()` helper returning `(target, block_dim, data_size)`.

- [ ] **Step 4: Reinstall and rerun the tests**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_code.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/code.pxd candle_dvm/code.pyx tests/test_code.py
git commit -m "feat: add phase-1 code buffer and reloc support"
```

---

### Task 6: Implement `System` runtime loading and device-memory helpers

**Files:**
- Create: `candle_dvm/_acl.pxd`
- Create: `candle_dvm/_rt.pxd`
- Create: `candle_dvm/system.pxd`
- Create: `candle_dvm/system.pyx`
- Modify: `tests/conftest.py`
- Create: `tests/test_system.py`

- [ ] **Step 1: Write the failing hardware-dependent system test**

```python
# tests/test_system.py
import pytest
from candle_dvm.system import System


@pytest.mark.requires_910b
def test_system_initializes_and_loads_c220_binary():
    sys = System()
    sys.init(0)
    assert sys.arch_name() == "c220"
    assert sys.has_vector_handle()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_system.py::test_system_initializes_and_loads_c220_binary -v
```

Expected: FAIL because `candle_dvm.system` does not exist

- [ ] **Step 3: Implement the minimal runtime layer**

```cython
# candle_dvm/system.pyx
from posix.dlfcn cimport dlopen, dlsym, dlclose, RTLD_LAZY, RTLD_LOCAL
from candle_dvm.device_bin import load_c220_binary

cdef class System:
    cpdef void init(self, int device_id=0):
        self.device_id = device_id
        self._detect_soc()
        if not self._load_rt():
            self._load_acl()
        self._register_c220_binary()
```

Also implement:
- `soc_name()` / `arch_name()`
- `has_vector_handle()`
- `malloc_device()` / `free_device()` wrappers
- `create_stream()` / `destroy_stream()`
- launch-time `ffts_addr` patching in this layer, not in `kernel.pyx`

In `tests/conftest.py`, add:

```python
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_910b: requires 910B hardware")
```

- [ ] **Step 4: Reinstall and rerun the hardware test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_system.py::test_system_initializes_and_loads_c220_binary -v
```

Expected: PASS on the current 910B machine

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/_acl.pxd candle_dvm/_rt.pxd candle_dvm/system.pxd candle_dvm/system.pyx tests/conftest.py tests/test_system.py
git commit -m "feat: add runtime loader for 910b"
```

---

### Task 7: Implement the NDObject subset and no-op pass manager

**Files:**
- Create: `candle_dvm/ops.pxd`
- Create: `candle_dvm/ops.pyx`
- Create: `candle_dvm/pass_.pyx`
- Create: `tests/test_ops.py`

- [ ] **Step 1: Write the failing ops tests**

```python
# tests/test_ops.py
from candle_dvm.ops import NDLoad, NDStore, BinaryOp, DTYPE_F32, BIN_ADD


def test_binary_add_normalize_preserves_shape_and_dtype():
    a = NDLoad((32, 32), DTYPE_F32, 0)
    b = NDLoad((32, 32), DTYPE_F32, 1)
    op = BinaryOp(BIN_ADD, a, b)
    op.normalize(None)
    assert op.shape() == (32, 32)
    assert op.dtype() == DTYPE_F32


def test_emit_appends_vector_instruction_words():
    """Verify that emit() writes non-zero instruction words into the code buffer."""
    from candle_dvm.code import Code
    a = NDLoad((32, 32), DTYPE_F32, 0)
    b = NDLoad((32, 32), DTYPE_F32, 1)
    op = BinaryOp(BIN_ADD, a, b)
    op.normalize(None)
    code = Code(capacity=4096)
    before = code.data_size
    op.emit(code)
    assert code.data_size > before
    assert code.read_u64_at(0) != 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ops.py -v
```

Expected: FAIL because `candle_dvm.ops` does not exist

- [ ] **Step 3: Implement the phase-1 op hierarchy**

```cython
# candle_dvm/ops.pyx
cdef public int DTYPE_F32 = 3
cdef public int BIN_ADD = 0

cdef class NDObject:
    cpdef tuple shape(self):
        return self.shape_ref

cdef class NDLoad(NDAccess):
    cpdef void normalize(self, object kernel):
        self.normalized = True

cdef class BinaryOp(FlexOp):
    cpdef void normalize(self, object kernel):
        if self.lhs.shape() != self.rhs.shape():
            raise ValueError("shape mismatch")
        self.shape_ref = self.lhs.shape()
        self.type_id = self.lhs.dtype()
```

- [ ] **Step 3.5: Add the no-op pass manager stub and import smoke test**

Create:

```cython
# candle_dvm/pass_.pyx
cpdef list run_passes(list objects):
    return objects
```

Extend `tests/test_ops.py` with:

```python
from candle_dvm.pass_ import run_passes


def test_noop_pass_manager_returns_same_list():
    objs = [object(), object()]
    assert run_passes(objs) is objs
```



Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_ops.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/ops.pxd candle_dvm/ops.pyx candle_dvm/pass_.pyx tests/test_ops.py
git commit -m "feat: add phase-1 ndobject graph ops"
```

---

### Task 8: Implement `VKernelS` and lock the 910B phase-1 block dimension from the upstream trace

**Files:**
- Create: `candle_dvm/kernel.pxd`
- Create: `candle_dvm/kernel.pyx`
- Create: `tests/test_kernel.py`

- [ ] **Step 1: Write the failing kernel tests**

```python
# tests/test_kernel.py
from pathlib import Path
import re
from candle_dvm.kernel import VKernelS
from candle_dvm.ops import NDLoad, NDStore, BinaryOp, DTYPE_F32, BIN_ADD


def _trace_block_dim() -> int:
    text = Path("tests/fixtures/upstream_add_trace.txt").read_text().splitlines()[0]
    return int(re.search(r"block_dim=(\d+)", text).group(1))


def test_vkernel_s_codegen_uses_upstream_phase1_block_dim():
    """Prerequisite: Task 2 must have committed tests/fixtures/upstream_add_trace.txt"""
    block_dim = _trace_block_dim()
    assert isinstance(block_dim, int) and block_dim > 0, "trace fixture is missing or malformed"
    k = VKernelS()
    a = NDLoad((32, 32), DTYPE_F32, 0)
    b = NDLoad((32, 32), DTYPE_F32, 1)
    out = NDStore(BinaryOp(BIN_ADD, a, b), 0)
    k.append(a)
    k.append(b)
    k.append(out.src)
    k.append(out)
    k.normalize()
    k.codegen()
    assert k.debug_header()["block_dim"] == _trace_block_dim()
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_kernel.py::test_vkernel_s_codegen_uses_upstream_phase1_block_dim -v
```

Expected: FAIL because `candle_dvm.kernel` does not exist

- [ ] **Step 3: Implement the minimal kernel layer**

```cython
# candle_dvm/kernel.pyx
cdef class VKernel:
    def __cinit__(self):
        self.objects = []
        self.code = Code(4096)

cdef class VKernelS(VectorKernel):
    cpdef void normalize(self):
        for obj in self.objects:
            obj.normalize(self)

    cpdef void codegen(self):
        self.code.append_u64(0)  # ffts_addr
        self.code.append_u64(0)  # entry
        for obj in self.objects:
            obj.emit(self)
        self.code.append_u64(0)
```

Then do the critical bring-up step:
- read `tests/fixtures/upstream_add_trace.txt`
- extract the real upstream `block_dim`
- hard-code it as `DEFAULT_VECTOR_BLOCK_DIM_910B = <captured value>` in `kernel.pyx`
- set `self.code.block_dim = DEFAULT_VECTOR_BLOCK_DIM_910B`
- expose `debug_header()` so the test can compare against the trace

- [ ] **Step 4: Reinstall and rerun the test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_kernel.py::test_vkernel_s_codegen_uses_upstream_phase1_block_dim -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/kernel.pxd candle_dvm/kernel.pyx tests/test_kernel.py
git commit -m "feat: add phase-1 static vector kernel"
```

---

### Task 9: Expose the public `Kernel` graph-builder API

**Files:**
- Create: `candle_dvm/api.pyx`
- Modify: `candle_dvm/__init__.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the failing public-API test**

```python
# tests/test_api.py
from candle_dvm import Kernel, float32


def test_public_kernel_builds_add_graph():
    k = Kernel()
    a = k.load((32, 32), float32)
    b = k.load((32, 32), float32)
    out = k.store(k.add(a, b))
    assert out.shape() == (32, 32)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_api.py::test_public_kernel_builds_add_graph -v
```

Expected: FAIL with `ImportError` for `Kernel` / `float32`

- [ ] **Step 3: Implement the `Kernel` API and exports**

```cython
# candle_dvm/api.pyx
cdef class Kernel:
    def __cinit__(self):
        self._kernel = VKernelS()

    cpdef load(self, shape, int dtype):
        obj = NDLoad(tuple(shape), dtype, len(self._inputs))
        self._inputs.append(obj)
        self._kernel.append(obj)
        return obj
```

```python
# candle_dvm/__init__.py
from candle_dvm.api import Kernel
from candle_dvm.ops import DTYPE_F32 as float32

__all__ = ["Kernel", "float32"]
```

Add `store`, `add`, and `codegen` forwarding methods.

- [ ] **Step 4: Reinstall and rerun the test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_api.py::test_public_kernel_builds_add_graph -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/api.pyx candle_dvm/__init__.py tests/test_api.py
git commit -m "feat: expose public kernel graph api"
```

---

### Task 10: Implement `PyKernel`, the decorator path, and the end-to-end `add` run

**Files:**
- Create: `candle_dvm/pykernel.py`
- Modify: `candle_dvm/__init__.py`
- Create: `tests/test_add.py`
- Create: `examples/01_add.py`
- Create: `candle_dvm/xkernel.pyx`
- Create: `candle_dvm/tuning.pyx`

- [ ] **Step 1: Write the failing end-to-end test**

```python
# tests/test_add.py
import sys
import numpy as np
import pytest
import candle_dvm as dvm


@pytest.mark.requires_910b
def test_decorated_add_runs_without_importing_upstream_dvm():
    assert "dvm" not in sys.modules
    assert "_dvm_py" not in sys.modules

    @dvm.kernel()
    def my_add(k, x, y):
        a = k.load(x.shape, dvm.float32)
        b = k.load(y.shape, dvm.float32)
        return k.store(k.add(a, b))

    x = np.full([32, 32], 0.1, np.float32)
    y = np.full([32, 32], 0.3, np.float32)
    z = my_add(x, y)

    np.testing.assert_allclose(z, x + y, rtol=1e-5, atol=1e-5)
    assert "dvm" not in sys.modules
    assert "_dvm_py" not in sys.modules
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m pytest tests/test_add.py::test_decorated_add_runs_without_importing_upstream_dvm -v
```

Expected: FAIL because `kernel()` / `PyKernel` do not exist yet

- [ ] **Step 3: Implement the execution path**

```python
# candle_dvm/pykernel.py
import numpy as np
from candle_dvm.api import Kernel
from candle_dvm.system import get_system


class PyKernel:
    def __init__(self, kernel_type="vector", device_id=0):
        self.kernel = Kernel()
        self.system = get_system()
        self.system.init(device_id)
        self._built = False
```

Implement:
- decorator-based symbolic graph build
- numpy input binding
- `aclrtMalloc` / `aclrtMemcpy` H2D through `system.pyx`
- reloc patch + launch
- D2H copy for outputs
- strict assertion that no code path imports upstream `dvm`

- [ ] **Step 3.6: Add the `xkernel.pyx` placeholder and import failure test**

Create:

```cython
# candle_dvm/xkernel.pyx
raise NotImplementedError("cube/mix kernels are not implemented in phase 1")
```

Add a small assertion to the test module:

```python
import pytest


def test_xkernel_placeholder_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        __import__("candle_dvm.xkernel")
```
```

Add the demo script mirroring upstream shape and values:

```python
# examples/01_add.py
import numpy as np
import candle_dvm as dvm


@dvm.kernel()
def my_add(k, x, y):
    a = k.load(x.shape, dvm.float32)
    b = k.load(y.shape, dvm.float32)
    return k.store(k.add(a, b))
```

- [ ] **Step 4: Reinstall and rerun the end-to-end test**

Run:

```bash
python -m pip install -e '.[test]' && python -m pytest tests/test_add.py::test_decorated_add_runs_without_importing_upstream_dvm -v
```

Expected: PASS on 910B

- [ ] **Step 5: Commit**

```bash
git add setup.py candle_dvm/pykernel.py candle_dvm/__init__.py candle_dvm/xkernel.pyx candle_dvm/tuning.pyx tests/test_add.py examples/01_add.py
git commit -m "feat: run phase-1 add end to end"
```

---

### Task 11: Run the full phase-1 verification set and freeze the baseline

**Files:**
- Modify: `tests/conftest.py` (only if a missing skip/helper is discovered)
- Modify: `examples/01_add.py` (only if the demo is broken)

- [ ] **Step 1: Write the failing aggregate verification checklist into the task log**

Use this exact checklist:

```text
- portable tests pass
- hardware tests pass on 910B
- example script runs
- no upstream dvm import in candle_dvm package
```

- [ ] **Step 2: Run the portable tests**

Run:

```bash
python -m pytest tests/test_import.py tests/test_trace_fixture.py tests/test_binary_resources.py tests/test_isa.py tests/test_code.py tests/test_ops.py tests/test_kernel.py tests/test_api.py -v
```

Expected: PASS

- [ ] **Step 3: Run the hardware tests and example**

Run:

```bash
python -m pytest tests/test_system.py tests/test_add.py -v && python examples/01_add.py
```

Expected:
- hardware pytest PASS on 910B
- example prints correct `float32` arrays near `0.4`

- [ ] **Step 4: Verify upstream DVM is not imported by the package**

Run:

```bash
python - <<'PY'
import sys
import candle_dvm
assert 'dvm' not in sys.modules
assert '_dvm_py' not in sys.modules
print('import isolation OK')
PY
```

Expected: prints `import isolation OK`

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py examples/01_add.py
git commit -m "test: verify candle-dvm phase-1 baseline"
```

---

## Notes for the implementing agent

- Do not “improve” the design by adding cube/mix support in phase 1.
- Do not add CPU fallbacks for missing NPU behavior.
- Use the upstream `upstream_add_trace.txt` fixture as the single source of truth for the phase-1 launch shape and `block_dim`.
- Keep `debug_header()` / similar debug helpers until at least phase 2; they are needed for parity checks.
- If any Cython extension import fails after adding a new `.pyx`, reinstall editable package before rerunning pytest.

## Final verification commands

When all tasks are done, these should all pass from repo root:

```bash
python -m pip install -e '.[test]'
python -m pytest tests/test_import.py tests/test_trace_fixture.py tests/test_binary_resources.py tests/test_isa.py tests/test_code.py tests/test_ops.py tests/test_kernel.py tests/test_api.py -v
python -m pytest tests/test_system.py tests/test_add.py -v
python examples/01_add.py
```
