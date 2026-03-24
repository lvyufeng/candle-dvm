"""PyKernel: high-level decorator and execution path for DVM kernels.

Provides the ``kernel`` decorator and ``PyKernel`` class that handle:
- Deferred graph construction on first call
- Codegen, relocation binding, H2D/D2H memcpy, and kernel launch

This is a pure-Python module (no Cython) for easy iteration.

Example
-------
>>> import numpy as np
>>> import candle_dvm as dvm
>>>
>>> @dvm.kernel()
... def my_add(k, x, y):
...     a = k.load(x.shape, dvm.float32)
...     b = k.load(y.shape, dvm.float32)
...     return k.store(k.add(a, b))
>>>
>>> z = my_add(np.ones((4, 4), np.float32), np.ones((4, 4), np.float32))
"""

import numpy as np


# Dtype enum -> numpy dtype mapping (matches ops.pyx DTYPE_* constants)
_DTYPE_TO_NP = {
    0: np.bool_,     # DTYPE_BOOL
    1: np.float16,   # DTYPE_FP16
    2: np.float16,   # DTYPE_BF16 (numpy has no bfloat16; use float16 as proxy)
    3: np.float32,   # DTYPE_F32
    4: np.int32,     # DTYPE_INT32
}


class PyKernel:
    """High-level kernel wrapper that manages graph build, codegen,
    device memory, and kernel launch.

    Parameters
    ----------
    kernel_type : str
        Target type (``"vector"``, ``"cube"``, ``"mix"``).  Phase 1 only
        supports ``"vector"``.
    device_id : int
        Device ordinal to run on.
    """

    def __init__(self, kernel_type="vector", device_id=0):
        self._kernel_type = kernel_type
        self._device_id = device_id
        self._func = None
        self._api_kernel = None   # api.Kernel instance
        self._built = False
        self._codegen_done = False

    def build(self, func):
        """Register the user function for deferred graph construction.

        The graph is not built until the first ``__call__``.
        """
        self._func = func

    def __call__(self, *args):
        """Execute the kernel: build graph (first call), codegen, H2D,
        launch, D2H, and return numpy result(s).
        """
        from candle_dvm.api import Kernel
        from candle_dvm.system import get_system

        sys = get_system()

        # ---- Step 1: build the symbolic graph on first invocation ----
        if not self._built:
            self._api_kernel = Kernel()
            # Call user function with the kernel and the numpy arrays.
            # The user function calls k.load(shape, dtype) etc.
            self._func(self._api_kernel, *args)
            self._built = True

        # ---- Step 2: codegen (once) ----
        if not self._codegen_done:
            self._api_kernel.codegen()
            self._codegen_done = True

        # ---- Step 3: get relocs and Code from the compiled kernel ----
        relocs = self._api_kernel.get_relocs()
        # The Code object is accessible via the public .code property
        code = self._api_kernel.code

        stream = sys.create_stream()
        dev_ptrs = []
        out_dev_ptrs = []
        try:
            # ---- Step 4: allocate device buffers for inputs + H2D ----
            for arr in args:
                nbytes = arr.nbytes
                dev_ptr = sys.malloc_device(nbytes)
                sys.memcpy_h2d(dev_ptr, arr.ctypes.data, nbytes, stream)
                dev_ptrs.append(dev_ptr)

            # ---- Step 5: allocate device buffers for outputs ----
            out_shapes = []
            out_dtypes = []
            for store_obj in self._api_kernel.outputs:
                shape = store_obj.shape_ref
                np_dtype = _DTYPE_TO_NP.get(store_obj.type_id, np.float32)
                nbytes = int(np.prod(shape)) * np.dtype(np_dtype).itemsize
                dev_ptr = sys.malloc_device(nbytes)
                out_dev_ptrs.append(dev_ptr)
                out_shapes.append(shape)
                out_dtypes.append(np_dtype)

            # ---- Step 6: bind relocations ----
            # Relocs are emitted in object order: load0, load1, ..., store0, store1, ...
            # Values must be in the same order: input device ptrs, then output device ptrs.
            all_values = list(dev_ptrs) + list(out_dev_ptrs)
            if len(relocs) != len(all_values):
                raise RuntimeError(
                    f"Relocation count mismatch: {len(relocs)} relocs vs "
                    f"{len(all_values)} device pointers "
                    f"({len(dev_ptrs)} inputs + {len(out_dev_ptrs)} outputs)"
                )
            code.bind_relocs(relocs, all_values)

            # ---- Step 7: sync H2D, launch, sync launch ----
            sys.sync_stream(stream)
            sys.launch(code, 0, stream)
            sys.sync_stream(stream)

            # ---- Step 8: D2H copy outputs ----
            results = []
            for i, (shape, np_dtype) in enumerate(zip(out_shapes, out_dtypes)):
                out_arr = np.empty(shape, dtype=np_dtype)
                sys.memcpy_d2h(
                    out_arr.ctypes.data, out_dev_ptrs[i],
                    out_arr.nbytes, stream,
                )
                sys.sync_stream(stream)
                results.append(out_arr)

            return results[0] if len(results) == 1 else results

        finally:
            # ---- Cleanup device memory and stream ----
            for ptr in dev_ptrs:
                sys.free_device(ptr)
            for ptr in out_dev_ptrs:
                sys.free_device(ptr)
            sys.destroy_stream(stream)


def kernel(kernel_type="vector", dynamic=False):
    """Decorator factory for DVM kernels.

    Parameters
    ----------
    kernel_type : str
        Target type (default ``"vector"``).
    dynamic : bool
        Reserved for future dynamic-shape support.

    Returns
    -------
    callable
        A decorator that wraps a graph-building function in a ``PyKernel``.

    Example
    -------
    >>> @kernel()
    ... def my_add(k, x, y):
    ...     a = k.load(x.shape, dvm.float32)
    ...     b = k.load(y.shape, dvm.float32)
    ...     return k.store(k.add(a, b))
    """
    def wrapper(func):
        pk = PyKernel(kernel_type=kernel_type)
        pk.build(func)
        return pk
    return wrapper
