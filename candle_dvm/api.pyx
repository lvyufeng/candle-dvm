# cython: language_level=3
"""Public Kernel graph-builder API for candle-dvm.

Provides a user-friendly ``Kernel`` class that wraps the internal
``VKernelS`` static-shape vector kernel and exposes graph construction
methods (``load``, ``store``, ``add``), compilation (``codegen``), and
introspection (``debug_header``, ``get_relocs``).

Example
-------
>>> from candle_dvm import Kernel, float32
>>> k = Kernel()
>>> a = k.load((32, 32), float32)
>>> b = k.load((32, 32), float32)
>>> k.store(k.add(a, b))
>>> k.codegen()
>>> k.debug_header()
{'target': 0, 'block_dim': 1, 'data_size': ..., 'tile_num': 32}
"""

from candle_dvm.kernel import VKernelS
from candle_dvm.ops import NDLoad, NDStore, BinaryOp, BIN_ADD


cdef class Kernel:
    """High-level graph builder for DVM compute kernels.

    Attributes
    ----------
    _kernel : VKernelS
        The underlying static-shape vector kernel.
    _inputs : list
        List of NDLoad nodes appended so far.
    _outputs : list
        List of NDStore nodes appended so far.
    _input_count : int
        Number of inputs (used for io_index assignment).
    _output_count : int
        Number of outputs (used for io_index assignment).
    """

    cdef object _kernel
    cdef list _inputs
    cdef list _outputs
    cdef int _input_count
    cdef int _output_count

    def __init__(self):
        self._kernel = VKernelS()
        self._inputs = []
        self._outputs = []
        self._input_count = 0
        self._output_count = 0

    cpdef object load(self, tuple shape, int dtype):
        """Create an input load node.

        Parameters
        ----------
        shape : tuple of int
            Shape of the input tensor.
        dtype : int
            Data type (e.g. ``float32`` / ``DTYPE_F32``).

        Returns
        -------
        NDLoad
            The created load node, already appended to the kernel.
        """
        cdef object node = NDLoad(self._input_count, shape, dtype)
        self._kernel.append(node)
        self._inputs.append(node)
        self._input_count += 1
        return node

    cpdef object store(self, object src):
        """Create an output store node.

        Parameters
        ----------
        src : NDObject
            The source node whose result is to be stored.

        Returns
        -------
        NDStore
            The created store node, already appended to the kernel.
        """
        cdef object node = NDStore(self._output_count, src)
        self._kernel.append(node)
        self._outputs.append(node)
        self._output_count += 1
        return node

    cpdef object add(self, object a, object b):
        """Create an element-wise add node.

        Parameters
        ----------
        a : NDObject
            Left-hand operand.
        b : NDObject
            Right-hand operand.

        Returns
        -------
        BinaryOp
            The created add node, already appended to the kernel.
        """
        cdef object node = BinaryOp(BIN_ADD, a, b)
        self._kernel.append(node)
        return node

    cpdef void codegen(self):
        """Compile the graph into DVM bytecode.

        Calls ``normalize()`` then ``codegen()`` on the underlying kernel.
        """
        self._kernel.normalize()
        self._kernel.codegen()

    cpdef dict debug_header(self):
        """Return a dict summarising the compiled kernel header.

        Returns
        -------
        dict
            Keys: ``target``, ``block_dim``, ``data_size``, ``tile_num``.
        """
        return self._kernel.debug_header()

    cpdef list get_relocs(self):
        """Return the list of relocation entries after codegen.

        Returns
        -------
        list[RelocAddr]
            One relocation per IO operand (loads + stores).
        """
        return self._kernel.relocs

    @property
    def code(self):
        """Return the compiled Code object from the underlying kernel."""
        return self._kernel.code

    @property
    def inputs(self):
        """Return the list of input NDLoad nodes."""
        return list(self._inputs)

    @property
    def outputs(self):
        """Return the list of output NDStore nodes."""
        return list(self._outputs)
