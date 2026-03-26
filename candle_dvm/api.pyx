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
from candle_dvm.ops import (
    NDObject, NDLoad, NDStore, BinaryOp, BIN_ADD, BIN_SUB, BIN_MUL, BIN_DIV,
    BIN_MAX, BIN_MIN,
    BinaryScalarOp, BINS_ADD, BINS_MUL, BINS_DIV, BINS_MAX, BINS_MIN,
    UnaryOp, UNARY_SQRT, UNARY_ABS, UNARY_LOG, UNARY_EXP,
    UNARY_ROUND, UNARY_FLOOR, UNARY_CEIL, UNARY_TRUNC, UNARY_ISFINITE,
)

# Mapping from tensor-tensor BinaryType to scalar BinarySOpType
_BIN_TO_BINS = {
    BIN_ADD: BINS_ADD,
    BIN_MUL: BINS_MUL,
    BIN_DIV: BINS_DIV,
    BIN_MAX: BINS_MAX,
    BIN_MIN: BINS_MIN,
}

# Commutative ops allow scalar on the left (swap to tensor-right-scalar)
_COMMUTATIVE_OPS = frozenset({BIN_ADD, BIN_MUL, BIN_MAX, BIN_MIN})


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

    cdef object _binary_dispatch(self, int bin_op, object a, object b):
        """Route a binary op to BinaryOp or BinaryScalarOp as appropriate.

        Parameters
        ----------
        bin_op : int
            BinaryType enum (e.g. BIN_ADD).
        a, b : NDObject or scalar (int/float)
            Operands.

        Returns
        -------
        NDObject
            The created binary node, already appended to the kernel.

        Raises
        ------
        TypeError
            If both operands are scalars.
        NotImplementedError
            If a scalar appears on the left for a non-commutative op.
        """
        cdef bint a_is_tensor = isinstance(a, NDObject)
        cdef bint b_is_tensor = isinstance(b, NDObject)

        if not a_is_tensor and not b_is_tensor:
            raise TypeError(
                "Both operands are scalars; at least one must be a tensor"
            )

        cdef object node
        if a_is_tensor and b_is_tensor:
            node = BinaryOp(bin_op, a, b)
        elif a_is_tensor and not b_is_tensor:
            # tensor op scalar  ->  BinaryScalarOp(BINS_*, tensor, scalar)
            node = BinaryScalarOp(_BIN_TO_BINS[bin_op], a, float(b))
        else:
            # scalar op tensor  (a is scalar, b is tensor)
            if bin_op not in _COMMUTATIVE_OPS:
                raise NotImplementedError(
                    f"scalar-left not supported for non-commutative op {bin_op}"
                )
            # Commutative: swap so tensor is first
            node = BinaryScalarOp(_BIN_TO_BINS[bin_op], b, float(a))

        self._kernel.append(node)
        return node

    cpdef object add(self, object a, object b):
        """Create an element-wise add node.

        Accepts tensor-tensor, tensor-scalar, or scalar-tensor (commutative).

        Parameters
        ----------
        a : NDObject or scalar
            Left-hand operand.
        b : NDObject or scalar
            Right-hand operand.

        Returns
        -------
        BinaryOp or BinaryScalarOp
            The created add node, already appended to the kernel.
        """
        return self._binary_dispatch(BIN_ADD, a, b)

    cpdef object sub(self, object a, object b):
        """Create an element-wise subtract node.

        Parameters
        ----------
        a : NDObject
            Left-hand operand.
        b : NDObject
            Right-hand operand.

        Returns
        -------
        BinaryOp
            The created sub node, already appended to the kernel.
        """
        cdef object node = BinaryOp(BIN_SUB, a, b)
        self._kernel.append(node)
        return node

    cpdef object mul(self, object a, object b):
        """Create an element-wise multiply node.

        Accepts tensor-tensor, tensor-scalar, or scalar-tensor (commutative).

        Parameters
        ----------
        a : NDObject or scalar
            Left-hand operand.
        b : NDObject or scalar
            Right-hand operand.

        Returns
        -------
        BinaryOp or BinaryScalarOp
            The created mul node, already appended to the kernel.
        """
        return self._binary_dispatch(BIN_MUL, a, b)

    cpdef object div(self, object a, object b):
        """Create an element-wise divide node.

        Accepts tensor-tensor or tensor-scalar.  scalar-left raises
        ``NotImplementedError`` because division is non-commutative.

        Parameters
        ----------
        a : NDObject or scalar
            Left-hand operand.
        b : NDObject or scalar
            Right-hand operand.

        Returns
        -------
        BinaryOp or BinaryScalarOp
            The created div node, already appended to the kernel.
        """
        return self._binary_dispatch(BIN_DIV, a, b)

    cpdef object maximum(self, object a, object b):
        """Create an element-wise maximum node.

        Accepts tensor-tensor, tensor-scalar, or scalar-tensor (commutative).

        Parameters
        ----------
        a : NDObject or scalar
            Left-hand operand.
        b : NDObject or scalar
            Right-hand operand.

        Returns
        -------
        BinaryOp or BinaryScalarOp
            The created max node, already appended to the kernel.
        """
        return self._binary_dispatch(BIN_MAX, a, b)

    cpdef object minimum(self, object a, object b):
        """Create an element-wise minimum node.

        Accepts tensor-tensor, tensor-scalar, or scalar-tensor (commutative).

        Parameters
        ----------
        a : NDObject or scalar
            Left-hand operand.
        b : NDObject or scalar
            Right-hand operand.

        Returns
        -------
        BinaryOp or BinaryScalarOp
            The created min node, already appended to the kernel.
        """
        return self._binary_dispatch(BIN_MIN, a, b)

    # ---------------------------------------------------------------
    # Unary methods (Batch A)
    # ---------------------------------------------------------------

    cpdef object sqrt(self, object x):
        """Element-wise square root."""
        cdef object node = UnaryOp(UNARY_SQRT, x)
        self._kernel.append(node)
        return node

    cpdef object abs(self, object x):
        """Element-wise absolute value."""
        cdef object node = UnaryOp(UNARY_ABS, x)
        self._kernel.append(node)
        return node

    cpdef object log(self, object x):
        """Element-wise natural logarithm."""
        cdef object node = UnaryOp(UNARY_LOG, x)
        self._kernel.append(node)
        return node

    cpdef object exp(self, object x):
        """Element-wise exponential."""
        cdef object node = UnaryOp(UNARY_EXP, x)
        self._kernel.append(node)
        return node

    cpdef object round(self, object x):
        """Element-wise round to nearest integer."""
        cdef object node = UnaryOp(UNARY_ROUND, x)
        self._kernel.append(node)
        return node

    cpdef object floor(self, object x):
        """Element-wise floor."""
        cdef object node = UnaryOp(UNARY_FLOOR, x)
        self._kernel.append(node)
        return node

    cpdef object ceil(self, object x):
        """Element-wise ceiling."""
        cdef object node = UnaryOp(UNARY_CEIL, x)
        self._kernel.append(node)
        return node

    cpdef object trunc(self, object x):
        """Element-wise truncation toward zero."""
        cdef object node = UnaryOp(UNARY_TRUNC, x)
        self._kernel.append(node)
        return node

    cpdef object isfinite(self, object x):
        """Element-wise finiteness test (returns bool dtype)."""
        cdef object node = UnaryOp(UNARY_ISFINITE, x)
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
