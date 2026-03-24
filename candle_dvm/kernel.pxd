# cython: language_level=3
"""Cython declaration file for candle_dvm.kernel -- VKernel hierarchy."""

from candle_dvm.code cimport Code
from candle_dvm.ops cimport NDObject


# ---------------------------------------------------------------------------
# VKernel (base)
# ---------------------------------------------------------------------------
cdef class VKernel:
    cdef public Code code
    cdef public list relocs
    cdef public int ktype
    cdef public int flags


# ---------------------------------------------------------------------------
# VectorKernel
# ---------------------------------------------------------------------------
cdef class VectorKernel(VKernel):
    cdef public list objects
    cdef public long long tile_num
    cdef public long long tile_size
    cdef public int target


# ---------------------------------------------------------------------------
# VKernelS (static shape vector kernel)
# ---------------------------------------------------------------------------
cdef class VKernelS(VectorKernel):
    cdef public list build_ops


# ---------------------------------------------------------------------------
# VKernelD (placeholder)
# ---------------------------------------------------------------------------
cdef class VKernelD(VKernelS):
    pass
