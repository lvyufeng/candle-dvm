# cython: language_level=3
"""Cython declaration file for candle_dvm.code."""

from libc.stdint cimport uint64_t


# ---------------------------------------------------------------------------
# RelocAddr
# ---------------------------------------------------------------------------
cdef class RelocAddr:
    cdef readonly Py_ssize_t offset


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------
cdef class Code:
    cdef unsigned char *_buf
    cdef Py_ssize_t _capacity
    cdef Py_ssize_t _size

    cdef public str target
    cdef public int block_dim
    cdef public int data_size

    cpdef void append_u64(self, unsigned long long value)
    cpdef unsigned long long read_u64_at(self, Py_ssize_t offset)
    cpdef void free(self)
