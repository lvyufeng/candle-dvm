# cython: language_level=3
"""Cython declaration file for candle_dvm.ops -- NDObject hierarchy."""

from candle_dvm.code cimport Code, RelocAddr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
cdef int _DTYPE_F32
cdef int _BIN_ADD
cdef int _OBJ_LOAD
cdef int _OBJ_STORE
cdef int _OBJ_BINARY


# ---------------------------------------------------------------------------
# NDObject base
# ---------------------------------------------------------------------------
cdef class NDObject:
    cdef public int obj_id
    cdef public int type_id
    cdef public tuple shape_ref
    cdef public NDObject lhs
    cdef public NDObject rhs
    cdef public int index
    cdef public int xbuf
    cdef public bint normalized
    # Sync flags for pipeline synchronization
    cdef public int sync_set
    cdef public int sync_wait
    cdef public int sync_set_event
    cdef public int sync_wait_event
    cdef public int sync_back_set
    cdef public int sync_back_wait
    cdef public int sync_b_set_event
    cdef public int sync_b_wait_event


# ---------------------------------------------------------------------------
# NDAccess
# ---------------------------------------------------------------------------
cdef class NDAccess(NDObject):
    cdef public int io_index


# ---------------------------------------------------------------------------
# NDLoad
# ---------------------------------------------------------------------------
cdef class NDLoad(NDAccess):
    pass


# ---------------------------------------------------------------------------
# NDStore
# ---------------------------------------------------------------------------
cdef class NDStore(NDAccess):
    pass


# ---------------------------------------------------------------------------
# FlexOp
# ---------------------------------------------------------------------------
cdef class FlexOp(NDObject):
    cdef public int workspace_xbuf


# ---------------------------------------------------------------------------
# BinaryOp
# ---------------------------------------------------------------------------
cdef class BinaryOp(FlexOp):
    cdef public int op_type


# ---------------------------------------------------------------------------
# UnaryOp
# ---------------------------------------------------------------------------
cdef class UnaryOp(FlexOp):
    cdef public int op_type


# ---------------------------------------------------------------------------
# BinaryScalarOp
# ---------------------------------------------------------------------------
cdef class BinaryScalarOp(FlexOp):
    cdef public int op_type
    cdef public double scalar


# ---------------------------------------------------------------------------
# CompareOp
# ---------------------------------------------------------------------------
cdef class CompareOp(FlexOp):
    cdef public int cmp_type
