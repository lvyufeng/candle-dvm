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
    pass


# ---------------------------------------------------------------------------
# BinaryOp
# ---------------------------------------------------------------------------
cdef class BinaryOp(FlexOp):
    cdef public int op_type
