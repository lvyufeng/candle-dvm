# cython: language_level=3
"""DVM code buffer with relocation support.

Provides a raw memory buffer for DVM bytecode, relocation address tracking,
and the ``gen_entry_v`` class method that computes a vector entry word using
the exact DVM formula from ``Code::GenEntryV``.

Phase-1 scope: vector target only; no workspace binding or launch logic.
"""

from libc.stdlib cimport malloc, free as c_free
from libc.string cimport memset
from libc.stdint cimport uint64_t

# Re-export the constants we need from isa (already compiled)
from candle_dvm.isa import (
    V_ENTRY_TYPE_V,
    V_ENTRY_CODE_SIZE_OFFSET,
    V_ENTRY_V_TILE_TAIL_OFFSET,
    V_ENTRY_V_TILE_BODY_OFFSET,
)


# ===================================================================
# RelocAddr -- a relocation pointing at a uint64_t slot
# ===================================================================

cdef class RelocAddr:
    """Describes a relocation: an offset within the Code buffer that
    should be overwritten with a device address at bind time.

    The entire 64-bit word at *offset* is replaced -- no masking needed.
    """

    # cdef readonly Py_ssize_t offset  -- declared in code.pxd

    def __init__(self, Py_ssize_t offset):
        self.offset = offset

    def __repr__(self):
        return f"RelocAddr(offset={self.offset})"


# ===================================================================
# Code -- raw code buffer
# ===================================================================

cdef class Code:
    """DVM code buffer backed by malloc'd memory.

    Layout (phase 1):
    - word 0: ``ffts_addr``  (patched at launch time by System layer)
    - word 1: ``entry``      (program entry descriptor)
    - word 2+: encoded bytecode
    - terminated by a zero word

    Attributes
    ----------
    target : str
        Target type (``"vector"``, ``"cube"``, etc.). Phase 1 only uses ``"vector"``.
    block_dim : int
        Block dimension for the kernel.
    data_size : int
        Data size in bytes.
    """

    # All cdef attributes declared in code.pxd:
    #   _buf, _capacity, _size, target, block_dim, data_size

    def __cinit__(self, Py_ssize_t capacity=4096):
        self._buf = <unsigned char *>malloc(capacity)
        if self._buf == NULL:
            raise MemoryError("Failed to allocate code buffer")
        memset(self._buf, 0, capacity)
        self._capacity = capacity
        self._size = 0
        self.target = "vector"
        self.block_dim = 0
        self.data_size = 0

    def __dealloc__(self):
        if self._buf != NULL:
            c_free(self._buf)
            self._buf = NULL

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self):
        """Total allocated size in bytes."""
        return self._capacity

    @property
    def size(self):
        """Current number of bytes written."""
        return self._size

    # ------------------------------------------------------------------
    # Buffer operations
    # ------------------------------------------------------------------

    cpdef void append_u64(self, unsigned long long value):
        """Append a 64-bit word to the buffer.

        Raises OverflowError if the buffer is full.
        """
        if self._size + 8 > self._capacity:
            raise OverflowError(
                f"Code buffer overflow: need {self._size + 8} bytes, "
                f"capacity is {self._capacity}"
            )
        (<uint64_t *>(self._buf + self._size))[0] = <uint64_t>value
        self._size += 8

    cpdef unsigned long long read_u64_at(self, Py_ssize_t offset):
        """Read a 64-bit word at the given byte offset.

        Raises IndexError if the buffer is empty or the offset is out of range.
        Raises ValueError if the offset is not 8-byte aligned.
        """
        if self._size == 0:
            raise IndexError("Code buffer is empty")
        if offset % 8 != 0:
            raise ValueError(
                f"Offset {offset} is not 8-byte aligned"
            )
        if offset < 0 or offset + 8 > self._size:
            raise IndexError(
                f"Offset {offset} out of range [0, {self._size - 8}]"
            )
        return (<uint64_t *>(self._buf + offset))[0]

    def bind_relocs(self, list relocs, list values):
        """Overwrite the 64-bit slot at each reloc offset with the
        corresponding value.

        Parameters
        ----------
        relocs : list[RelocAddr]
            Relocation descriptors.
        values : list[int]
            Device addresses / values to write.
        """
        cdef Py_ssize_t i
        cdef RelocAddr r
        cdef unsigned long long v
        if len(relocs) != len(values):
            raise ValueError("relocs and values must have the same length")
        for i in range(len(relocs)):
            r = <RelocAddr>relocs[i]
            v = <unsigned long long>values[i]
            if r.offset % 8 != 0:
                raise ValueError(
                    f"Reloc offset {r.offset} is not 8-byte aligned"
                )
            if r.offset < 0 or r.offset + 8 > self._size:
                raise IndexError(
                    f"Reloc offset {r.offset} out of range [0, {self._size - 8}]"
                )
            (<uint64_t *>(self._buf + r.offset))[0] = <uint64_t>v

    cpdef void free(self):
        """Explicitly free the underlying buffer.

        Safe to call multiple times.
        """
        if self._buf != NULL:
            c_free(self._buf)
            self._buf = NULL
            self._capacity = 0
            self._size = 0

    # ------------------------------------------------------------------
    # Entry word generation
    # ------------------------------------------------------------------

    @staticmethod
    def gen_entry_v(int tile_num, int block_dim, int data_size):
        """Generate a vector program entry word using the DVM formula.

        Parameters
        ----------
        tile_num : int
            Total number of tiles.
        block_dim : int
            Block dimension.
        data_size : int
            Total data size in bytes (must be a multiple of 8).

        Returns
        -------
        int
            64-bit entry word.
        """
        cdef int block_tile = (tile_num + block_dim - 1) // block_dim
        cdef int block_tail = block_dim * block_tile - tile_num
        return (
            V_ENTRY_TYPE_V
            | ((data_size // 8) << V_ENTRY_CODE_SIZE_OFFSET)
            | (block_tail << V_ENTRY_V_TILE_TAIL_OFFSET)
            | (block_tile << V_ENTRY_V_TILE_BODY_OFFSET)
        )

    # ------------------------------------------------------------------
    # Debug / introspection
    # ------------------------------------------------------------------

    def debug_header(self):
        """Return a dict summarising the current header fields.

        Returns
        -------
        dict
            Keys: ``target``, ``block_dim``, ``data_size``.
        """
        return {
            "target": self.target,
            "block_dim": self.block_dim,
            "data_size": self.data_size,
        }
