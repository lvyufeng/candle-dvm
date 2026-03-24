# cython: language_level=3
"""Cython declaration file for candle_dvm.system."""

from libc.stdint cimport uint64_t, uintptr_t
from candle_dvm.code cimport Code


cdef class System:
    cdef bint _inited
    cdef str _arch_name
    cdef str _soc_name

    # Runtime path: "rt" or "acl"
    cdef str _runtime_path

    # dlopen handle for libruntime.so or libascendcl.so
    cdef void *_rt_handle

    # Function handles (3 slots: vector=0, cube=1, mix=2)
    # For RT path: these are stub pointers (self_addr + target index)
    # For ACL path: these are aclrtFuncHandle values from aclrtBinaryGetFunction
    cdef void *_func_handles[3]
    cdef bint _has_vector_handle
    cdef bint _has_cube_handle
    cdef bint _has_mix_handle

    # Function pointers resolved via dlsym
    cdef void *_kernel_launch_func
    cdef void *_get_ffts_addr_func

    # Renamed binary buffer (ACL path only)
    cdef void *_renamed_bin

    # Hardware info
    cdef int _device_id
    cdef uint64_t _vector_core_num
    cdef uint64_t _cube_core_num
    cdef uint64_t _local_mem_size
    cdef uint64_t _l2_size

    # --- Private cdef methods ---
    cdef void _load_runtime(self) except *
    cdef void _save_registration_state(self)
    cdef bint _try_rt_path(self, const unsigned char *bin_ptr,
                           unsigned int bin_len) except -1
    cdef void _register_binary_rt(self, void *reg_binary_func,
                                  void *reg_function_func,
                                  const unsigned char *bin_data,
                                  unsigned int bin_len) except *
    cdef bint _try_acl_path(self, const unsigned char *bin_ptr,
                            unsigned int bin_len, object bin_data) except -1
    cdef void _register_binary_acl(self, const unsigned char *bin_ptr,
                                   unsigned int bin_len,
                                   object bin_data) except *
    cdef void _launch_rt(self, Code code, int target_idx,
                         uintptr_t extern_ws, uintptr_t stream) except *
    cdef void _launch_acl(self, Code code, int target_idx,
                          uintptr_t extern_ws, uintptr_t stream) except *
    cdef void _patch_ffts_rt(self, Code code) except *
    cdef void _patch_ffts_acl(self, Code code) except *
