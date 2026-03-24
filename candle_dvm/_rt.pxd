# cython: language_level=3
"""Cython declarations for RTS runtime types and dlopen/dlsym.

The RT path uses ``dlopen("libruntime.so")`` + ``dlsym`` to resolve
symbols at runtime. This file declares the POSIX dl functions and the
rtDevBinary_t struct needed for the RT registration path.
"""

from libc.stdint cimport uint32_t, uint64_t, int32_t

cdef extern from "dlfcn.h" nogil:
    int RTLD_LAZY
    int RTLD_LOCAL
    void *dlopen(const char *filename, int flag)
    void *dlsym(void *handle, const char *symbol)
    int dlclose(void *handle)
    char *dlerror()


# ---- RT-specific types (from system.cc) ----

cdef struct rtDevBinary_t:
    uint32_t magic
    uint32_t version
    const void *data
    uint64_t length

# Magic constants for rtDevBinary_t
cdef enum:
    RT_DEV_BINARY_MAGIC_ELF        = 0x43554245
    RT_DEV_BINARY_MAGIC_ELF_AIVEC  = 0x41415246
    RT_DEV_BINARY_MAGIC_ELF_AICUBE = 0x41494343

# RT error code
cdef enum:
    RT_ERROR_NONE = 0

# ---- Function pointer typedefs for dlsym'd RT symbols ----

ctypedef int32_t (*rtDevBinaryRegisterFunc)(const rtDevBinary_t *bin_desc,
                                            void **handle) noexcept nogil
ctypedef int32_t (*rtFunctionRegisterFunc)(void *binHandle,
                                           const void *stubFunc,
                                           const char *stubName,
                                           const void *kernelInfoExt,
                                           uint32_t funcMode) noexcept nogil
ctypedef int32_t (*rtKernelLaunchFunc)(const void *stubFunc,
                                       uint32_t blockDim,
                                       void *args,
                                       uint32_t argsSize,
                                       void *smDesc,
                                       void *stream) noexcept nogil
ctypedef int32_t (*rtGetC2cCtrlAddrFunc)(uint64_t *addr,
                                         uint32_t *length) noexcept nogil
