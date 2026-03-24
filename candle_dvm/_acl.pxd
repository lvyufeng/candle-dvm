# cython: language_level=3
"""Cython declarations for ACL (Ascend Computing Language) runtime APIs.

These are the direct C-level declarations from ``acl/acl_rt.h`` and
``acl/acl_base_rt.h`` used by the System layer for device management,
memory operations, and stream control.
"""

cdef extern from "acl/acl_base_rt.h" nogil:
    ctypedef void *aclrtStream
    ctypedef void *aclrtEvent
    ctypedef int aclError

    int ACL_SUCCESS "ACL_SUCCESS"

    aclError aclrtSetDevice(int deviceId)
    const char *aclrtGetSocName()


cdef extern from "acl/acl_rt.h" nogil:
    # --- Memory copy kinds ---
    ctypedef enum aclrtMemcpyKind:
        ACL_MEMCPY_HOST_TO_HOST
        ACL_MEMCPY_HOST_TO_DEVICE
        ACL_MEMCPY_DEVICE_TO_HOST
        ACL_MEMCPY_DEVICE_TO_DEVICE

    # --- Memory allocation policy ---
    ctypedef enum aclrtMemMallocPolicy:
        ACL_MEM_MALLOC_HUGE_FIRST
        ACL_MEM_MALLOC_HUGE_ONLY
        ACL_MEM_MALLOC_NORMAL_ONLY

    # --- Binary loading types (CANN 8.5+) ---
    unsigned int ACL_RT_BINARY_MAGIC_ELF_AICORE "ACL_RT_BINARY_MAGIC_ELF_AICORE"
    unsigned int ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE "ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE"
    unsigned int ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE "ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE"

    ctypedef enum aclrtBinaryLoadOptionType:
        ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD
        ACL_RT_BINARY_LOAD_OPT_MAGIC

    ctypedef union aclrtBinaryLoadOptionValue:
        unsigned int isLazyLoad
        unsigned int magic
        int cpuKernelMode
        unsigned int rsv[4]

    ctypedef struct aclrtBinaryLoadOption:
        aclrtBinaryLoadOptionType type
        aclrtBinaryLoadOptionValue value

    ctypedef struct aclrtBinaryLoadOptions:
        aclrtBinaryLoadOption *options
        size_t numOpt

    ctypedef void *aclrtBinHandle
    ctypedef void *aclrtFuncHandle

    ctypedef struct aclrtLaunchKernelCfg:
        pass
    ctypedef struct aclrtPlaceHolderInfo:
        pass

    # --- Device / stream / memory APIs ---
    aclError aclrtCreateStream(aclrtStream *stream)
    aclError aclrtDestroyStream(aclrtStream stream)
    aclError aclrtSynchronizeStream(aclrtStream stream)

    aclError aclrtMalloc(void **devPtr, size_t size,
                         aclrtMemMallocPolicy policy)
    aclError aclrtFree(void *devPtr)

    aclError aclrtMemcpy(void *dst, size_t destMax,
                         const void *src, size_t count,
                         aclrtMemcpyKind kind)
    aclError aclrtMemcpyAsync(void *dst, size_t destMax,
                              const void *src, size_t count,
                              aclrtMemcpyKind kind, aclrtStream stream)

    # --- Binary load / function resolution (CANN 8.5+) ---
    aclError aclrtBinaryLoadFromData(const void *data, size_t length,
                                     const aclrtBinaryLoadOptions *options,
                                     aclrtBinHandle *binHandle)
    aclError aclrtBinaryGetFunction(aclrtBinHandle binHandle,
                                    const char *kernelName,
                                    aclrtFuncHandle *funcHandle)

    # --- Kernel launch ---
    aclError aclrtLaunchKernelWithHostArgs(aclrtFuncHandle funcHandle,
                                           unsigned int blockDim,
                                           aclrtStream stream,
                                           aclrtLaunchKernelCfg *cfg,
                                           void *hostArgs, size_t argsSize,
                                           aclrtPlaceHolderInfo *placeHolderArray,
                                           size_t placeHolderNum)

    # --- Hardware sync addr ---
    aclError aclrtGetHardwareSyncAddr(void **addr)
