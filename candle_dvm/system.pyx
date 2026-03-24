# cython: language_level=3
"""DVM System runtime loader and device-memory helpers.

Handles:
- SoC detection via ``aclrtGetSocName``
- Runtime loading: tries RT path (``libruntime.so``) first, then ACL path
  (``libascendcl.so`` on CANN 8.5+)
- Binary registration of ``g_vkernel_c220.bin``
- Device memory allocation/free, memcpy, stream management
- Kernel launch with ffts_addr patching
"""

from libc.stdlib cimport malloc, free as c_free
from libc.string cimport memcpy as c_memcpy
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t, uintptr_t

from candle_dvm._acl cimport (
    aclError, aclrtStream, ACL_SUCCESS,
    aclrtSetDevice, aclrtGetSocName,
    aclrtCreateStream, aclrtDestroyStream, aclrtSynchronizeStream,
    aclrtMalloc, aclrtFree,
    aclrtMemcpyAsync,
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST,
    ACL_MEM_MALLOC_HUGE_FIRST,
    # Binary load types (ACL path)
    aclrtBinHandle, aclrtFuncHandle,
    aclrtBinaryLoadOption, aclrtBinaryLoadOptions,
    aclrtBinaryLoadFromData, aclrtBinaryGetFunction,
    aclrtLaunchKernelWithHostArgs, aclrtGetHardwareSyncAddr,
    aclrtLaunchKernelCfg,
    ACL_RT_BINARY_LOAD_OPT_MAGIC, ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD,
    ACL_RT_BINARY_MAGIC_ELF_AICORE,
    ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE,
    ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE,
)

from candle_dvm._rt cimport (
    dlopen, dlsym, dlclose, dlerror,
    RTLD_LAZY, RTLD_LOCAL,
    rtDevBinary_t,
    RT_DEV_BINARY_MAGIC_ELF, RT_DEV_BINARY_MAGIC_ELF_AIVEC,
    RT_DEV_BINARY_MAGIC_ELF_AICUBE, RT_ERROR_NONE,
    rtDevBinaryRegisterFunc, rtFunctionRegisterFunc,
    rtKernelLaunchFunc, rtGetC2cCtrlAddrFunc,
)

from candle_dvm.code cimport Code

# ACL launch function pointer type (used in _launch_acl)
ctypedef int (*ACLLaunchFunc)(void *, uint32_t, void *, void *,
                               void *, size_t, void *, size_t) noexcept nogil

# ACL get-ffts function pointer type (used in _patch_ffts_acl)
ctypedef int (*ACLGetFftsFunc)(void **addr) noexcept nogil

# ---------------------------------------------------------------------------
# Module-level registration state (process-wide, survives multiple System instances)
# ---------------------------------------------------------------------------
cdef bint _binary_registered = False
cdef void *_registered_func_handles[3]
cdef void *_registered_rt_handle = NULL
cdef void *_registered_kernel_launch_func = NULL
cdef void *_registered_get_ffts_addr_func = NULL
cdef void *_registered_renamed_bin = NULL
cdef str _registered_runtime_path = ""

_registered_func_handles[0] = NULL
_registered_func_handles[1] = NULL
_registered_func_handles[2] = NULL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target indices matching dvm::Code::kTargetVec/kTargetCube/kTargetMix
DEF TARGET_VEC  = 0
DEF TARGET_CUBE = 1
DEF TARGET_MIX  = 2

DEF PARAM_TABLE_LIMIT = 4096

# Symbol positions in g_vkernel_c220.bin where 'm' -> 'a' for ACL rename
# (from vm.cc: g_mix_symbols_c220)
cdef uint64_t _MIX_SYMBOLS_C220[6]
_MIX_SYMBOLS_C220[:] = [62464, 62513, 64959, 64971, 64985, 66685]
DEF MIX_SYMBOL_LEN_C220 = 6

# ---------------------------------------------------------------------------
# SoC configuration table
# ---------------------------------------------------------------------------

cdef struct SocConfig:
    const char *name
    const char *arch_name
    uint64_t aicore_num
    uint64_t l2_size

DEF MB = 1024 * 1024

# Phase 1: only 910B (C220) SoCs
cdef SocConfig _SOC_CONFIGS[12]
_SOC_CONFIGS[0]  = SocConfig(b"Ascend910B1",     b"c220", 25, 192 * MB)
_SOC_CONFIGS[1]  = SocConfig(b"Ascend910B2",     b"c220", 24, 192 * MB)
_SOC_CONFIGS[2]  = SocConfig(b"Ascend910B2C",    b"c220", 24, 192 * MB)
_SOC_CONFIGS[3]  = SocConfig(b"Ascend910B3",     b"c220", 20, 192 * MB)
_SOC_CONFIGS[4]  = SocConfig(b"Ascend910B4",     b"c220", 20, 96 * MB)
_SOC_CONFIGS[5]  = SocConfig(b"Ascend910B4-1",   b"c220", 20, 96 * MB)
_SOC_CONFIGS[6]  = SocConfig(b"Ascend910_9391",  b"c220", 25, 192 * MB)
_SOC_CONFIGS[7]  = SocConfig(b"Ascend910_9392",  b"c220", 25, 192 * MB)
_SOC_CONFIGS[8]  = SocConfig(b"Ascend910_9381",  b"c220", 24, 192 * MB)
_SOC_CONFIGS[9]  = SocConfig(b"Ascend910_9382",  b"c220", 24, 192 * MB)
_SOC_CONFIGS[10] = SocConfig(b"Ascend910_9372",  b"c220", 20, 192 * MB)
_SOC_CONFIGS[11] = SocConfig(b"Ascend910_9361",  b"c220", 20, 96 * MB)

DEF NUM_SOC_CONFIGS = 12


# ---------------------------------------------------------------------------
# Target name -> index mapping
# ---------------------------------------------------------------------------

cdef int _target_to_index(str target) except -1:
    if target == "vector":
        return TARGET_VEC
    elif target == "cube":
        return TARGET_CUBE
    elif target == "mix":
        return TARGET_MIX
    else:
        raise ValueError(f"Unknown target: {target}")


# ---------------------------------------------------------------------------
# System class
# ---------------------------------------------------------------------------

cdef class System:
    """DVM runtime system -- handles device init, binary registration, and launch.

    Usage::

        sys = System()
        sys.init(0)          # initialize on device 0
        stream = sys.create_stream()
        # ... allocate memory, launch kernels ...
        sys.destroy_stream(stream)
    """

    def __cinit__(self):
        self._inited = False
        self._arch_name = ""
        self._soc_name = ""
        self._runtime_path = ""
        self._rt_handle = NULL
        self._func_handles[0] = NULL
        self._func_handles[1] = NULL
        self._func_handles[2] = NULL
        self._has_vector_handle = False
        self._has_cube_handle = False
        self._has_mix_handle = False
        self._kernel_launch_func = NULL
        self._get_ffts_addr_func = NULL
        self._renamed_bin = NULL
        self._device_id = -1
        self._vector_core_num = 0
        self._cube_core_num = 0
        self._local_mem_size = 0
        self._l2_size = 0

    def __dealloc__(self):
        # Note: We intentionally do NOT close the rt_handle or free the
        # renamed_bin here. These are process-wide resources shared across
        # all System instances and must remain valid for the lifetime of
        # the process (same as the C++ g_system global).
        pass

    def init(self, int device_id=0):
        """Initialize the system on the given device.

        Parameters
        ----------
        device_id : int
            Device ID to use (default 0).

        Raises
        ------
        RuntimeError
            If the SoC is unrecognized or runtime loading fails.
        """
        if self._inited:
            return
        self._device_id = device_id

        # Set device
        cdef aclError err = aclrtSetDevice(device_id)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtSetDevice({device_id}) failed with error {err}")

        # Detect SoC
        import os
        cdef const char *soc_cstr
        soc_env = os.environ.get("DVM_SOC_NAME")
        if soc_env is not None:
            soc_name_bytes = soc_env.encode("utf-8")
        else:
            soc_cstr = aclrtGetSocName()
            if soc_cstr == NULL:
                raise RuntimeError("aclrtGetSocName() returned NULL")
            soc_name_bytes = <bytes>soc_cstr

        self._soc_name = soc_name_bytes.decode("utf-8")

        # Look up SoC config
        cdef const SocConfig *config = NULL
        cdef int i
        for i in range(NUM_SOC_CONFIGS):
            if soc_name_bytes == _SOC_CONFIGS[i].name:
                config = &_SOC_CONFIGS[i]
                break

        if config == NULL:
            raise RuntimeError(f"Unrecognized SoC: {self._soc_name}")

        self._arch_name = config.arch_name.decode("utf-8")
        self._cube_core_num = config.aicore_num
        self._vector_core_num = config.aicore_num * 2
        self._l2_size = config.l2_size
        self._local_mem_size = 192 * 1024 - 512  # C220 specific

        # Load binary and register with runtime
        self._load_runtime()
        self._inited = True

    cdef void _load_runtime(self) except *:
        """Try RT path first, then ACL path.

        If the binary has already been registered in this process (by another
        System instance), reuse the cached handles instead of re-registering.
        """
        global _binary_registered, _registered_runtime_path

        if _binary_registered:
            # Reuse previously registered handles
            self._runtime_path = _registered_runtime_path
            self._rt_handle = _registered_rt_handle
            self._kernel_launch_func = _registered_kernel_launch_func
            self._get_ffts_addr_func = _registered_get_ffts_addr_func
            self._renamed_bin = NULL  # owned by the first instance
            self._func_handles[0] = _registered_func_handles[0]
            self._func_handles[1] = _registered_func_handles[1]
            self._func_handles[2] = _registered_func_handles[2]
            self._has_vector_handle = self._func_handles[0] != NULL
            self._has_cube_handle = self._func_handles[1] != NULL
            self._has_mix_handle = self._func_handles[2] != NULL
            return

        # Load the binary data
        from candle_dvm.device_bin import load_c220_binary
        bin_data = load_c220_binary()
        cdef const unsigned char *bin_ptr = bin_data
        cdef unsigned int bin_len = <unsigned int>len(bin_data)

        # --- Try RT path ---
        if self._try_rt_path(bin_ptr, bin_len):
            self._runtime_path = "rt"
            self._save_registration_state()
            return

        # --- Try ACL path ---
        if self._try_acl_path(bin_ptr, bin_len, bin_data):
            self._runtime_path = "acl"
            self._save_registration_state()
            return

        raise RuntimeError(
            "Failed to load DVM runtime. Neither libruntime.so (RT path) "
            "nor libascendcl.so (ACL path) provided all required symbols."
        )

    cdef void _save_registration_state(self):
        """Cache registration state at module level for reuse."""
        global _binary_registered, _registered_runtime_path
        global _registered_rt_handle, _registered_kernel_launch_func
        global _registered_get_ffts_addr_func, _registered_renamed_bin

        _binary_registered = True
        _registered_runtime_path = self._runtime_path
        _registered_rt_handle = self._rt_handle
        _registered_kernel_launch_func = self._kernel_launch_func
        _registered_get_ffts_addr_func = self._get_ffts_addr_func
        _registered_renamed_bin = self._renamed_bin
        _registered_func_handles[0] = self._func_handles[0]
        _registered_func_handles[1] = self._func_handles[1]
        _registered_func_handles[2] = self._func_handles[2]

    cdef bint _try_rt_path(self, const unsigned char *bin_ptr,
                           unsigned int bin_len) except -1:
        """Attempt to load via libruntime.so RT path.

        Returns True if successful, False if we should fall through to ACL.
        """
        cdef void *handle = dlopen(b"libruntime.so", RTLD_LAZY | RTLD_LOCAL)
        if handle == NULL:
            return False

        cdef void *sym_launch = dlsym(handle, b"rtKernelLaunch")
        cdef void *sym_ffts = dlsym(handle, b"rtGetC2cCtrlAddr")
        cdef void *sym_reg_bin = dlsym(handle, b"rtDevBinaryRegister")
        cdef void *sym_reg_func = dlsym(handle, b"rtFunctionRegister")

        if not (sym_launch and sym_ffts and sym_reg_bin and sym_reg_func):
            dlclose(handle)
            return False

        self._rt_handle = handle
        self._kernel_launch_func = sym_launch
        self._get_ffts_addr_func = sym_ffts

        # Register binary with RT
        self._register_binary_rt(sym_reg_bin, sym_reg_func, bin_ptr, bin_len)
        return True

    cdef void _register_binary_rt(self, void *reg_binary_func,
                                  void *reg_function_func,
                                  const unsigned char *bin_data,
                                  unsigned int bin_len) except *:
        """Register the kernel binary via the RT path."""
        cdef rtDevBinaryRegisterFunc reg_binary = <rtDevBinaryRegisterFunc>reg_binary_func
        cdef rtFunctionRegisterFunc reg_function = <rtFunctionRegisterFunc>reg_function_func
        cdef int32_t err
        cdef void *module = NULL
        cdef rtDevBinary_t dev_bin

        # Use self address + target index as stub function pointers (same as C++ DVM)
        cdef uintptr_t self_addr = <uintptr_t><void *>self
        self._func_handles[TARGET_VEC] = <void *>(self_addr + TARGET_VEC)
        self._func_handles[TARGET_CUBE] = <void *>(self_addr + TARGET_CUBE)
        self._func_handles[TARGET_MIX] = <void *>(self_addr + TARGET_MIX)

        # Vector
        dev_bin.version = 0
        dev_bin.data = bin_data
        dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC
        dev_bin.length = bin_len
        err = reg_binary(&dev_bin, &module)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register vector binary failed (err={err})")
        err = reg_function(module, self._func_handles[TARGET_VEC],
                           b"dvm_mix_aiv", b"dvm_mix_aiv", 0)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register vector function failed (err={err})")
        self._has_vector_handle = True

        # Cube
        dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF_AICUBE
        err = reg_binary(&dev_bin, &module)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register cube binary failed (err={err})")
        err = reg_function(module, self._func_handles[TARGET_CUBE],
                           b"dvm_mix_aic", b"dvm_mix_aic", 0)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register cube function failed (err={err})")
        self._has_cube_handle = True

        # Mix
        dev_bin.magic = RT_DEV_BINARY_MAGIC_ELF
        err = reg_binary(&dev_bin, &module)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register mix binary failed (err={err})")
        err = reg_function(module, self._func_handles[TARGET_MIX],
                           b"dvm", b"dvm", 0)
        if err != RT_ERROR_NONE:
            raise RuntimeError(f"RT: register mix function failed (err={err})")
        self._has_mix_handle = True

    cdef bint _try_acl_path(self, const unsigned char *bin_ptr,
                            unsigned int bin_len, object bin_data) except -1:
        """Attempt to load via libascendcl.so ACL path (CANN 8.5+).

        Returns True if successful, False otherwise.
        """
        cdef void *handle = dlopen(b"libascendcl.so", RTLD_LAZY | RTLD_LOCAL)
        if handle == NULL:
            return False

        cdef void *sym_load = dlsym(handle, b"aclrtBinaryLoadFromData")
        cdef void *sym_getfunc = dlsym(handle, b"aclrtBinaryGetFunction")
        cdef void *sym_launch = dlsym(handle, b"aclrtLaunchKernelWithHostArgs")
        cdef void *sym_ffts = dlsym(handle, b"aclrtGetHardwareSyncAddr")

        if not (sym_load and sym_getfunc and sym_launch and sym_ffts):
            dlclose(handle)
            return False

        self._rt_handle = handle
        self._kernel_launch_func = sym_launch
        self._get_ffts_addr_func = sym_ffts

        # Register binary with ACL
        self._register_binary_acl(bin_ptr, bin_len, bin_data)
        return True

    cdef void _register_binary_acl(self, const unsigned char *bin_ptr,
                                   unsigned int bin_len,
                                   object bin_data) except *:
        """Register kernel binary via ACL path with symbol rename hack.

        The ACL path renames 'dvm_mix_aiv' -> 'dvm_aix_aiv' (and aic)
        by replacing 'm' with 'a' at known positions in the binary.
        """
        cdef aclError err
        cdef uint32_t pos

        # Create renamed copy: replace 'm' with 'a' at symbol positions
        self._renamed_bin = malloc(bin_len)
        if self._renamed_bin == NULL:
            raise MemoryError("Failed to allocate renamed binary buffer")
        c_memcpy(self._renamed_bin, bin_ptr, bin_len)
        for pos in range(MIX_SYMBOL_LEN_C220):
            (<char *>self._renamed_bin)[_MIX_SYMBOLS_C220[pos]] = ord('a')

        # Set up binary load options
        cdef aclrtBinaryLoadOption opt_data[2]
        opt_data[0].type = ACL_RT_BINARY_LOAD_OPT_MAGIC
        opt_data[1].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD
        opt_data[1].value.isLazyLoad = 1

        cdef aclrtBinaryLoadOptions bin_opt
        bin_opt.numOpt = 2
        bin_opt.options = opt_data

        cdef aclrtBinHandle bin_handle

        # Vector: load renamed binary with vector magic
        opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE
        err = aclrtBinaryLoadFromData(self._renamed_bin, bin_len,
                                      &bin_opt, &bin_handle)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: load vector binary failed (err={err})")
        err = aclrtBinaryGetFunction(bin_handle, b"dvm_aix_aiv",
                                     <aclrtFuncHandle *>&self._func_handles[TARGET_VEC])
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: get vector function failed (err={err})")
        self._has_vector_handle = True

        # Cube: load renamed binary with cube magic
        opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_CUBE_CORE
        err = aclrtBinaryLoadFromData(self._renamed_bin, bin_len,
                                      &bin_opt, &bin_handle)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: load cube binary failed (err={err})")
        err = aclrtBinaryGetFunction(bin_handle, b"dvm_aix_aic",
                                     <aclrtFuncHandle *>&self._func_handles[TARGET_CUBE])
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: get cube function failed (err={err})")
        self._has_cube_handle = True

        # Mix: load ORIGINAL binary with aicore magic
        opt_data[0].value.magic = ACL_RT_BINARY_MAGIC_ELF_AICORE
        err = aclrtBinaryLoadFromData(bin_ptr, bin_len,
                                      &bin_opt, &bin_handle)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: load mix binary failed (err={err})")
        err = aclrtBinaryGetFunction(bin_handle, b"dvm",
                                     <aclrtFuncHandle *>&self._func_handles[TARGET_MIX])
        if err != ACL_SUCCESS:
            raise RuntimeError(f"ACL: get mix function failed (err={err})")
        self._has_mix_handle = True

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def arch_name(self):
        """Return the architecture name (e.g. 'c220')."""
        return self._arch_name

    def soc_name(self):
        """Return the SoC name string."""
        return self._soc_name

    def runtime_path(self):
        """Return 'rt' or 'acl' depending on which path was loaded."""
        return self._runtime_path

    def has_vector_handle(self):
        """Return True if the vector function handle was registered."""
        return self._has_vector_handle

    def has_cube_handle(self):
        """Return True if the cube function handle was registered."""
        return self._has_cube_handle

    def has_mix_handle(self):
        """Return True if the mix function handle was registered."""
        return self._has_mix_handle

    def vector_core_num(self):
        """Return the number of vector cores."""
        return self._vector_core_num

    def cube_core_num(self):
        """Return the number of cube cores."""
        return self._cube_core_num

    # ------------------------------------------------------------------
    # Device memory helpers
    # ------------------------------------------------------------------

    def malloc_device(self, size_t size):
        """Allocate device memory.

        Parameters
        ----------
        size : int
            Number of bytes to allocate.

        Returns
        -------
        int
            Device pointer as an integer.
        """
        cdef void *dev_ptr = NULL
        cdef aclError err = aclrtMalloc(&dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtMalloc({size}) failed with error {err}")
        return <uintptr_t>dev_ptr

    def free_device(self, uintptr_t ptr):
        """Free device memory.

        Parameters
        ----------
        ptr : int
            Device pointer returned by ``malloc_device``.
        """
        cdef aclError err = aclrtFree(<void *>ptr)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtFree failed with error {err}")

    def memcpy_h2d(self, uintptr_t dst, uintptr_t src, size_t size,
                   uintptr_t stream):
        """Async memcpy from host to device.

        Parameters
        ----------
        dst : int
            Device destination pointer.
        src : int
            Host source pointer.
        size : int
            Number of bytes.
        stream : int
            Stream handle.
        """
        cdef aclError err = aclrtMemcpyAsync(
            <void *>dst, size, <void *>src, size,
            ACL_MEMCPY_HOST_TO_DEVICE, <aclrtStream>stream)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtMemcpyAsync H2D failed with error {err}")

    def memcpy_d2h(self, uintptr_t dst, uintptr_t src, size_t size,
                   uintptr_t stream):
        """Async memcpy from device to host.

        Parameters
        ----------
        dst : int
            Host destination pointer.
        src : int
            Device source pointer.
        size : int
            Number of bytes.
        stream : int
            Stream handle.
        """
        cdef aclError err = aclrtMemcpyAsync(
            <void *>dst, size, <void *>src, size,
            ACL_MEMCPY_DEVICE_TO_HOST, <aclrtStream>stream)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtMemcpyAsync D2H failed with error {err}")

    # ------------------------------------------------------------------
    # Stream helpers
    # ------------------------------------------------------------------

    def create_stream(self):
        """Create a new device stream.

        Returns
        -------
        int
            Stream handle as an integer.
        """
        cdef aclrtStream stream = NULL
        cdef aclError err = aclrtCreateStream(&stream)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtCreateStream failed with error {err}")
        return <uintptr_t>stream

    def destroy_stream(self, uintptr_t stream):
        """Destroy a device stream.

        Parameters
        ----------
        stream : int
            Stream handle.
        """
        cdef aclError err = aclrtDestroyStream(<aclrtStream>stream)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtDestroyStream failed with error {err}")

    def sync_stream(self, uintptr_t stream):
        """Synchronize a device stream (block until all tasks complete).

        Parameters
        ----------
        stream : int
            Stream handle.
        """
        cdef aclError err = aclrtSynchronizeStream(<aclrtStream>stream)
        if err != ACL_SUCCESS:
            raise RuntimeError(f"aclrtSynchronizeStream failed with error {err}")

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------

    def launch(self, Code code, uintptr_t extern_ws, uintptr_t stream):
        """Launch a DVM kernel.

        Parameters
        ----------
        code : Code
            The code buffer to launch.
        extern_ws : int
            External workspace device pointer (for large args).
        stream : int
            Stream handle.
        """
        if not self._inited:
            raise RuntimeError("System not initialized; call init() first")

        cdef int target_idx = _target_to_index(code.target)

        if self._runtime_path == "rt":
            self._launch_rt(code, target_idx, extern_ws, stream)
        elif self._runtime_path == "acl":
            self._launch_acl(code, target_idx, extern_ws, stream)
        else:
            raise RuntimeError("No runtime loaded")

    cdef void _launch_rt(self, Code code, int target_idx,
                         uintptr_t extern_ws, uintptr_t stream) except *:
        """Launch via RT path."""
        cdef rtKernelLaunchFunc launch_fn = <rtKernelLaunchFunc>self._kernel_launch_func
        cdef void *func_handle = self._func_handles[target_idx]
        cdef int32_t err
        cdef aclError acl_err
        cdef uint64_t rt_args[2]

        # Patch ffts_addr for mix target on C220
        if self._arch_name == "c220" and target_idx == TARGET_MIX:
            self._patch_ffts_rt(code)

        if code.data_size <= PARAM_TABLE_LIMIT:
            err = launch_fn(func_handle, code.block_dim,
                            code._buf, code.data_size, NULL, <void *>stream)
        else:
            # Copy code buffer to device workspace
            acl_err = aclrtMemcpyAsync(
                <void *>extern_ws, code.data_size,
                code._buf, code.data_size,
                ACL_MEMCPY_HOST_TO_DEVICE, <aclrtStream>stream)
            if acl_err != ACL_SUCCESS:
                raise RuntimeError(f"RT launch: memcpy failed (err={acl_err})")

            # Compact arg block: [device_ptr, entry_word]
            rt_args[0] = <uint64_t>extern_ws
            rt_args[1] = (<uint64_t *>code._buf)[1]
            err = launch_fn(func_handle, code.block_dim,
                            rt_args, sizeof(rt_args), NULL, <void *>stream)

        if err != 0:
            raise RuntimeError(f"rtKernelLaunch failed (err={err})")

    cdef void _launch_acl(self, Code code, int target_idx,
                          uintptr_t extern_ws, uintptr_t stream) except *:
        """Launch via ACL path."""
        cdef void *func_handle = self._func_handles[target_idx]
        cdef int err
        cdef aclError acl_err
        cdef uint64_t acl_args[2]
        cdef ACLLaunchFunc launch_fn = <ACLLaunchFunc>self._kernel_launch_func

        # Patch ffts_addr for mix target on C220
        if self._arch_name == "c220" and target_idx == TARGET_MIX:
            self._patch_ffts_acl(code)

        if code.data_size <= PARAM_TABLE_LIMIT:
            err = launch_fn(func_handle, code.block_dim,
                            <void *>stream, NULL,
                            code._buf, code.data_size, NULL, 0)
        else:
            # Copy code buffer to device workspace
            acl_err = aclrtMemcpyAsync(
                <void *>extern_ws, code.data_size,
                code._buf, code.data_size,
                ACL_MEMCPY_HOST_TO_DEVICE, <aclrtStream>stream)
            if acl_err != ACL_SUCCESS:
                raise RuntimeError(f"ACL launch: memcpy failed (err={acl_err})")

            # Compact arg block
            acl_args[0] = <uint64_t>extern_ws
            acl_args[1] = (<uint64_t *>code._buf)[1]
            err = launch_fn(func_handle, code.block_dim,
                            <void *>stream, NULL,
                            acl_args, sizeof(acl_args), NULL, 0)

        if err != 0:
            raise RuntimeError(f"aclrtLaunchKernelWithHostArgs failed (err={err})")

    cdef void _patch_ffts_rt(self, Code code) except *:
        """Patch code.data[0:8] with ffts addr via RT path."""
        cdef rtGetC2cCtrlAddrFunc get_ffts = <rtGetC2cCtrlAddrFunc>self._get_ffts_addr_func
        cdef uint32_t length = 0
        cdef int32_t err = get_ffts(<uint64_t *>code._buf, &length)
        if err != 0:
            raise RuntimeError(f"rtGetC2cCtrlAddr failed (err={err})")

    cdef void _patch_ffts_acl(self, Code code) except *:
        """Patch code.data[0:8] with ffts addr via ACL path."""
        cdef ACLGetFftsFunc get_ffts = <ACLGetFftsFunc>self._get_ffts_addr_func
        cdef int err = get_ffts(<void **>code._buf)
        if err != 0:
            raise RuntimeError(f"aclrtGetHardwareSyncAddr failed (err={err})")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

cdef System _singleton = None

def get_system():
    """Return the lazily-initialized System singleton.

    The singleton is initialized on device 0 on first call.
    """
    global _singleton
    if _singleton is None:
        _singleton = System()
        _singleton.init(0)
    return _singleton
