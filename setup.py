import os
from setuptools import setup, Extension
from Cython.Build import cythonize

# --- Locate CANN toolkit directories ---
_CANN_CANDIDATES = [
    "/usr/local/Ascend/cann-8.5.0/aarch64-linux",
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux",
    "/usr/local/Ascend/ascend-toolkit/latest",
]

cann_include_dir = None
cann_lib_dir = None
for base in _CANN_CANDIDATES:
    inc = os.path.join(base, "include")
    lib = os.path.join(base, "lib64")
    if os.path.isfile(os.path.join(inc, "acl", "acl_rt.h")):
        cann_include_dir = inc
        if os.path.isdir(lib):
            cann_lib_dir = lib
        break

# Build the system extension with include dirs and libraries
system_ext = Extension(
    "candle_dvm.system",
    sources=["candle_dvm/system.pyx"],
    include_dirs=[cann_include_dir] if cann_include_dir else [],
    library_dirs=[cann_lib_dir] if cann_lib_dir else [],
    runtime_library_dirs=[cann_lib_dir] if cann_lib_dir else [],
    libraries=["dl", "ascendcl"],
)

setup(
    name="candle_dvm",
    packages=["candle_dvm", "candle_dvm.data"],
    package_data={
        "candle_dvm.data": ["g_vkernel_c220.bin", "README.md"],
    },
    ext_modules=cythonize([
        "candle_dvm/device_bin.pyx",
        "candle_dvm/isa.pyx",
        "candle_dvm/code.pyx",
        "candle_dvm/ops.pyx",
        "candle_dvm/pass_.pyx",
        system_ext,
    ]),
    extras_require={"test": ["pytest"]},
)
