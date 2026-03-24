from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="candle_dvm",
    packages=["candle_dvm", "candle_dvm.data"],
    package_data={
        "candle_dvm.data": ["g_vkernel_c220.bin", "README.md"],
    },
    ext_modules=cythonize([
        "candle_dvm/device_bin.pyx",
    ]),
    extras_require={"test": ["pytest"]},
)
