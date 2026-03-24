from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = cythonize([], compiler_directives={"language_level": "3"})

setup(
    packages=["candle_dvm"],
    ext_modules=ext_modules,
)
