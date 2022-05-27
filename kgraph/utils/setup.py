### for test


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension(
        "memory",["memory.pyx"]
    ),
    Extension(
        "read",["read.pyx"]
    ),
    Extension("corrupt", ["corrupt.pyx"])
]

setup(
    name = "test",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)