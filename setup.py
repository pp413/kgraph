# !/usr/bin/env python3
import sys, os
import numpy as np

import platform
import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.extension import Extension
from distutils.sysconfig import customize_compiler, get_python_inc

from pathlib import Path

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    from Cython.Compiler import Options
except:
    print("You don't seem to have Cython installed.")
    sys.exit(1)

ROOT = Path(__file__).parent
PACKAGE_ROOT = ROOT / 'kgraph'

Options.docstrings = True

PACKAGES = find_packages()

MOD_NAMES = [
    "kgraph.utils.cache_data",
    "kgraph.utils.corrupt",
    "kgraph.utils.mem",
    "kgraph.utils.random_int64",
    "kgraph.utils.read",
    "kgraph.utils.sample",
    "kgraph.utils.test",
    "kgraph.utils.tools",
]


COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O3", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O3", "-Wno-strict-prototypes", "-Wno-unused-function"]
}

LINK_OPTIONS = {"msvc": ["-std=c++11"], "mingw32": ["-std=c++11"], "other": []}

COMPILER_DIRECTIVES = {
    "language_level": 3,
}

COPY_FILES = {
    ROOT / "setup.cfg": ROOT / "tests" / "package",
    ROOT / "README.md": ROOT / "tests" / "package",
    # ROOT / "LICENSE": PACKAGE_ROOT / "tests" / "package",
}


def is_new_osx():
    """Check whether we're on OSX >= 10.7"""
    if sys.platform != "darwin":
        return False
    mac_ver = platform.mac_ver()[0]
    if mac_ver.startswith("10"):
        minor_version = int(mac_ver.split(".")[1])
        if minor_version >= 7:
            return True
        else:
            return False
    return False


if is_new_osx():
    # On Mac, use libc++ because Apple deprecated use of
    # libstdc
    COMPILE_OPTIONS["other"].append("-stdlib=libc++")
    LINK_OPTIONS["other"].append("-lc++")
    # g++ (used by unix compiler on mac) links to libstdc++ as a default lib.
    # See: https://stackoverflow.com/questions/1653047/avoid-linking-to-libstdc
    LINK_OPTIONS["other"].append("-nodefaultlibs")

# By subclassing build_extensions we have the actual compiler that will be used which is really known only after finalize_options
# http://stackoverflow.com/questions/724664/python-distutils-how-to-get-a-compiler-that-is-going-to-be-used
class build_ext_options:
    def build_options(self):
        for e in self.extensions:
            e.extra_compile_args += COMPILE_OPTIONS.get(
                self.compiler.compiler_type, COMPILE_OPTIONS["other"]
            )
        for e in self.extensions:
            e.extra_link_args += LINK_OPTIONS.get(
                self.compiler.compiler_type, LINK_OPTIONS["other"]
            )


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def clean(path):
    for path in path.glob("**/*"):
        if path.is_file() and path.suffix in (".so", ".cpp", ".html"):
            print(f"Deleting {path.name}")
            path.unlink()

def setup_package():
    # write_git_info_py()
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(PACKAGE_ROOT)

    # with (PACKAGE_ROOT / "about.py").open("r") as f:
    #     about = {}
    #     exec(f.read(), about)

    for copy_file, target_dir in COPY_FILES.items():
        if copy_file.exists():
            shutil.copy(str(copy_file), str(target_dir))
            print(f"Copied {copy_file} -> {target_dir}")

    include_dirs = [
        np.get_include(),
        get_python_inc(plat_specific=True),
    ]
    ext_modules = []
    for name in MOD_NAMES:
        mod_path = name.replace(".", "/") + ".pyx"
        ext = Extension(
            name, [mod_path], language='c++', include_dirs=include_dirs, extra_compile_args=["-std=c++11"]
        )
        ext_modules.append(ext)
    print("Cythonizing sources")
    ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)
    # ext_modules = cythonize(ext_modules, language_level=3)

    setup(
        name="kgraph",
        packages=PACKAGES,
        version="1.0.6",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_subclass},
        package_data={"": ["*.pyx", "*.pxd", "*.pxi"]},
    )
    
    clean(PACKAGE_ROOT)


if __name__ == "__main__":
    setup_package()
