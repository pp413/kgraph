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
    "kgraph.utils.corrupt",
    "kgraph.utils.memory",
    "kgraph.utils.read",
    "kgraph.utils.sample",
    "kgraph.utils.evaluation",
    "kgraph.utils.tools",
    "kgraph.utils.classification",
]


COMPILE_OPTIONS = {
    "msvc": ["/Ox", "/EHsc"],
    "mingw32": ["-O3", "-Wno-strict-prototypes", "-Wno-unused-function"],
    "other": ["-O3", "-Wno-strict-prototypes", "-Wno-unused-function"]
}

LINK_OPTIONS = {"mingw32": ["-std=c++11"], "other": []}

COMPILER_DIRECTIVES = {
    "language_level": 3,
}

COPY_FILES = {
    ROOT / "setup.cfg": ROOT / "kgraph" / "package",
    ROOT / "README.md": ROOT / "kgraph" / "package",
    # ROOT / "LICENSE": PACKAGE_ROOT / "kgraph" / "package",
}


def is_new_osx():
    """Check whether we're on OSX >= 10.7"""
    if sys.platform != "darwin":
        return False
    major_version, minor_version, _ = platform.mac_ver(release_level=0)
    return int(major_version) > 10 or (int(major_version) == 10 and int(minor_version) >= 7)


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
        """
        Builds compile and link options for the current compiler type and applies them to each extension.
        
        :return: None
        """
        compile_options = COMPILE_OPTIONS.get(
            self.compiler.compiler_type, COMPILE_OPTIONS["other"]
        )
        link_options = LINK_OPTIONS.get(
            self.compiler.compiler_type, LINK_OPTIONS["other"]
        )
        for e in self.extensions:
            e.extra_compile_args += compile_options
            e.extra_link_args += link_options


class build_ext_subclass(build_ext, build_ext_options):
    def build_extensions(self):
        build_ext_options.build_options(self)
        build_ext.build_extensions(self)


def clean(path):
    """
    Deletes all the files with certain suffixes within the directory and its subdirectories.
    
    :param path: The directory path to clean up.
    :type path: Path object
    
    :return: None
    """
    to_delete = [p for p in path.glob("**/*") if p.is_file() and p.suffix in (".so", ".cpp", ".pyd", ".html")]
    for p in to_delete:
        try:
            p.unlink()
            print(f"Deleted {p.name}")
        except PermissionError:
            print(f"Permission denied for {p.name}")

def setup_package():
    if len(sys.argv) > 1 and sys.argv[1] == "clean":
        return clean(PACKAGE_ROOT)

    copy_files()

    include_dirs = [np.get_include(), get_python_inc(plat_specific=True)]
    ext_modules = [Extension(
        name,
        [name.replace(".", "/") + ".pyx"],
        language='c++',
        include_dirs=include_dirs,
        extra_compile_args=["-std=c++11"]) for name in MOD_NAMES]

    print("Cythonizing sources")
    ext_modules = cythonize(ext_modules, compiler_directives=COMPILER_DIRECTIVES)

    setup(
        name="kgraph",
        packages=PACKAGES,
        version="1.0.9",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_subclass},
        package_data={"": ["*.pyx", "*.pxd", "*.pxi"]})

def copy_files():
    root = os.getcwd()
    package_dir = os.path.join(root, "kgraph", "package")
    os.makedirs(package_dir, exist_ok=True)

    for copy_file, target_dir in COPY_FILES.items():
        if copy_file.exists():
            shutil.copy(str(copy_file), str(target_dir))
            print(f"Copied {copy_file} -> {target_dir}")


if __name__ == "__main__":
    setup_package()
