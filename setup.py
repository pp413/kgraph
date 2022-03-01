import sys, os
import numpy as np

from setuptools import find_packages, setup
from setuptools.extension import Extension
from distutils.sysconfig import customize_compiler

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed.")
    sys.exit(1)

package_name = "kgraph"

class my_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler(self.compiler)
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        build_ext.build_extensions(self)

def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith('.pyx'):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    print(extPath)
    return Extension(
        name=extName,
        sources=[extPath],
        include_dirs=[np.get_include(), "."],
        extra_compile_args=["-O3", "-Wall"],
        extra_link_args=['-g'],
    )

extNames = scandir(package_name)
print(extNames)

extensions = [makeExtension(name) for name in extNames]

setup(
    name=package_name,
    version='1.0.2',
    description='A Python library for relational learning on knowledge graphs.',
    url='https://github.com/YaoShuang-long/kgraph',
    author='Yao Shuang-long',
    author_email='shuanglongyao@gmail.com',
    license='Apache 2.0',
    platforms=['linux_x86_64'],
    packages=find_packages(),
    package_data = {package_name: [package_name+"*.pxd"]},
    include_package_data=True,
    zip_safe=False,
    ext_modules=cythonize(extensions, language_level=3),
    cmdclass={'build_ext': build_ext},
    install_requires=[
                    'numpy>=1.14.3',
                    'torch>=1.6.0',
                    'tqdm>=4.23.4',
                    'prettytable>=0.7.2',],
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries'
    ],
)

for root, dirs, files in os.walk(os.path.join(package_name, '_utils')):
    for file in files:
        if file.endswith(".cpp"):
            os.remove(os.path.join(root, file))
