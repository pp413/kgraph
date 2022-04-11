使用auditwheel工具装换轮子为manylinux。

https://github.com/pypa/auditwheel

安装：

sudo apt-get install patchelf
pip install auditwheel

auditwheel show xxxxxxxxx.whl

auditwheel repair xxxxxxxxx.whl

制作发布包
python setup.py bdist_wheel

twine upload dist/* -u name -p password --verbose

安装：
pip install auditwheel-symbols

USAGE:
    auditwheel-symbols [OPTIONS] <FILE>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -m, --manylinux <manylinux>     [possible values: 1, 2010, 2014, 2_24]

ARGS:
    <FILE>
