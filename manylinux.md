使用auditwheel工具装换轮子为manylinux。

https://github.com/pypa/auditwheel

安装：

pip install twine
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


如果使用auditwheel repair XXXXXXX.whl报错，则：

1. auditwheel-symbols XXXXXXX.whl --manylinux 1 (2_17, 2_24) 等等，找出版本号。

2. auditwheel repair XXXXXXX.whl --plat manylinux_2_31_x86_64 根据版本号设置Linux版本。



--plat: 'linux_x86_64', 'manylinux_2_5_x86_64', 'manylinux_2_12_x86_64', 'manylinux_2_17_x86_64', 'manylinux_2_24_x86_64',
'manylinux_2_27_x86_64', 'manylinux_2_28_x86_64', 'manylinux_2_31_x86_64', 'manylinux1_x86_64', 'manylinux2010_x86_64',
'manylinux2014_x86_64'


#manylinux_2_31_x86_64



