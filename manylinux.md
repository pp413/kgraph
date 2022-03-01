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


