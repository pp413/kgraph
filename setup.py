from setuptools import setup, find_packages

setup_params = dict(name='kgraph',
                    version='0.2.0',
                    description='A Python library for relational learning on knowledge graphs.',
                    url='https://github.com/YaoShuang-long/kgraph',
                    author='Yao Shuang-long',
                    author_email='shuanglongyao@gmail.com',
                    license='Apache 2.0',
                    platforms=['linux'],
                    packages=find_packages(exclude=['kgraph', 'kgraph/release']),
                    # packages=['kgraph', 'kgraph.release', 'kgraph.datasets', 'kgraph.module',
                    #           'kgraph.module.loss', 'kgraph.module.model', 'kgraph.module.strategy',
                    #           'kgraph.utils'],
                    include_package_data=True,
                    # zip_safe=True,
                    zip_safe=False,
                    install_requires=[
                        'numpy>=1.14.3',
                        # 'joblib>=0.11',
                        'tqdm>=4.23.4',
                        'pandas>=0.23.1',
                        'scipy>=1.3.0',
                        'setuptools>=36',
                        'prettytable>=0.7.2',
                        'arrow>=0.15.6'],
                    package_dir={'release': 'kgraph/release'},
                    package_data={'release': ['Base.so']},
                    # ext_modules='release/Base.so'
                    # data_files=[('release', ['kgraph/release/Base.so'])],
                    )

if __name__ == '__main__':
    setup(**setup_params)