from setuptools import setup, find_packages

setup_params = dict(name='graphai',
                    version='0.0.0',
                    description='A Python library for relational learning on knowledge graphs.',
                    # url='https://github.com/Accenture/AmpliGraph/',
                    # author='Accenture Dublin Labs',
                    # author_email='about@ampligraph.org',
                    license='Apache 2.0',
                    packages=find_packages(),
                    include_package_data=True,
                    zip_safe=True,
                    install_requires=[
                        'numpy>=1.14.3',
                        # 'joblib>=0.11',
                        'tqdm>=4.23.4',
                        'pandas>=0.23.1',
                        'scipy>=1.3.0',
                        'setuptools>=36'
                    ])

if __name__ == '__main__':
    setup(**setup_params)