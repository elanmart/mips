# See https://github.com/pybind/python_example
# and https://github.com/facebookresearch/fastText/blob/master/setup.py

import shutil
import warnings

from setuptools import setup


def _readme():
    with open("README.md") as f:
        return f.read()

try:
    shutil.copyfile("bin/_pymips.so", "python/pymips/index/_pymips.so")
except:
    warnings.warn('It seems that you did not compile our c++ source. Falling-back to python-only install')

setup(
    name='pymips',
    version='0.1.0',
    author='Marcin Elantkowski',
    author_email='marcin.elantkowski@gmail.com',
    description='MIPS library wrapper',
    long_description=_readme(),
    url='https://github.com/walkowiak/mips',
    license='BSD',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Operating System :: Unix',
    ],
    packages=[
        'pymips',
        'pymips.utils',
        'pymips.index',
        'pymips.plugins'
    ],
    package_dir={
        '': 'python'
    },
    package_data={
        'pymips': ['index/_pymips.so'],
    },
    zip_safe=False,
)