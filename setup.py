# See https://github.com/pybind/python_example
# and https://github.com/facebookresearch/fastText/blob/master/setup.py

import os
from itertools import chain
from os import path as osp

import pybind11
from setuptools import setup, Extension

_SRC = "src"
_cpp_sources = [osp.join(_SRC, fname)
                for fname in os.listdir(_SRC)
                if fname.endswith('.cpp')]

ext_modules = [
    Extension(name='pymips_wrapper',
              sources=chain(
                  _cpp_sources,
                  ['python/pybind/mips_wrapper.cpp'],
              ),
              include_dirs=[
                  pybind11.get_include(False),
                  pybind11.get_include(True),
                  _SRC,
              ],
              language='c++',
              extra_compile_args=["-std=c++11 -fopenmp -fPIC -O3 -funroll-loops -pthread -march=native"]
              )
]


def _readme():
    with open("README.md") as f:
        return f.read()


setup(
    name='fasttext',
    version='0.1.0',
    author='Marcin Elantkowski',
    author_email='marcin.elantkowski@gmail.com',
    description='MIPS library wrapper',
    long_description=_readme(),
    ext_modules=ext_modules,
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
    ],
    package_dir={
        '': 'python'
    },
    zip_safe=False,
)