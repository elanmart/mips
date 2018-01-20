
# Introduction

This repository accompanies an engineering dissertation prepared by Marcin Elantkowski, Adam Krasuski, 
Franciszek Walkowiak and Agnieszka Lipska. 

Its main focus is a `maximum inner product search` problem -- we've implemented several algorithms and utilities to 
perform efficient `nearest-neighbour` search according to the `inner-product` metric. 

We've implemented three algorithms as described in

* Clustering is efficient for approximate maximum inner product search (Auvolat, Larochelle, Chandar, Vincent, Bengio)
* Quantization based fast inner product search (Guo, Kumar, Choromanski, Simcha)
* Asymmetric LSH for sublinear maximum inner product search (Shrivastava, Li)

# Compile py src vs compile cpp src

# Installation (python package)

First, install some dependencies
`conda install -c conda-forge -c pytorch numpy mkl cmake pybind11 faiss-cpu`

Then simply run 
`python setup.py install`

# Installation
`TODO:` Note that this is only for `Ubuntu-16.04`

Clone the repo including submodules

```bash
git clone --recursive https://github.com/walkowiak/mips
```

To compile the `C++` sources, you'll need a `BLAS` implementation and a compiler supproting `c++ 11` standard. 
For `BLAS` the easiest way to go is with `OpenBLAS`. You can also use `MKL` installed via `conda`, but setting the
correct linking flags can be a real pain.

To install `OpenBLAS`, run:

```bash
sudo apt install libopenblas-dev
```

You'll also need to *copy* and *modify* a `makefile.inc` from `faiss/example_makefiles` to the repo root directory.
Please refer to `faiss/INSTALL.md` for details regarding the process.

For `Ubuntu-16.04` and `OpenBLAS` you can follow these steps:

```bash
cp  faiss/example_makefiles/makefile.inc.Linux makefile.inc
vim makefile.inc
```
Uncomment `BLASLDFLAGS` for `Ubuntu 16.04` or find appropriate location for your distro.

Comment other irrelevant `BLASLDFLAGS` e.g. for `CentOS`.

If you want to use `Python` wrappers, you'll need to set python include paths (`PYTHONCFLAGS`), for example:

You cant find the correct flags with the command below:
```bash
python -c "import numpy as np ; \
           inc_py= ; \
           inc_np= ; \
           print('PYTHONCFLAGS=-I{} -I{}'.format(inc_py, inc_np)"
```

To use python bindings you'll need `python 3.6+` and 
several packages. We recommend installing them via `conda`:

```bash
conda install
```

To compile simply run:
`make`

To compile python bindings, run:
`make py`