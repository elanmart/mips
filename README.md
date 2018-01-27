
# Introduction

This repository accompanies an engineering dissertation prepared by Marcin Elantkowski, Adam Krasuski, 
Franciszek Walkowiak and Agnieszka Lipska. 

Its main focus is a `maximum inner product search` problem -- we've implemented several algorithms and utilities to 
perform efficient `nearest-neighbour` search according to the `inner-product` metric. 

We've implemented three algorithms as described in

* Clustering is efficient for approximate maximum inner product search (Auvolat, Larochelle, Chandar, Vincent, Bengio)
* Quantization based fast inner product search (Guo, Kumar, Choromanski, Simcha)
* Asymmetric LSH for sublinear maximum inner product search (Shrivastava, Li)

The `hierarchical k-means` implementation of Auvolat et. al is quite fast (faster than `FAISS`), 
the other two not so much.

This repository also provides some examples of how you can incorporate an index in your `MIPS`-bounded code. 

# Compilation

## python-only
To use only the python utils, 
you can just run 

```bash
conda install -c pytorch faiss-cpu
python setup.py install
```

## python + our c++ indexes
To build the `C++` code in this repo, you'll also need to build `FAISS`. 
To build on `Ubuntu-16.04` with `openblas` isntalled (`sudo apt install libopenblas-dev`)
all you have to do is 

```bash
conda install -c conda-forge pybind11
git clone --recursive https://github.com/elanmart/mips
make
python setup.py install
```

If you're on a different platform, you'll need to adjust `makefile.inc` according to instructions 
in `faiss/INSTALLATION`

You can also use a different `BLAS` implementation, but for `mkl` the compilation is a real pain. 

# Examples

See `python/examples` for some examples.