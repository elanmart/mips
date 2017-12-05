This library is based on ideas from these articles:
 * Clustering is efficient for approximate maximum inner product search (Auvolat, Larochelle, Chandar, Vincent, Bengio)
 * Quantization based fast inner product search (Guo, Kumar, Choromanski, Simcha)
 * Asymmetric LSH for sublinear maximum inner product search (Shrivastava, Li)

## Setup
Here, it is described how to prepare your system for using the library.
This manual assumes you have clean and up-to-date Ubuntu 16.04 LTS installation.
Adapting it for other Linux distros shouldn't be too hard if you know the steps given below.

1. Install packages and clone this repository.

```bash
apt install git libopenblas-dev python-numpy python-dev python3-numpy python3-dev
git clone --recursive https://github.com/walkowiak/mips
```

2. Copy faiss' makefile and customize it.

```bash
cp mips/faiss/example_makefiles/makefile.inc.Linux mips/makefile.inc
nano mips/makefile.inc
```

Uncomment `BLASLDFLAGS` for Ubuntu 16.04 or find appropriate location for your distro.
Comment other irrelevant `BLASLDFLAGS` e.g. for CentOS.
If you plan to use Python 3 wrapper, change `PYTHONCFLAGS`, for example:

```bash
PYTHONCFLAGS=-I/usr/include/python3.5/ -I/usr/lib/python3/dist-packages/numpy/core/include/
```

These paths may vary depending on exact Python version and distro. If you use Anaconda they could as well be
```bash
PYTHONCFLAGS=-I/home/<user>/anaconda3/include/python3.6m -I/home/<user>/anaconda3/lib/python3.6/site-packages/numpy/core/include
```

3. Compile the library.
```bash
cd mips
make -j10
```

## Benchmarking on sift dataset

4. Download dataset and extract it to `data` directory.

```bash
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xf sift.tar.gz -C data
rm sift.tar.gz
mv data/sift data/sift1M
```

5. Compile Python wrapper and generate MIPS ground truth (by default it is NN ground truth).

```bash
cd faiss
make py
cd ..
export PYTHONPATH=faiss
python3 python/misc/make_gt_IP.py --skip-tests data/sift1M
```

6. Run example benchmarks.
```bash
bin/bench_kmeans 2 1 0.85 0 40 80 120
bin/bench_quant 128 32 0 0 0
bin/bench_alsh 64 96 10 1 0.85
```
