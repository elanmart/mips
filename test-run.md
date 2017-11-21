get data from LSHTC

```bash
https://drive.google.com/open?id=0B3lPMIHmG6vGSHE1SWx4TVRva3c
```

Prepare data with python
```python
>>> from experiments.data import get_data
>>> from experiments.misc.utils import to_ft
>>> X, Y = get_data('./data/LSHTC', 'train')
>>> to_ft(X, Y, './data/LSHTC/ft/train.txt')
>>> X, Y = get_data('./data/LSHTC', 'test')
>>> to_ft(X, Y, './data/LSHTC/ft/test.txt')
```

train the model:
```bash
cd fastText
./fasttext supervised -input ../data/LSHTC/ft/train.txt \
                      -output ../data/LSHTC/ft/model.ft \
                      -minCount      5     \
                      -minCountLabel 5     \
                      -lr            0.1   \
                      -lrUpdateRate  100   \
                      -dim           256   \
                      -ws            5     \
                      -epoch         25    \
                      -neg           25    \
                      -loss          ns    \
                      -thread        8     \
                      -saveOutput    1
```

Train the index
```bash
./fasttext train-index ../data/LSHTC/ft/model.ft.bin
```

Use the index
```bash
time ./fasttext approx-predict ../data/LSHTC/ft/model.ft.bin ../data/LSHTC/ft/test.txt > /tmp/approx.txt
167,51s user 51,81s system 663% cpu 33,067 total

time ./fasttext predict ../data/LSHTC/ft/model.ft.bin ../data/LSHTC/ft/test.txt > /tmp/exact.txt
```

WRITEME
```bash
time ./fasttext to-fvecs ../data/LSHTC/ft/model.ft.bin ../data/LSHTC/ft/test.txt /tmp/fvecs
```
