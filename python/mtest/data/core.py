import array
import logging
import os
import subprocess

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from mtest.utils.data import load_csr, save_csr

logger = logging.getLogger(__name__)


def libsvm_to_csr(path):

    with open(path) as f_in:

        num_documents, num_features, num_labels = [int(val) for val in
                                                   next(f_in).strip().split(' ')]

        y_indices = array.array('I')
        y_indptr = array.array('I', [0])

        x_indices = array.array('I')
        x_data = array.array('f')
        x_indptr = array.array('I', [0])

        with tqdm(total=num_documents, desc=f'libsvm to csr for {path}') as pb:

            for i, line in enumerate(f_in):
                labels, *features = line.strip().split()

                features = [item.split(":") for item in features]
                labels = [int(y) for y in labels.split(',')]

                if len(features) == 0:
                    row_indices, row_values = [], []

                else:
                    row_indices, row_values = zip(*features)
                    row_indices, row_values = map(int, row_indices), map(float, row_values)

                x_indices.extend(row_indices)
                x_data.extend(row_values)
                x_indptr.append(len(x_indices))

                y_indices.extend(labels)
                y_indptr.append(len(y_indices))

                pb.update(1)

        x_indices = np.frombuffer(x_indices, dtype=np.uint32)
        x_indptr = np.frombuffer(x_indptr, dtype=np.uint32)
        x_data = np.frombuffer(x_data, dtype=np.float32)
        x_shape = (num_documents, num_features)

        y_indices = np.frombuffer(y_indices, dtype=np.uint32)
        y_indptr = np.frombuffer(y_indptr, dtype=np.uint32)
        y_data = np.ones_like(y_indices, dtype=np.float32)
        y_shape = (num_documents, num_labels)

        X = sp.csr_matrix((x_data, x_indices, x_indptr), shape=x_shape)
        Y = sp.csr_matrix((y_data, y_indices, y_indptr), shape=y_shape)

        return X, Y


def trim(_X, dim, t):
    _X   = _X.tocsc() if (dim == 0) else _X.tocsr()
    _mask = np.array((_X > 0).sum(dim) >= t).ravel()

    return _mask


def load_libsvm(path, name='train', force=False,
                min_words=1, min_labels=1,
                words_mask=None, labels_mask=None):

    # data paths
    RAW_PATH = os.path.join(path, f'{name}.txt')
    X_PATH   = os.path.join(path, f'X_{name}.csr.npz')
    Y_PATH   = os.path.join(path, f'Y_{name}.csr.npz')

    # data already read
    if os.path.exists(X_PATH) and force is False:
        logger.info(f"Data already present at {X_PATH}. Loading...")

        X = load_csr(X_PATH)
        Y = load_csr(Y_PATH)

        return X, Y, None, None

    # data only in libsvm
    logger.info(f"Data not found or `force` flag was passed.")
    logger.debug(f"I'm going to prepare it and store at {X_PATH}.")

    X, Y = libsvm_to_csr(RAW_PATH)

    logger.debug('# compute masks to get rid of examples with too little words or labels')
    if words_mask is None:
        words_mask = trim(X, dim=0, t=min_words)
        labels_mask = trim(Y, dim=0, t=min_labels)
        

    logger.debug('# discard unwanted columns')
    X = X.tocsc()[:, words_mask].tocsr()
    Y = Y.tocsc()[:, labels_mask].tocsr()

    logger.debug('# make sure each example has at leas one nonzero feature and one label')
    row_mask = trim(X, dim=1, t=1) & trim(Y, dim=1, t=1)
    X = X[row_mask, :]  # type: sp.csr_matrix
    Y = Y[row_mask, :]

    logger.debug('# fix csr matrices')
    X.sort_indices()
    X.sum_duplicates()
    Y.sort_indices()
    Y.sum_duplicates()

    logger.debug('# save the result')
    save_csr(X, X_PATH)
    save_csr(Y, Y_PATH)

    return X, Y, words_mask, labels_mask


def to_ft(X: sp.csr_matrix, Y: sp.csr_matrix, fname: str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, 'w') as f:

        for row_x, row_y in tqdm(zip(X, Y), total=X.shape[0], desc=f'to_ft ({fname})'):
            xs, ys = row_x.indices, row_y.indices

            labels = [f'__label__{y}' for y in ys]
            feats  = [f'feature_{x}' for x in xs]

            line = ' '.join(labels + feats)

            print(line, file=f)


def prepare(path_in, path_out, force=False):
    os.makedirs(path_out, exist_ok=True)

    logger.debug('Preparing train csr')
    (X_tr, 
     Y_tr, 
     w_mask, 
     l_mask) = load_libsvm(path_in, 'train', min_words=3, min_labels=3, force=force)

    logger.debug('Preparing test csr')
    (X_te,
     Y_te, 
     _, 
     __) = load_libsvm(path_in, 'test', words_mask=w_mask, labels_mask=l_mask, force=force)

    logger.debug('Preparing ft')
    to_ft(X_tr, Y_tr, os.path.join(path_out, 'train.ft.txt'))
    to_ft(X_te, Y_te, os.path.join(path_out, 'test.ft.txt'))


def _fasttext_cmd(path, *args, **kwargs):
    args = ' '.join(args)
    opts = ' '.join(f'-{k} {v}' for k, v in kwargs.items())
    cmd  = f'{path} {args} {opts}'
    
    return cmd.split()


def make_ft_fvecs(fasttext, path, thread):
    path = str(path)

    train_cmd = _fasttext_cmd(fasttext, 'supervised',
                              input         = os.path.join(path, 'train.ft.txt'),
                              output        = os.path.join(path, 'model.ft'),
                              minCount      = 3,
                              minCountLabel = 3,
                              lr            = 0.1,
                              lrUpdateRate  = 100,
                              dim           = 256,
                              ws            = 5,
                              epoch         = 25,
                              neg           = 25,
                              loss          = 'ns',
                              thread        = thread,
                              saveOutput    = 1)

    generate_cmd = _fasttext_cmd(fasttext, 'to-fvecs',
                                 os.path.join(path, 'model.ft.bin'),
                                 os.path.join(path, 'test.ft.txt'),
                                 os.path.join(path, 'data'))

    subprocess.call(train_cmd)
    subprocess.call(generate_cmd)
