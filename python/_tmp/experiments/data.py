import array
import logging
import os

import numpy as np
import scipy.sparse as sp
import torch as th
from torch.autograd import Variable as V
from torch.utils.data.dataset import Dataset

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def libsvm_to_csr(path):
    """ Read data in libsvm (extreme repo) format, and convert it into scipy csr matrices

    """

    with open(path) as f_in:

        num_documents, num_features, num_labels = [int(val) for val in
                                                   next(f_in).strip().split(' ')]

        y_indices = array.array('I')
        y_indptr = array.array('I', [0])

        x_indices = array.array('I')
        x_data = array.array('f')
        x_indptr = array.array('I', [0])

        for i, line in enumerate(f_in):
            labels, *features = line.strip().split()

            features = [item.split(":") for item in features]
            labels = [int(y) for y in labels.split(',')]

            if len(labels) <= 0 or len(features) <= 0:
                continue

            row_indices, row_values = zip(*features)
            row_indices, row_values = map(int, row_indices), map(float, row_values)

            x_indices.extend(row_indices)
            x_data.extend(row_values)
            x_indptr.append(len(x_indices))

            y_indices.extend(labels)
            y_indptr.append(len(y_indices))

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


def get_data(path, name='train', force=False,
             min_words=1, min_labels=1,
             words_mask=None, labels_mask=None):

    # data paths
    RAW_PATH = os.path.join(path, f'{name}.txt')
    X_PATH   = os.path.join(path, f'X_{name}.csr.npz')
    Y_PATH   = os.path.join(path, f'Y_{name}.csr.npz')

    # data already read
    if os.path.exists(X_PATH) and force is False:
        logging.info(f"Data already present at {X_PATH}. Loading...")

        X = load_csr(X_PATH)
        Y = load_csr(Y_PATH)

        return X, Y, None, None

    # data only in libsvm
    logging.info(f"Data not found or `force` flag was passed, I'm going to prepare it and store at {X_PATH}.")

    X, Y = libsvm_to_csr(RAW_PATH)

    # compute masks to get rid of examples with too little words or labels
    if words_mask is None:
        words_mask = trim(X, dim=0, t=min_words)
        labels_mask = trim(Y, dim=0, t=min_labels)
    else:
        assert labels_mask is not None

    # discard unwanted columns
    X = X.tocsc()[:, words_mask].tocsr()
    Y = Y.tocsc()[:, labels_mask].tocsr()

    # make sure each example has at leas one nonzero feature and one label
    row_mask = trim(X, dim=1, t=1) & trim(Y, dim=1, t=1)
    X = X[row_mask, :]  # type: sp.csr_matrix
    Y = Y[row_mask, :]

    # fix csr matrices
    X.sort_indices()
    X.sum_duplicates()
    Y.sort_indices()
    Y.sum_duplicates()

    # save the result
    save_csr(X, X_PATH)
    save_csr(Y, Y_PATH)

    return X, Y, words_mask, labels_mask


class CSRDataset(Dataset):
    def __init__(self, X_csr, Y_csr):
        assert X_csr.shape[0] == Y_csr.shape[0]

        self.X_csr = X_csr
        self.Y_csr = Y_csr
        self.sorted_indices = None

    def __getitem__(self, index):
        X_row = self.X_csr[index]
        Y_row = self.Y_csr[index]

        return X_row.indices, Y_row.indices

    def __len__(self):
        return self.X_csr.shape[0]


class Preprocessor:
    def __init__(self, n_labels,  scale_y=True, sample_y=False):

        self.n_labels       = n_labels
        self.scale_y        = scale_y
        self.sample_y       = sample_y

        if self.sample_y:
            self.merge_fn = th.LongTensor
        else:
            self.merge_fn = th.stack

    def x_transform(self, x):
        # noinspection PyArgumentList
        return th.LongTensor(x)

    def y_transform(self, labels):
        assert len(labels) > 0

        if self.sample_y:
            return int(np.random.choice(labels))

        value = 1 if (not self.scale_y) else 1. / len(labels)
        y = np.zeros((self.n_labels, ), dtype=np.float32)
        y[labels] = value

        return th.from_numpy(y)

    def _collate_bag(self, xs, ys):
        indices = th.cat(xs)

        offsets, cur_offset = [], 0

        for I in xs:
            offsets.append(cur_offset)
            cur_offset += len(I)

        # noinspection PyArgumentList
        offsets = th.LongTensor(offsets)

        return (indices, offsets), self.merge_fn(ys)

    def collate_fn(self, batch):
        batch = [(self.x_transform(x), self.y_transform(y))
                 for x, y in batch]

        xs, ys = zip(*batch)
        ret_x, ret_y = self._collate_bag(xs, ys)

        return ret_x, ret_y


def get_weights(Y, mode):
    min_freq    = 2. / Y.shape[0]
    frequencies = th.from_numpy(np.array(Y.mean(0)).ravel().astype(np.float32))
    frequencies = th.clamp(frequencies, min_freq, 1.)

    pos_weights = (1. / frequencies)
    neg_weights = (1. / (1 - frequencies))

    assert mode in {None, 'sqrt', 'sqrt-pos', 'row', 'full'}

    if mode == 'sqrt':
        pos_weights = th.sqrt(pos_weights)
        neg_weights = th.sqrt(neg_weights)
    elif mode == 'sqrt-pos':
        pos_weights = th.sqrt(pos_weights)
    elif mode == 'row':
        summed = pos_weights + neg_weights
        pos_weights = pos_weights / summed
        neg_weights = neg_weights / summed
    elif mode == 'full':
        total = th.sum(pos_weights) + th.sum(neg_weights)
        pos_weights = pos_weights / total
        neg_weights = neg_weights / total

    return pos_weights, neg_weights


def to_cuda_var(*x, cuda=False, var=True, volatile=False, requires_grad=False):

    if len(x) == 1:
        x, = x

    if x is None:
        return x

    typ = type(x)
    if typ in {list, tuple}:
        return typ(to_cuda_var(item, cuda=cuda) for item in x)

    if var:
        if cuda:
            # noinspection PyUnresolvedReferences
            x = x.cuda(async=True)

        x = V(x, requires_grad=requires_grad, volatile=volatile)

    else:
        if cuda:
            # noinspection PyUnresolvedReferences
            x = x.cuda()

    return x


def save_csr(obj, filename):
    np.savez(filename, data=obj.data, indices=obj.indices, indptr=obj.indptr,
             shape=obj.shape)


def load_csr(filename: str):
    loader = np.load(filename)

    data    = loader['data']
    indices = loader['indices']
    indptr  = loader['indptr']
    shape   = loader['shape']

    return sp.csr_matrix((data, indices, indptr),
                         shape=shape)