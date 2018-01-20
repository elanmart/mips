# test_capitalize.py
import os

import numpy as np
import pytest
import torch as th
from scipy import sparse as sp
from torch.autograd import Variable

from experiments.data import (libsvm_to_csr,
                              save_csr, load_csr,
                              trim, to_cuda_var, get_data, CSRDataset, LocallySequentialSampler, Preprocessor)


@pytest.fixture(scope='class')
def libsvm_data(tmpdir_factory):
    fn = tmpdir_factory.mktemp('libsvm').join('train.txt')
    fn = str(fn)

    labels = np.array([
        [1,  1,  0,  0],
        [1,  0,  0,  0],
        [0,  1,  1,  0],
        [0,  0,  0,  1],
    ])

    features = np.array([
        [1.1,  0.0,  1.7,  0.0,  0.0],
        [1.2,  1.3,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  1.4,  0.0],
        [0.0,  1.5,  1.6,  0.0,  0.0],
    ])

    contents = """4 5 4
0,1 0:1.1 2:1.7
0 0:1.2 1:1.3
1,2 3:1.4
3 1:1.5 2:1.6"""

    with open(fn, 'w') as f:
        f.write(contents)

    return fn, features, labels


def to_arr(*items):
    if len(items) == 1:
        obj = items[0]

        if sp.issparse(obj):
            obj = obj.todense()

        if type(obj) in {th.FloatTensor, th.LongTensor}:
            obj = obj.numpy()

        return np.array(obj)

    return [to_arr(csr) for csr in items]


def assert_identical(a, b):
    a, b = to_arr(a, b)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_libsvm_to_csr(libsvm_data):
    fn, features, labels = libsvm_data
    X, Y = libsvm_to_csr(fn)

    assert X.shape == features.shape
    assert Y.shape == labels.shape

    assert_identical(X, features)
    assert_identical(Y, labels)


def test_save_load(tmpdir_factory):
    fn = str(tmpdir_factory.mktemp('sparse').join('X.npz'))

    X = sp.csr_matrix(np.array([
        0, 0, 1.,
        2., 0, 0,
        1.3, 4., 50.,
        0., 0., 0.
    ]))

    save_csr(X, fn)
    assert_identical(X, load_csr(fn))


def test_trim_cols():
    X_in = sp.csr_matrix(np.array([
        [0., 0., 1.],
        [0., 1., 1.]
    ]))

    mask_1 = np.array([False, True, True])
    mask_2 = np.array([False, False, True])

    assert_identical(trim(X_in, dim=0, t=1), mask_1)
    assert_identical(trim(X_in, dim=0, t=2), mask_2)


def test_trim_rows():
    X_in = sp.csr_matrix(np.array([
        [0., 0., 0.],
        [0., 0., 1.],
        [0., 1., 1.],
    ]))

    mask_1 = np.array([False, True, True])
    mask_2 = np.array([False, False, True])

    assert_identical(trim(X_in, dim=1, t=1), mask_1)
    assert_identical(trim(X_in, dim=1, t=2), mask_2)


def test_to_cuda_var():
    x = th.randn(2, 2)
    m = th.nn.Linear(2, 2)

    assert isinstance(to_cuda_var(x, cuda=False, var=False, volatile=False, requires_grad=False),
                      th.FloatTensor)

    assert isinstance(to_cuda_var(x, cuda=True, var=False, volatile=False, requires_grad=False),
                      th.cuda.FloatTensor)

    assert isinstance(to_cuda_var(x, cuda=False, var=True, volatile=False, requires_grad=False),
                      Variable)

    assert isinstance(to_cuda_var(x, cuda=False, var=True, volatile=False, requires_grad=False).data,
                      th.FloatTensor)

    assert isinstance(to_cuda_var(x, cuda=True, var=True, volatile=False, requires_grad=False).data,
                      th.cuda.FloatTensor)

    assert isinstance(to_cuda_var(m, cuda=True, var=False, volatile=False, requires_grad=False),
                      th.nn.Linear)

    assert to_cuda_var(x, cuda=False, var=True, volatile=False, requires_grad=False).is_leaf
    assert to_cuda_var(x, cuda=True,  var=True, volatile=False, requires_grad=False).is_leaf
    assert to_cuda_var(x, cuda=False, var=True, volatile=False, requires_grad=True).is_leaf
    assert to_cuda_var(x, cuda=True,  var=True, volatile=False, requires_grad=True).is_leaf
    assert to_cuda_var(x, cuda=False, var=True, volatile=True, requires_grad=False).is_leaf
    assert to_cuda_var(x, cuda=True, var=True, volatile=True, requires_grad=False).is_leaf


def test_get_data_simple(libsvm_data):
    fn, features, labels = libsvm_data
    dirname = os.path.dirname(fn)

    X, Y = get_data(dirname, 0, 0)

    assert_identical(X, features)
    assert_identical(Y, labels)

    X, Y = get_data(dirname, 0, 1)

    assert_identical(X, features)
    assert_identical(Y, labels)


def test_get_data_trimmed_Y(libsvm_data):
    fn, features, labels = libsvm_data
    dirname = os.path.dirname(fn)

    X, Y = get_data(dirname, 0, 2)

    target_features = np.array([
        [1.1,  0.0,  1.7,  0.0,  0.0],
        [1.2,  1.3,  0.0,  0.0,  0.0],
        [0.0,  0.0,  0.0,  1.4,  0.0],
    ])

    target_labels = np.array([
        [1, 1],
        [1, 0],
        [0, 1]
    ])

    assert_identical(Y, target_labels)
    assert_identical(X, target_features)


def test_get_data_trimmed_XY(libsvm_data):
    fn, features, labels = libsvm_data
    dirname = os.path.dirname(fn)

    X, Y = get_data(dirname, 2, 2)

    target_features = np.array([
        [1.1,  0.0,  1.7],
        [1.2,  1.3,  0.0],
    ])

    target_labels = np.array([
        [1, 1],
        [1, 0],
    ])

    assert_identical(X, target_features)
    assert_identical(Y, target_labels)


def test_CSRDataset_random(libsvm_data):
    fn, gt_features, gt_labels = libsvm_data
    X, Y = libsvm_to_csr(fn)

    dataset = CSRDataset(X, Y, sorted=False)

    for i in range(len(dataset)):
        (indices, weights), labels = dataset[i]

        true_indices = gt_features[i, :].nonzero()[0].ravel()

        assert_identical(indices, true_indices)
        assert_identical(weights, gt_features[i, true_indices])

        true_labels = gt_labels[i, :].nonzero()[0].ravel()
        assert_identical(labels, true_labels)

    assert (i + 1) == X.shape[0]


def test_CSRDataset_sorted():
    X = sp.csr_matrix(np.array([
        [1.4, 1.5, 1.6],
        [0.0, 0.0, 1.1],
        [0.0, 1.2, 1.3]
    ]))

    Y = sp.csr_matrix(np.array([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]))

    dataset = CSRDataset(X, Y, sorted=True)

    xs = [X[1, :], X[2, :], X[0, :]]
    ys = [Y[1, :], Y[2, :], Y[0, :]]

    for i in range(len(dataset)):
        (indices, weights), labels = dataset[i]

        assert_identical(indices, xs[i].indices)
        assert_identical(weights, xs[i].data)
        assert_identical(labels, ys[i].indices)

    assert (i + 1) == X.shape[0]


def test_LocallySequentialSampler():
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    window = 4
    n_windows = len(data) // window

    sampler = LocallySequentialSampler(data, window)
    it = iter(sampler)

    for i in range(n_windows):
        last = None

        for j in range(window):
            item = next(it)
            assert (last is None) or (item - last == 1)

            last = item

        assert j == (window - 1)
    assert i == (n_windows - 1)

    with pytest.raises(StopIteration):
        next(it)


def test_Preprocessor():
    preproc = Preprocessor(4,
                           scale_y=False,
                           sample_y=False,
                           weight_y=None,
                           log_transform=False,
                           sqrt_transform=False,
                           use_bag=False,
                           max_repeat=1,
                           subsample=np.inf)

    X = sp.csr_matrix(np.array([
        [1.1, 1.2, 1.3],
        [0.0, 0.0, 1.4],
        [0.0, 1.5, 1.6]
    ]))

    Y = sp.csr_matrix(np.array([
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]
    ]))

    batch = [
        [(X[0].indices, X[0].data), Y[0].indices],
        [(X[1].indices, X[1].data), Y[1].indices],
        [(X[2].indices, X[2].data), Y[2].indices]
    ]

    (I, W), Y = preproc.collate_fn(batch)
