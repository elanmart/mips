import numpy as np
import scipy.sparse as sp
import os


def load_sift(fname, dtype=np.float32):
    data = np.fromfile(fname, dtype=dtype)
    d = data[0].view(np.int32)

    data = data.reshape(-1, d + 1)[:, 1:]
    data = np.ascontiguousarray(data.copy())

    return data


def save_sift(obj, fname, dtype=np.float32):
    obj = np.hstack([
        np.ones((obj.shape[0], 1)) * obj.shape[1],
        obj
    ]).astype(dtype)

    obj.tofile(fname)


def to_ft(X: sp.csr_matrix, Y: sp.csr_matrix, fname: str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)

    with open(fname, 'w') as f:

        for row_x, row_y in zip(X, Y):
            xs, ys = row_x.indices, row_y.indices

            labels = [f'__label__{y}' for y in ys]
            feats  = [f'feature_{x}' for x in xs]

            line = ' '.join(labels + feats)

            print(line, file=f)
