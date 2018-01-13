import logging
import time
from os.path import join

import faiss
import numpy as np
from tqdm import tqdm

from ..utils.data import load_sift, save_sift

logger = logging.getLogger(__name__)
GT_IP_FNAME = 'sift_groundtruth.IP.ivecs'
GT_IP_TXT   = 'data.labels.txt'
MAGIC_NUM   = -90378


def load(path):
    logger.debug(f'Loading {path}')

    xt = load_sift(join(path, "sift_learn.fvecs"))
    xb = load_sift(join(path, "sift_base.fvecs"))
    xq = load_sift(join(path, "sift_query.fvecs"))
    gt = load_sift(join(path, "sift_groundtruth.ivecs"), dtype=np.int32)

    return xt, xb, xq, gt


def _save(I, path):
    logger.debug(f'saving {GT_IP_FNAME}')
    save_sift(I, join(path, GT_IP_FNAME), dtype=np.int32)

    logger.debug(f'saving {GT_IP_TXT}')
    with open(join(path, GT_IP_TXT), 'w') as f:

        for row in tqdm(I):
            row = ' '.join([str(item) for item in row])
            print(row, file=f)


def _eval(index, xq, gt, prefix=""):
    nq, k_max = gt.shape

    for k in [1, 5, 10]:
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)

        print('\t' + prefix + ": k={} {:.3f} s, R@1 {:.4f}".format(k, t1 - t0, recall_at_1))


def test(data):
    logger.debug('Testing')

    xt, xb, xq, gt = data

    d = xt.shape[1]

    for Index in [faiss.IndexFlatIP, faiss.IndexFlatL2]:
        index = Index(d)
        index.add(xb)

        _eval(index, xq, gt, prefix=index.__class__.__name__)


def generate_gtIP(data, path, skip_tests=False):
    logger.info("Generating data to be stored at {}".format(join(path, GT_IP_FNAME)))

    if not isinstance(data, tuple):
        data = load(str(data))

    logger.debug(f'Preparing the index')

    xt, xb, xq, gt = data

    d = xt.shape[1]
    k = gt.shape[1]

    indexIP = faiss.IndexFlatIP(d)
    indexIP.add(xb)

    logger.debug('Trainig the index')

    _, I = indexIP.search(xq, k)

    logger.debug('Predictions ready')

    # ignore all but the best vector
    I[:, 1:] = MAGIC_NUM

    _save(I, path)

    # sanity-check
    if not skip_tests:
        logger.info("Testing for inner-product ground-truth")
        test((xt, xb, xq, I))
