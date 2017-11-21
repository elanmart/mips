import argparse
import logging
import os
import time

import numpy as np
import faiss

from utils import load_sift, save_sift


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def load(path):
    xt = load_sift(os.path.join(path, "sift_learn.fvecs"))
    xb = load_sift(os.path.join(path, "sift_base.fvecs"))
    xq = load_sift(os.path.join(path, "sift_query.fvecs"))
    gt = load_sift(os.path.join(path, "sift_groundtruth.ivecs"), dtype=np.int32)

    return xt, xb, xq, gt


def _eval(index, xq, gt, prefix=""):
    nq, k_max = gt.shape

    for k in [1, 5, 10]:
        t0 = time.time()
        D, I = index.search(xq, k)
        t1 = time.time()

        recall_at_1 = (I[:, :1] == gt[:, :1]).sum() / float(nq)

        print('\t' + prefix + ": k={} {:.3f} s, R@1 {:.4f}".format(k, t1 - t0, recall_at_1))


def test(data):
    xt, xb, xq, gt = data

    d = xt.shape[1]

    for Index in [faiss.IndexFlatIP, faiss.IndexFlatL2]:
        index = Index(d)
        index.add(xb)

        _eval(index, xq, gt, prefix=index.__class__.__name__)


def generate_gtIP(data, path, skip_tests=False):
    GT_IP_FNAME = 'sift_groundtruth_IP.ivecs'
    logging.info("Generating data to be stored at {}".format(os.path.join(path, GT_IP_FNAME)))

    xt, xb, xq, gt = data

    d = xt.shape[1]
    k = gt.shape[1]

    indexIP = faiss.IndexFlatIP(d)
    indexIP.add(xb)

    _, I = indexIP.search(xq, k)
    # ignore all but the best vector
    I[:, 1:] = -1
    save_sift(I, os.path.join(path, GT_IP_FNAME), dtype=np.int32)

    # sanity-check
    if not skip_tests:
        logging.info("Testing for inner-product ground-truth")
        test((xt, xb, xq, I))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("This script will read a dataset in sift format. "
                                     "And evaluate the performance of FlatIndex L2, and FlatIndex InnerProduct. "
                                     "It'll also create a new groundtruth for the InnerProduct Search")

    parser.add_argument('path',
                        help='Path to the dataset')
    parser.add_argument('--skip-tests',
                        action='store_true',
                        help="Will not execute performance tests of each index, "
                             "and only generate the new groundtruth")

    args = parser.parse_args()
    data = load(args.path)

    if not args.skip_tests:
        logging.info("Testing for default L2-based ground-truth")
        test(data)

    generate_gtIP(data, args.path, skip_tests=args.skip_tests)
