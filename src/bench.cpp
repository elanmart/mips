#include "bench.h"

#include <cstdio>
#include <cassert>
#include <set>

#include <sys/time.h>

#include "../faiss/AutoTune.h"
#include "../src/common.h"


double elapsed () {
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

#if 1
std::string filenames[4] = {
    "data/sift1M/sift_learn.fvecs",
    "data/sift1M/sift_base.fvecs",
    "data/sift1M/sift_query.fvecs",
    "data/sift1M/sift_groundtruth_IP.ivecs",
};
#endif
#if 0
std::string filenames[4] = {
     "data/siftsmall/sift_learn.fvecs",
     "data/siftsmall/sift_base.fvecs",
     "data/siftsmall/sift_query.fvecs",
     "data/siftsmall/sift_groundtruth_IP.ivecs",
};
#endif

faiss::Index* bench_train(faiss::Index* get_trained_index(const FloatMatrix& xt)) {
    double t0 = elapsed();

    faiss::Index* index;
    size_t d;

    printf ("[%.3f s] Loading train set\n", elapsed() - t0);

    // Note: training on full vector set.
    FloatMatrix xt = load_vecs<float>(filenames[1]);
    d = xt.vector_length;

    printf ("[%.3f s] Preparing index d=%zu and training on %zu vectors\n",
            elapsed() - t0, d, xt.vector_count());
    double begin_train = elapsed();
    index = get_trained_index(xt);
    double train_time = elapsed() - begin_train;

    printf("Train time = %.6f\n", train_time);

    return index;
}

void bench_add(faiss::Index* index) {
    double t0 = elapsed();

    size_t d = index->d;

    printf ("[%.3f s] Loading database\n", elapsed() - t0);

    FloatMatrix xb = load_vecs<float>(filenames[1]);
    size_t d2 = xb.vector_length;
    size_t nb = xb.vector_count();
    assert(d == d2 || !"dataset does not have same dimension as train set");

    printf ("[%.3f s] Indexing database, size %ld*%ld\n",
            elapsed() - t0, nb, d);
    double begin_add = elapsed();
    index->add(nb, xb.data.data());
    double add_time = elapsed() - begin_add;
    printf("Add time = %.6f\n", add_time);
}

void bench_query(faiss::Index* index) {
    double t0 = elapsed();

    size_t d = index->d;

    FloatMatrix xq;
    size_t nq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        xq = load_vecs<float>(filenames[2]);
        size_t d2 = xq.vector_length;
        nq = xq.vector_count();
        assert(d == d2 || !"query does not have same dimension as train set");

    }

    size_t k; // nb of results per query in the GT
    FlatMatrix<faiss::Index::idx_t> gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        FlatMatrix<int> gt_int = load_vecs<int>(filenames[3]);
        k = gt_int.vector_length;
        size_t ngt = gt_int.vector_count();

        assert(ngt == nq || !"incorrect nb of ground truth entries");

        gt.resize(ngt, k);
        for(size_t i = 0; i < k * nq; i++) {
            gt.data[i] = gt_int.data[i];
        }
    }


    {
        printf ("[%.3f s] Perform a search on %ld queries\n",
                elapsed() - t0, nq);

        // output buffers
        faiss::Index::idx_t *I = new faiss::Index::idx_t[nq * k];
        float *D = new float[nq * k];

        double begin_search = elapsed();
        index->search(nq, xq.data.data(), k, D, I);
        double search_time = elapsed() - begin_search;

        printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(size_t i = 0; i < nq; i++) {
            // construct a set with nearsets neighbours of queries
            std::set<int> nn_set;
            for (size_t j = 0; j < k; j++) {
                if (gt.at(i, j) != -1)
                    nn_set.insert(gt.at(i, j));
            }

            for(size_t j = 0; j < k; j++) {
                // check if result is in the set of nn
                if (nn_set.find(I[i * k + j]) != nn_set.end()) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }

        // find intersection of ground truth top100 and our top100
        size_t common_count = 0;
        for (size_t i = 0; i < nq; i++) {
            for (size_t l = 0; l < 100; l++) {
                int current_val = gt.at(i, l);
                for (size_t j = 0; j < 100; j++) {
                    if (I[i * k + j] == current_val) {
                        common_count++;
                        break;
                    }
                }
            }
        }

        printf("Intersection = %.6f\n", common_count / float(100 * nq));
        printf("Search time = %.6f\n", search_time);
        printf("R@1 = %.6f\n", n_1 / float(nq));
        printf("R@10 = %.6f\n", n_10 / float(nq));
        printf("R@100 = %.6f\n", n_100 / float(nq));
    }
}
