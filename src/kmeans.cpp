#include "kmeans.h"

#include "common.h"

#include "../faiss/utils.h"
#include "../faiss/Clustering.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>

using namespace std;
using layer_t = IndexHierarchicKmeans::layer_t;


static vector<layer_t> make_layers(const FloatMatrix& vectors, size_t L, size_t nprobe, bool spherical) {
    vector<layer_t> layers = vector<layer_t>(L + 1);

    size_t cnt = vectors.vector_count();
    layers[0].children_range.resize(cnt);
    layers[0].points.resize(cnt, vectors.vector_length);
    for (size_t i = 0; i < cnt; i++) {
        layers[0].children_range[i] = {i, i};
        memcpy(
                layers[0].points.row(i),
                vectors.row(i),
                sizeof(float) * vectors.vector_length);
    }

    for (size_t layer_id = 1; layer_id < L + 1; layer_id++) {
        // Compute number of clusters and cluster size on this layer.
        size_t cluster_num = floor(
                pow(vectors.vector_count(), (L + 1 - layer_id) / (float) (L + 1))
                * pow(nprobe, layer_id / (float) (L + 1))
        );
        printf("Layer %zu size: %zu\n", layer_id, cluster_num);

        auto kr = perform_kmeans(layers[layer_id - 1].points, cluster_num, spherical);

        vector<vector<FloatMatrix>> prev_layer(cluster_num);
        vector<vector<IndexHierarchicKmeans::range>> prev_layer_range(cluster_num);

        for (size_t i = 0; i < kr.assignments.size(); i++) {
            auto cid = kr.assignments[i];
            FloatMatrix row;
            row.resize(1, vectors.vector_length);
            memcpy(
                    row.row(0),
                    layers[layer_id - 1].points.row(i),
                    sizeof(float) * vectors.vector_length);
            prev_layer[cid].push_back(row);
            prev_layer_range[cid].push_back(layers[layer_id - 1].children_range[i]);
        }

        layers[layer_id].points = kr.centroids;
        layers[layer_id].children_range.resize(cluster_num);
        size_t cnt = layers[layer_id - 1].points.vector_count();
        layers[layer_id - 1].dist_to_parent.resize(cnt);

        size_t j = 0;
        for (size_t i = 0; i < cluster_num; i++) {
            size_t start = j;
            for (size_t k = 0; k < prev_layer[i].size(); k++) {
                memcpy(
                        layers[layer_id - 1].points.row(j),
                        prev_layer[i][k].row(0),
                        sizeof(float) * vectors.vector_length);
                layers[layer_id - 1].children_range[j] = prev_layer_range[i][k];
                layers[layer_id - 1].dist_to_parent[j] = sqrt(faiss::fvec_L2sqr(
                        layers[layer_id - 1].points.row(j),
                        kr.centroids.row(i),
                        kr.centroids.vector_length));
                j++;
            }
            size_t end = j;
            layers[layer_id].children_range[i] = {start, end};
        }
    }

    return layers;
}

template <typename T>
struct FixedQueue : public priority_queue<
                    T, vector<T>, greater<T>> {
    FixedQueue(size_t cap) : cap(cap) {
        this->c.reserve(cap + 1);
    }
    void add (T p) {
        this->push(p);
        if (this->size() > cap) {
            this->pop();
        }
    }
    typename std::vector<T>::iterator begin() {
        return this->c.begin();
    }
    typename std::vector<T>::iterator end() {
        return this->c.end();
    }
    size_t cap;
};

static vector<size_t> predict(const vector<layer_t>& layers, FloatMatrix& queries, size_t qnum,
        size_t opened_trees, bool branch_n_bound, size_t k_needed = 1) {

    const float* query = queries.row(qnum);

    FixedQueue<pair<float, size_t>> candidates(opened_trees);
    for (size_t i = 0; i < layers.back().points.vector_count(); i++) {
        float result = faiss::fvec_inner_product(
                query,
                layers.back().points.row(i),
                queries.vector_length);
        candidates.add({result, i});
    }

    for (size_t layer_id = layers.size() - 1; layer_id > 0; layer_id--) {
        FixedQueue<pair<float, size_t>> next_candidates(
                (layer_id == 1) ? k_needed : opened_trees);

        for (auto val_cid: candidates) {
            size_t cid = val_cid.second;
            for (size_t i = layers[layer_id].children_range[cid].left;
                       i < layers[layer_id].children_range[cid].right; i++) {

                if (branch_n_bound && next_candidates.size() >= next_candidates.cap) {
                    float optimistic = val_cid.first +
                        layers[layer_id - 1].dist_to_parent[i];
                    float worst_good = next_candidates.top().first;
                    if (optimistic < worst_good) continue;
                }
                float result = faiss::fvec_inner_product(
                        query,
                        layers[layer_id - 1].points.row(i),
                        queries.vector_length);

                next_candidates.add({result, i});
            }
        }

        swap(candidates, next_candidates);
    }
    vector<size_t> res(candidates.size());
    size_t i = res.size() - 1;
    while (!candidates.empty()) {
        auto top = candidates.top();
        candidates.pop();
        res[i--] = layers[0].children_range[top.second].left;
    }
    return res;
}

IndexHierarchicKmeans::IndexHierarchicKmeans(
        size_t dim, size_t layers_count, size_t opened_trees,
           MipsAugmentation* aug, bool branch_n_bound, bool spherical):
    Index(dim, faiss::METRIC_INNER_PRODUCT),
    layers_count(layers_count), opened_trees(opened_trees),
    branch_n_bound(branch_n_bound),
    spherical(spherical),
    augmentation(aug)
{
}

void IndexHierarchicKmeans::add(idx_t n, const float* data) {
    vectors_original.resize(n, d);
    memcpy(vectors_original.data.data(), data, n * d * sizeof(float));
    auto vectors = augmentation->extend(data, n);
    layers = make_layers(vectors, layers_count, opened_trees, spherical);
}

void IndexHierarchicKmeans::reset() {
    vectors_original.data.clear();
    layers.clear();
}

void IndexHierarchicKmeans::search(idx_t n, const float* data, idx_t k,
        float* distances, idx_t* labels) const {

    FloatMatrix queries_original;
    queries_original.resize(n, d);
    memcpy(queries_original.data.data(), data, n * d * sizeof(float));
    FloatMatrix queries = augmentation->extend_queries(data, n);

    #pragma omp parallel for
    for (size_t i = 0; i < queries.vector_count(); i++) {
        vector<size_t> predictions = predict(layers, queries, i, opened_trees, branch_n_bound, k);
        for (idx_t j = 0; j < k; j++) {
            labels[i * k + j] = (size_t(j) < predictions.size()) ? predictions[j] : -1;
        }

        for (idx_t j = 0; j < k; j++) {
            idx_t lab = labels[i * k + j];
            if (lab != -1) {
                distances[i * k + j] = faiss::fvec_inner_product(
                    vectors_original.row(lab),
                    queries_original.row(i),
                    d
                );
            }
        }
    }
}

void IndexHierarchicKmeans::save(const char* fname) const {
    FILE* f = fopen(fname, "wb");
    write_floatmatrix(vectors_original, f);
    for (const auto& l: layers) {
        write_vec(l.children_range, f);
        write_vec(l.dist_to_parent, f);
        write_floatmatrix(l.points, f);
    }
    fclose(f);
}

void IndexHierarchicKmeans::load(const char* fname) {
    FILE* f = fopen(fname, "rb");
    // Assume parameters are already there (i.e. index was constructed using
    // provided constructor).
    read_floatmatrix(vectors_original, f);
    layers.resize(layers_count + 1);
    for (auto& l: layers) {
        read_vec(l.children_range, f);
        read_vec(l.dist_to_parent, f);
        read_floatmatrix(l.points, f);
    }
    fclose(f);
}
