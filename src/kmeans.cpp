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

using namespace std;
using layer_t = IndexHierarchicKmeans::layer_t;


static vector<layer_t> make_layers(const FloatMatrix& vectors, size_t L) {
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
                pow(vectors.vector_count(), (L + 1 - layer_id) / (float) (L + 1)));

        auto kr = perform_kmeans(layers[layer_id - 1].points, cluster_num);

        vector<vector<FloatMatrix>> prev_layer(cluster_num);
        vector<vector<pair<size_t, size_t>>> prev_layer_range(cluster_num);

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
        size_t j = 0;
        for (size_t i = 0; i < cluster_num; i++) {
            size_t start = j;
            for (size_t k = 0; k < prev_layer[i].size(); k++) {
                memcpy(
                        layers[layer_id - 1].points.row(j),
                        prev_layer[i][k].row(0),
                        sizeof(float) * vectors.vector_length);
                layers[layer_id - 1].children_range[j] = prev_layer_range[i][k];
                j++;
            }
            size_t end = j;
            layers[layer_id].children_range[i] = {start, end};
        }
    }

    return layers;
}

static vector<size_t> predict(const vector<layer_t>& layers, FloatMatrix& queries, size_t qnum,
        size_t opened_trees, size_t k_needed = 1) {

    const float* query = queries.row(qnum);

    vector<pair<float, size_t>> candidates;
    for (size_t i = 0; i < layers.back().points.vector_count(); i++) {
        float result = faiss::fvec_inner_product(
                query, 
                layers.back().points.row(i),
                queries.vector_length);
        candidates.push_back({result, i});
    }

    for (size_t layer_id = layers.size() - 1; layer_id > 0; layer_id--) {
        if (candidates.size() > opened_trees) {
            nth_element(
                    candidates.begin(), 
                    candidates.begin() + opened_trees,
                    candidates.end(),
                    greater<std::pair<float, size_t>>());
            candidates.resize(opened_trees);
        }
        // sort?

        vector<pair<float, size_t>> next_candidates;

        for (auto val_cid: candidates) {
            size_t cid = val_cid.second;
            for (size_t i = layers[layer_id].children_range[cid].first;
                       i < layers[layer_id].children_range[cid].second; i++) {
                
                float result = faiss::fvec_inner_product(
                        query,
                        layers[layer_id - 1].points.row(i),
                        queries.vector_length);

                next_candidates.push_back({result, i});
            }
        }

        swap(candidates, next_candidates);
    }
    // Last layer - find best match.
    if (candidates.size() > k_needed) {
        nth_element(
                candidates.begin(), 
                candidates.begin() + k_needed,
                candidates.end(),
                greater<std::pair<float, size_t>>());
        candidates.resize(k_needed);
    }
    sort(candidates.rbegin(), candidates.rend());

    vector<size_t> res;
    for (size_t i = 0; i < candidates.size(); i++) {
        res.push_back(layers[0].children_range[candidates[i].second].first);
    }
    return res;
}

IndexHierarchicKmeans::IndexHierarchicKmeans(
        size_t dim, size_t layers_count, size_t opened_trees, MipsAugmentation* aug):
    Index(dim, faiss::METRIC_INNER_PRODUCT),
    layers_count(layers_count), opened_trees(opened_trees), augmentation(aug)
{
}

void IndexHierarchicKmeans::add(idx_t n, const float* data) {
    vectors_original.resize(n, d);
    memcpy(vectors_original.data.data(), data, n * d * sizeof(float));
    vectors = augmentation->extend(data, n);
    layers = make_layers(vectors, layers_count);
}

void IndexHierarchicKmeans::reset() {
    vectors.data.clear();
    vectors_original.data.clear();
    layers.clear();
}

void IndexHierarchicKmeans::search(idx_t n, const float* data, idx_t k, 
        float* distances, idx_t* labels) const {
    FloatMatrix queries_original;
    queries_original.resize(n, d);
    memcpy(queries_original.data.data(), data, n * d * sizeof(float));
    FloatMatrix queries = augmentation->extend_queries(data, n);

    FlatMatrix<idx_t> labels_matrix;
    labels_matrix.resize(n, k);
    #pragma omp parallel for
    for (size_t i = 0; i < queries.vector_count(); i++) {
        vector<size_t> predictions = predict(layers, queries, i, opened_trees, k);
        for (idx_t j = 0; j < k; j++) {
            labels_matrix.at(i, j) = (size_t(j) < predictions.size()) ? predictions[j] : -1;
        }

        for (idx_t j = 0; j < k; j++) {
            idx_t lab = labels_matrix.at(i, j);
            if (lab != -1) {
                distances[i * k + j] = faiss::fvec_inner_product(
                    vectors_original.row(lab),
                    queries_original.row(i),
                    d
                );
            }
        }
    }
    memcpy(labels, labels_matrix.data.data(), n * k * sizeof(idx_t));
}
