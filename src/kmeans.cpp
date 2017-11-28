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
#include <set>
#include <queue>

using namespace std;
using layer_t = IndexHierarchicKmeans::layer_t;


static vector<layer_t> make_layers(const FloatMatrix& vectors, size_t L) {
    vector<layer_t> layers = vector<layer_t>(L);

    for (size_t layer_id = 0; layer_id < L; layer_id++) {
        layer_t& layer = layers[layer_id];

        // Compute number of clusters and cluster size on this layer.
        size_t cluster_size = floor(
                pow(vectors.vector_count(), (layer_id + 1) / (float) (L + 1)));

        layer.cluster_num = floor(
                vectors.vector_count() / (float) cluster_size);

        const FloatMatrix& points = (layer_id == 0) ?
               vectors : layers[layer_id - 1].kr.centroids;

        layer.kr = perform_kmeans(points, layer.cluster_num);

        layer.centroid_children.resize(layer.cluster_num);
        for (size_t i = 0; i < layer.kr.assignments.size(); i++) {
            layer.centroid_children[layer.kr.assignments[i]].push_back(i);
        }
    }

    return layers;
}

static vector<size_t> predict(const vector<layer_t>& layers, FloatMatrix& queries, size_t qnum,
        size_t opened_trees, const FloatMatrix& vectors, size_t k_needed = 1) {

    // Priority queue of {layer, centroid number} indexed by IP.
    priority_queue<pair<float, pair<size_t, size_t>>> queue;
    // The last layer (i.e. original points)
    vector<std::pair<float, size_t>> best_points;

    for (size_t i = 0; i < layers.back().cluster_num; i++) {
        float result = faiss::fvec_inner_product(
                queries.row(qnum), 
                layers.back().kr.centroids.row(i),
                queries.vector_length);
        queue.push({result, {layers.size() - 1, i}});
    }

    for (size_t i = 0; !queue.empty() && i < opened_trees; i++) {
        auto best = queue.top().second;
        queue.pop();
        
        size_t layer_id = best.first;
        size_t cid = best.second;

        for (auto child: layers[layer_id].centroid_children[cid]) {
            if (layer_id == 0) {
                float result = faiss::fvec_inner_product(
                        queries.row(qnum), 
                        vectors.row(child),
                        queries.vector_length);
                best_points.push_back({result, child});
            }
            else {
                float result = faiss::fvec_inner_product(
                        queries.row(qnum), 
                        layers[layer_id - 1].kr.centroids.row(child),
                        queries.vector_length);
                queue.push({result, {layer_id - 1, child}});
            }
        }
    }
    // Last layer - find best match.
    if (best_points.size() > k_needed) {
        nth_element(
                best_points.begin(), 
                best_points.begin() + k_needed,
                best_points.end(),
                greater<std::pair<float, size_t>>());
        best_points.resize(k_needed);
    }
    sort(best_points.rbegin(), best_points.rend());

    vector<size_t> res;
    for (size_t i = 0; i < best_points.size(); i++) {
        res.push_back(best_points[i].second);
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
        vector<size_t> predictions = predict(layers, queries, i, opened_trees, vectors, k);
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
