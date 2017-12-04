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


static vector<layer_t> make_layers(const FloatMatrix& vectors, size_t L, size_t nprobe) {
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

        auto kr = perform_kmeans(layers[layer_id - 1].points, cluster_num);

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
            for (size_t i = layers[layer_id].children_range[cid].left;
                       i < layers[layer_id].children_range[cid].right; i++) {
                
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
        res.push_back(layers[0].children_range[candidates[i].second].left);
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
    auto vectors = augmentation->extend(data, n);
    layers = make_layers(vectors, layers_count, opened_trees);
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
        vector<size_t> predictions = predict(layers, queries, i, opened_trees, k);
        for (idx_t j = 0; j < k; j++) {
            labels[i*k + j] = (size_t(j) < predictions.size()) ? predictions[j] : -1;
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

static void write_floatmatrix(const FloatMatrix& mat, FILE* f) {
    fwrite(&mat.vector_length, sizeof(mat.vector_length), 1, f);
    auto cnt = mat.vector_count();
    fwrite(&cnt, sizeof(cnt), 1, f);
    fwrite(mat.data.data(), sizeof(mat.data[0]), mat.data.size(), f);
}

void IndexHierarchicKmeans::save(const char* fname) const {
    FILE* f = fopen(fname, "wb");
    write_floatmatrix(vectors_original, f);
    for (const auto& l: layers) {
        size_t cnt = l.children_range.size();
        fwrite(&cnt, sizeof(cnt), 1, f);
        fwrite(l.children_range.data(), sizeof(range), l.children_range.size(), f);
        write_floatmatrix(l.points, f);
    }
    fclose(f);
}


static void read_floatmatrix(FloatMatrix& mat, FILE* f) {
    size_t len, cnt;
    fread(&len, sizeof(len), 1, f);
    fread(&cnt, sizeof(cnt), 1, f);
    mat.resize(cnt, len);
    fread(mat.data.data(), sizeof(mat.data[0]), mat.data.size(), f);
}

void IndexHierarchicKmeans::load(const char* fname) {
    FILE* f = fopen(fname, "rb");
    // Assume parameters are already there (i.e. index was constructed using
    // provided constructor).
    read_floatmatrix(vectors_original, f);
    layers.resize(layers_count + 1);
    for (auto& l: layers) {
        size_t cnt;
        fread(&cnt, sizeof(cnt), 1, f);
        l.children_range.resize(cnt);
        fread(l.children_range.data(), sizeof(range), l.children_range.size(), f);
        read_floatmatrix(l.points, f);
    }
    fclose(f);
}
