#include "quantization.h"

#include "common.h"

#include "../faiss/utils.h"
#include "../faiss/Clustering.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm> 
#include <numeric>  

using namespace std;

static vector<size_t> prepare_permutation(size_t m) {
    // TODO: Don't shuffle, but rotate randomly.
    // TODO: Also, probably use some better source of randomness.
    vector<size_t> permutation;
    for (size_t i = 0; i < m; i++) {
        permutation.push_back(i);
    }
    random_shuffle(permutation.begin(), permutation.end());
    return permutation;
}

static void apply_permutation(float* vec, vector<size_t> permutation) {
    for (size_t i = 0; i < permutation.size(); i++) {
        size_t current = i;
        while (i != permutation[current]) {
            int next = permutation[current];
            swap(vec[current], vec[next]);
            permutation[current] = current;
            current = next;
        }
        permutation[current] = current;
    }
}

static vector<FloatMatrix> make_parts(const FloatMatrix& data, size_t parts_count) {
    vector<FloatMatrix> result(parts_count);
    // Ceil division.
    size_t len = (data.vector_length + parts_count - 1) / parts_count;
    for (size_t i = 0; i < parts_count; i++) {
        if (i != parts_count - 1 || data.vector_length % parts_count == 0) {
            result[i].resize(data.vector_count(), len);
        }
        else {
            // If vector length is not a multiple of parts_count, the last part
            // will have less elements.
            result[i].resize(data.vector_count(), data.vector_length % len);
        }
    }
    for (size_t vec = 0; vec < data.vector_count(); vec++) {
        for (size_t ind = 0; ind < data.vector_length; ind++) {
            result[ind / len].at(vec, ind % len) = data.at(vec, ind);
        }
    }
    return result;
}

// Returns best guess of index of vector closest to query.
static vector<size_t> answer_query(
        const vector<kmeans_result>& kmeans,
           const vector<FloatMatrix>& queries,
           size_t query_number,
           size_t k_needed = 1) {

    assert(kmeans.size() == queries.size());
    assert(kmeans.size() > 0);
    size_t part_count = kmeans.size();
    size_t centroid_count = kmeans[0].centroids.vector_count();
    size_t vector_count = kmeans[0].assignments.size();

    FloatMatrix table;
    table.resize(part_count, centroid_count);
    for (size_t part = 0; part < part_count; part++) {
        size_t part_length = kmeans[part].centroids.vector_length;
        assert(part_length == queries[part].vector_length);

        for (size_t j = 0; j < centroid_count; j++) {
            float product = faiss::fvec_inner_product(
                    kmeans[part].centroids.row(j),
                    queries[part].row(query_number),
                    part_length);
            table.at(part, j) = product;
        }
    }

    vector<pair<float, faiss::Index::idx_t>> results;
    for (size_t vec = 0; vec < vector_count; vec++) {
        float sum = 0;
        for (size_t part = 0; part < part_count; part++) {
            sum += table.at(part, kmeans[part].assignments[vec]);
        }
        results.emplace_back(sum, vec);
    }

    if (results.size() > k_needed) {
        nth_element(results.begin(), results.begin() + k_needed, results.end(),
                greater<pair<float, faiss::Index::idx_t>>());
        results.resize(k_needed);
    }
    sort(results.rbegin(), results.rend());
    vector<size_t> ret(results.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = results[i].second;
    }

    return ret;
}

IndexSubspaceQuantization::IndexSubspaceQuantization(
        size_t dim, size_t subspace_count, size_t centroid_count):
    Index(dim), subspace_count(subspace_count), centroid_count(centroid_count) {
    
    permutation = prepare_permutation(dim);
}

void IndexSubspaceQuantization::add(idx_t n, const float* data) {
    FloatMatrix data_matrix;
    data_matrix.resize(n, d);

    memcpy(data_matrix.data.data(), data, n * d * sizeof(float));
    for (idx_t i = 0; i < n; i++) {
        apply_permutation(data_matrix.row(i), permutation);
    }

    vector<FloatMatrix> parts = make_parts(data_matrix, subspace_count);

    kmeans.resize(subspace_count);
    for(size_t i = 0; i < subspace_count; i++) {
        cout << "Clustering for part " << i << endl;
        kmeans[i] = perform_kmeans(parts[i], centroid_count, false);
    }
}

void IndexSubspaceQuantization::reset() {
    kmeans.clear();
    permutation.clear();
}

void IndexSubspaceQuantization::search(idx_t n, const float* data, idx_t k,
           float* distances, idx_t* labels) const { 

    FloatMatrix queries;
    queries.resize(n, d);
    memcpy(queries.data.data(), data, n * d * sizeof(float));
    for (idx_t i = 0; i < n; i++) {
        apply_permutation(queries.row(i), permutation);
    }

    vector<FloatMatrix> query_parts = make_parts(queries, subspace_count);
    #pragma omp parallel for
    for (size_t q = 0; q < queries.vector_count(); q++) {
        vector<size_t> ans = answer_query(kmeans, query_parts, q, k);
        for (size_t i = 0; i < size_t(k); i++) {
            idx_t lab = (i < ans.size()) ? ans[i] : -1;
            labels[q * k + i] = lab;
            // TODO write distances...
        }
    }

}
