#include "alsh.h"
#include "common.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <map>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <omp.h>

#include "../faiss/utils.h"
#include "../faiss/Clustering.h"

using namespace std;
typedef uint64_t hash_t;

int IndexALSH::dot_product_hash(
        const float* a, const float* x, const float b) const {
    float ax = faiss::fvec_inner_product(a, x, d);
    return floor((ax + b) / r);
}

static float randn() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<float> d(0, 1);

    return d(gen);
}

static float uniform(float low, float high) {
    static random_device rd;
    static mt19937 gen(rd());
    static uniform_real_distribution<float> d(low, high);

    return d(gen);
}

static lsh_metahash_t::hash_t combine_hashes(vector<int> values) {
    lsh_metahash_t::hash_t seed = 0;
    for (size_t i = 0; i < values.size(); i++) {
        int v = values[i];
        seed ^= v + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
}

void IndexALSH::hash_vectors(FloatMatrix &data) {
#pragma omp parallel for
    for (size_t l = 0; l < L; l++) {
        for (size_t i = 0; i < data.vector_count(); i++) {
            metahashes[l].table[calculate_metahash(l, data.row(i))].insert(i);
        }
    }
}

lsh_metahash_t::hash_t IndexALSH::calculate_metahash(size_t l, const float* data) const {
    vector<int> hash_vector(K);
    for (size_t k = 0; k < K; k++) {
        hash_vector[k] = dot_product_hash(
            metahashes[l].hashes[k].a.data(),
            data,
            metahashes[l].hashes[k].b);
    }
    return combine_hashes(hash_vector);
}

vector<faiss::Index::idx_t> IndexALSH::answer_query(float *query, size_t k_needed) const {
    unordered_set<idx_t> colliders;
    for (size_t l = 0; l < L; l++) {
        const auto it = metahashes[l].table.find(calculate_metahash(l, query));
        if (it != metahashes[l].table.end()) {
            // Insert every colliding vector to set.
            for (const auto vec_id: it->second) {
                colliders.insert(vec_id);
            }
        }
    }
    // Calculate inner product of every collider with query.
    vector<pair<float, idx_t>> score_vector(colliders.size());
    size_t i = 0;
    for (auto ind: colliders) {
        float result = faiss::fvec_inner_product(
                data_matrix.row(ind),
                query,
                data_matrix.vector_length);
        score_vector[i++] = {-result, ind};
    }
    if (score_vector.size() > k_needed) {
        nth_element(
            score_vector.begin(), 
            score_vector.begin() + k_needed, 
            score_vector.end());
        score_vector.resize(k_needed);
    }
    sort(score_vector.begin(), score_vector.end());
    vector<idx_t> res(score_vector.size());
    for (size_t i = 0; i < score_vector.size(); i++) {
        res[i] = score_vector[i].second;
    }
    return res;
}

IndexALSH::IndexALSH(
        size_t dim, size_t L, size_t K, float r, MipsAugmentation* aug):
        Index(dim, faiss::METRIC_INNER_PRODUCT),
        L(L), K(K), r(r), augmentation(aug) {

    // Initialize metahashes' coefficients.
    metahashes.resize(L);
    for (size_t l = 0; l < L; l++) {
        metahashes[l].hashes.resize(K);
        for (size_t k = 0; k < K; k++) {
            for (size_t i = 0; i < d + aug->m; i++) {
                metahashes[l].hashes[k].a.push_back(randn());
            }
            metahashes[l].hashes[k].b = uniform(0, r);
        }
    }
}

void IndexALSH::reset() {
    for (size_t l = 0; l < L; l++) {
        metahashes[l].table.clear();
    }
}

void IndexALSH::add(idx_t n, const float* data) {
    data_matrix = augmentation->extend(data, n);
    hash_vectors(data_matrix);
}

void IndexALSH::search(
        idx_t n, const float* data, idx_t k,
           float* distances, idx_t* labels) const {

    FloatMatrix queries = augmentation->extend_queries(data, n);
    #pragma omp parallel for
    for (size_t q = 0; q < queries.vector_count(); q++) {
        vector<idx_t> ans = answer_query(queries.row(q), k);
        for (idx_t j = 0; j < k; j++) {
            idx_t lab = (size_t(j) < ans.size()) ? ans[j] : -1;
            labels[q * k + j] = lab;
        }
    }
}

void IndexALSH::save(const char* fname) const {
    FILE* f = fopen(fname, "wb");
    write_floatmatrix(data_matrix, f); // 1
    size_t s = metahashes.size();
    fwrite(&s, sizeof(s), 1, f); // 2
    for (const auto& i: metahashes) {
        s = i.hashes.size();
        fwrite(&s, sizeof(s), 1, f); // 3
        for (const auto& j: i.hashes) {
            write_vec(j.a, f); // 4
            fwrite(&(j.b), sizeof(float), 1, f); // 5
        }
        s = i.table.size();
        fwrite(&s, sizeof(s), 1, f); // 6
        for (const auto& j: i.table) {
            fwrite(&(j.first), sizeof(hash_t), 1, f); // 7
            s = j.second.size();
            fwrite(&s, sizeof(s), 1, f); // 8
            for (const auto& k: j.second) {
                fwrite(&k, sizeof(faiss::Index::idx_t), 1, f); // 9
            }
        }
    }
    fclose(f);
}

void IndexALSH::load(const char* fname) {
    FILE* f = fopen(fname, "rb");
    read_floatmatrix(data_matrix, f); // 1
    size_t s;
    fread(&s, sizeof(s), 1, f);  // 2
    metahashes.resize(s);
    for (auto& i: metahashes) {
        fread(&s, sizeof(s), 1, f); // 3
        i.hashes.resize(s);
        for (auto& j: i.hashes) {
            read_vec(j.a, f); // 4
            fread(&(j.b), sizeof(float), 1, f); // 5
        }
        fread(&s, sizeof(s), 1, f); // 6
        for (size_t j = 0; j < s; j++) {
            hash_t key;
            fread(&key, sizeof(hash_t), 1, f); // 7
            size_t set_s;
            fread(&set_s, sizeof(set_s), 1, f); // 8
            for (size_t k = 0; k < set_s; k++) {
                faiss::Index::idx_t elem;
                fread(&elem, sizeof(elem), 1, f); // 9
                i.table[key].insert(elem);
            }
        }
    }
    fclose(f);
}
