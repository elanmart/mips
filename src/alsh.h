#include "common.h"

#include "faiss/Index.h"
#include <map>
#include <set>


struct lsh_hash_t {
    std::vector<float> a;
    float b;
};

struct lsh_metahash_t {
    typedef unsigned long long hash_t;

    std::vector<lsh_hash_t> hashes;
    std::map<hash_t, std::set<faiss::Index::idx_t> > table;
};

struct IndexALSH: public faiss::Index {
    IndexALSH(size_t dim, size_t L, size_t K, float r, MipsAugmentation* aug);
    void add(idx_t n, const float* data);
    void search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const;
    void reset();
    void train(idx_t, const float*) {}; // For Python bindings.

    std::vector<lsh_metahash_t> metahashes;
    FloatMatrix data_matrix;

    
    // Parameters:
    size_t L;
    size_t K;
    float r;

    MipsAugmentation* augmentation;

    void hash_vectors(FloatMatrix& data);
    int dot_product_hash(const float* a, const float* x, const float b) const;
    std::vector<idx_t> answer_query(float *query, size_t k_needed = 1) const;
    lsh_metahash_t::hash_t calculate_metahash(size_t l, const float* data) const;
};
