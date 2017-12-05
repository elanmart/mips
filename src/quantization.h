#include "common.h"

#include "../faiss/Index.h"


struct IndexSubspaceQuantization: public faiss::Index {
    IndexSubspaceQuantization(size_t dim, size_t subspace_count, size_t centroid_count);
    void add(idx_t n, const float* data);
    void search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const;
    void reset();
    void train(idx_t, const float*) {}; // For Python bindings.
    
    std::vector<kmeans_result> kmeans;
    std::vector<size_t> permutation;
    // Parameters:
    size_t subspace_count, centroid_count;
};
