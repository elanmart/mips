#include "common.h"

#include "../faiss/Index.h"

struct IndexHierarchicKmeans: public faiss::Index {
    struct range {
        size_t left, right;
    };
    struct layer_t {
        std::vector<range> children_range;
        FloatMatrix points;
        std::vector<float> dist_to_parent;
    };

    IndexHierarchicKmeans(size_t dim, size_t layers_count, size_t opened_trees,
               MipsAugmentation* aug, bool branch_n_bound);
    void add(idx_t n, const float* data);
    void search(idx_t n, const float* data, idx_t k, float* distances, idx_t* labels) const;
    void reset();
    void train(idx_t, const float*) {}; // For Python bindings.
    
    void save(const char* fname) const;
    void load(const char* fname);

    FloatMatrix vectors_original;
    std::vector<layer_t> layers;

    // Parameters:
    size_t layers_count;
    size_t opened_trees;
    bool branch_n_bound;
    MipsAugmentation* augmentation;
};
