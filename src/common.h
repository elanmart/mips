#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <cstdlib>
#include <string>


template <typename T>
struct FlatMatrix {
    std::vector<T> data;
    size_t vector_length;

    T& at(size_t vec, size_t ind);
    const T& at(size_t vec, size_t ind) const;

    T* row(size_t num);
    const T* row(size_t num) const;

    size_t vector_count() const;
    void print() const;
    void resize(size_t cnt, size_t dim);
};

typedef FlatMatrix<float> FloatMatrix;

template <typename T>
FlatMatrix<T> load_text_file(std::string filename);

template<typename T>
FlatMatrix<T> load_vecs (std::string filename);

struct kmeans_result {
    FloatMatrix centroids;
    std::vector<size_t> assignments;
};

kmeans_result perform_kmeans(const FloatMatrix& matrix, size_t k, bool spherical);


void scale(float* vec, float alpha, size_t size);


struct MipsAugmentation {
    MipsAugmentation(size_t dim, size_t m);
    virtual FloatMatrix extend(const float* data, size_t nvecs) = 0;
    virtual FloatMatrix extend_queries(const float* data, size_t nvecs) = 0;
    size_t dim;
       size_t m;
};

struct MipsAugmentationShrivastava: public MipsAugmentation {
    MipsAugmentationShrivastava(size_t dim, size_t m, float U = 0.8);
    FloatMatrix extend(const float* data, size_t nvecs);
    FloatMatrix extend_queries(const float* data, size_t nvecs);

    float U;
};

struct MipsAugmentationNeyshabur: public MipsAugmentation {
    MipsAugmentationNeyshabur(size_t dim);
    FloatMatrix extend(const float* data, size_t nvecs);
    FloatMatrix extend_queries(const float* data, size_t nvecs);
};

struct MipsAugmentationNone: public MipsAugmentation {
    MipsAugmentationNone(size_t dim);
    FloatMatrix extend(const float* data, size_t nvecs);
    FloatMatrix extend_queries(const float* data, size_t nvecs);
};


#include "common.inc.h"

#endif
