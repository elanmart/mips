#include "common.h"

#include "../faiss/Clustering.h"
#include "../faiss/IndexFlat.h"
#include "../faiss/utils.h"

#include <algorithm>

kmeans_result perform_kmeans(const FlatMatrix<float>& matrix, size_t k, bool spherical) {
    kmeans_result kr;
    kr.centroids.resize(k, matrix.vector_length);
    kr.assignments.resize(matrix.vector_count());
    std::vector<float> dist(matrix.vector_count());
    std::vector<faiss::Index::idx_t> assignments(matrix.vector_count());

    size_t dim = matrix.vector_length;
    size_t n = matrix.vector_count();

    faiss::ClusteringParameters cp;
    cp.spherical = spherical;
    faiss::Clustering clus (dim, k, cp);
    clus.verbose = dim * n * k > (1ULL << 30);
    // display logs if > 1Gflop per iteration
    faiss::IndexFlatL2 index_ (dim);
    clus.train (n, matrix.data.data(), index_);
    memcpy(kr.centroids.data.data(),
               clus.centroids.data(),
               sizeof(clus.centroids[0]) * dim * k);

    faiss::IndexFlatL2 index(matrix.vector_length);
    index.add(kr.centroids.vector_count(), kr.centroids.data.data());
    index.search(matrix.vector_count(), matrix.data.data(), 1,
            dist.data(), assignments.data());

    for (size_t i = 0; i < matrix.vector_count(); i++) {
        kr.assignments[i] = assignments[i];
    }
    return kr;
}

void scale(float* vec, float alpha, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vec[i] /= alpha;
    }
}

static FloatMatrix shrivastava_extend(const float* data, size_t nvecs, size_t dim, size_t m, float U) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim + m);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(data_matrix.row(i), data + i * dim, dim * sizeof(float));
    }

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm / U, dim);

        float vec_norm = sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim));
        for (size_t j = dim; j < dim + m; j++) {
            data_matrix.at(i, j) = 0.5 - vec_norm;
            vec_norm *= vec_norm;
        }
    }
    return data_matrix;
}

static FloatMatrix generic_extend_queries(const float* data, size_t nvecs, size_t dim, size_t m) {
    FloatMatrix queries;
    queries.resize(nvecs, dim + m);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(queries.row(i), data + i * dim, dim * sizeof(float));
    }
    for (size_t i = 0; i < queries.vector_count(); i++) {
        float qnorm = sqrt(faiss::fvec_norm_L2sqr(queries.row(i), dim));
        scale(queries.row(i), qnorm, dim);

        for (size_t j = dim; j < dim + m; j++) {
            queries.at(i, j) = 0.0;
        }
    }
    return queries;
}

static FloatMatrix neyshabur_extend(const float* data, size_t nvecs, size_t dim) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim + 1);
    for (size_t i = 0; i < nvecs; i++) {
        memcpy(data_matrix.row(i), data + i * dim, dim * sizeof(float));
    }

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm, dim);

        float norm_sqr = faiss::fvec_norm_L2sqr(data_matrix.row(i), dim);
        data_matrix.at(i, dim) = sqrt(1 - norm_sqr);
    }
    return data_matrix;
}

MipsAugmentation::MipsAugmentation(size_t dim, size_t m):
    dim(dim), m(m) {}

MipsAugmentationShrivastava::MipsAugmentationShrivastava(size_t dim, size_t m, float U):
    MipsAugmentation(dim, m), U(U) {}

FloatMatrix MipsAugmentationShrivastava::extend(const float* data, size_t nvecs) {
    return shrivastava_extend(data, nvecs, dim, m, U);
}

FloatMatrix MipsAugmentationShrivastava::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}

MipsAugmentationNeyshabur::MipsAugmentationNeyshabur(size_t dim):
    MipsAugmentation(dim, 1) {}

FloatMatrix MipsAugmentationNeyshabur::extend(const float* data, size_t nvecs) {
    return neyshabur_extend(data, nvecs, dim);
}

FloatMatrix MipsAugmentationNeyshabur::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}

MipsAugmentationNone::MipsAugmentationNone(size_t dim):
    MipsAugmentation(dim, 0) {}

FloatMatrix MipsAugmentationNone::extend(const float* data, size_t nvecs) {
    FloatMatrix data_matrix;
    data_matrix.resize(nvecs, dim);
    memcpy(data_matrix.data.data(), data, nvecs * dim * sizeof(float));

    double maxnorm = 0;
    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        maxnorm = std::max(maxnorm, sqrt(faiss::fvec_norm_L2sqr(data_matrix.row(i), dim)));
    }

    for (size_t i = 0; i < data_matrix.vector_count(); i++) {
        scale(data_matrix.row(i), maxnorm, dim);
    }
    return data_matrix;
}

FloatMatrix MipsAugmentationNone::extend_queries(const float* data, size_t nvecs) {
    return generic_extend_queries(data, nvecs, dim, m);
}

void write_floatmatrix(const FloatMatrix& mat, FILE* f) {
    fwrite(&mat.vector_length, sizeof(mat.vector_length), 1, f);
    auto cnt = mat.vector_count();
    fwrite(&cnt, sizeof(cnt), 1, f);
    fwrite(mat.data.data(), sizeof(mat.data[0]), mat.data.size(), f);
}

void read_floatmatrix(FloatMatrix& mat, FILE* f) {
    size_t len, cnt;
    fread(&len, sizeof(len), 1, f);
    fread(&cnt, sizeof(cnt), 1, f);
    mat.resize(cnt, len);
    fread(mat.data.data(), sizeof(mat.data[0]), mat.data.size(), f);
}
