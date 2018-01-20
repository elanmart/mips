#include "../src/common.h"
#include "../src/quantization.h"
#include "../src/kmeans.h"
#include "../src/alsh.h"
#include "util.wrap.h"

namespace py = pybind11;


PYBIND11_MODULE(mips, m) {
    m.doc() = "MIPS library";


    // MipsAugmentationShrivastava
    py::class_<MipsAugmentationShrivastava> msrvaug(m, "MipsAugmentationShrivastava");
    msrvaug.def(
        py::init<size_t, size_t, float>(),
        "MIPS Augmentation scheme",
        py::arg("dim"), py::arg("m"), py::arg("U")
    );


    // IndexHierarchicKmeans
    py::class_<IndexHierarchicKmeans> kmns(m, "IndexHierarchicKmeans");
    kmns.def(
        py::init([](size_t dim, size_t layers_count, size_t opened_trees, 
                    MipsAugmentationShrivastava& augmentation, 
                    bool branch_n_bound, bool spherical) {
                        return std::unique_ptr<IndexHierarchicKmeans>(
                            new IndexHierarchicKmeans(dim,
                                                      layers_count,
                                                      opened_trees,
                                                      &augmentation, 
                                                      branch_n_bound,
                                                      spherical));
                    }),
        "IndexHierarchicKmeans",
        py::arg("dim"), py::arg("layers_count"), 
        py::arg("opened_trees"), py::arg("augmentation"), 
        py::arg("branch_n_bound"), py::arg("spherical")
    );
    kmns.def("set_opened_trees", &IndexHierarchicKmeans::set_opened_trees);
    WRAP_INDEX_HELPER(IndexHierarchicKmeans, kmns);


    // IndexALSH
    py::class_<IndexALSH> alsh(m, "IndexALSH");
    alsh.def(
        py::init([](size_t dim, size_t L, 
                    size_t K, size_t r, 
                    MipsAugmentationShrivastava& augmentation) {
                        return std::unique_ptr<IndexALSH>(
                            new IndexALSH(dim,
                                          L,
                                          K,
                                          r,
                                          &augmentation));
                    }),
        "Asymmetric LSH",
        py::arg("dim"), py::arg("L"), py::arg("K"), py::arg("r"), py::arg("augmentation")
    );
    WRAP_INDEX_HELPER(IndexALSH, alsh);


    // IndexSubspaceQuantization(size_t dim, size_t subspace_count, size_t centroid_count);
    py::class_<IndexSubspaceQuantization> qnt(m, "IndexSubspaceQuantization");
    qnt.def(
        py::init<size_t, size_t, size_t>(),
        "IndexSubspaceQuantization",
        py::arg("dim"), py::arg("subspace_count"), py::arg("centroid_count")
    );
    WRAP_INDEX_HELPER(IndexSubspaceQuantization, qnt);
    
}

/*
    // IndexHierarchicKmeans
    py::class_<IndexHierarchicKmeans> hkm(m, "IndexHierarchicKmeans");
    hkm.def(
        py::init<size_t, size_t, size_t, size_t, MipsAugmentation>(),
        "Hierarchichal K-Means",
        py::arg("dim"), py::arg("m"), py::arg("layers_count"), py::arg("opened_trees"), py::arg("augmentation")
    );
    WRAP_INDEX_HELPER(IndexHierarchicKmeans, hkm);

    // QUANTIZATION ----------------------------------------------------------------------------------------------------
    py::class_<IndexSubspaceQuantization> sq(m, "IndexSQ");
    WRAP_INDEX_HELPER(IndexSubspaceQuantization, sq);
*/
