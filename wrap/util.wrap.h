#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include "../faiss/Index.h"

namespace py = pybind11;
using namespace py::literals;

#define WRAP_INDEX_HELPER(ClassName, obj) {                                                                            \
                                                                                                                       \
    obj.def("train",                                                                                                   \
            [](ClassName& self, py::array_t<float, py::array::c_style | py::array::forcecast> data) {                  \
                auto n = data.request().shape[0];                                                                      \
                auto data_ptr = (float*) data.request().ptr;                                                           \
                self.train(n, data_ptr);                                                                               \
            },                                                                                                         \
            "data"_a,                                                                                                  \
            "Train the Index");                                                                                        \
                                                                                                                       \
    obj.def("add",                                                                                                     \
            [](ClassName& self, py::array_t<float, py::array::c_style | py::array::forcecast> data) {                  \
                auto n = data.request().shape[0];                                                                      \
                auto data_ptr = (float*) data.request().ptr;                                                           \
                self.add(n, data_ptr);                                                                                 \
            },                                                                                                         \
            "data"_a,                                                                                                  \
            "Add database vectors to this index");                                                                     \
                                                                                                                       \
    obj.def("search",                                                                                                  \
            [](ClassName& self, py::array_t<float, py::array::c_style | py::array::forcecast> data, long k) {          \
                auto n = data.request().shape[0];                                                                      \
                                                                                                                       \
                py::array_t<float, py::array::c_style | py::array::forcecast> distances(n * k);                            \
                py::array_t<long,  py::array::c_style | py::array::forcecast> labels(n * k);                               \
                                                                                                                       \
                auto data_ptr      = (float*) data.request().ptr;                                                      \
                auto distances_ptr = (float*) distances.request().ptr;                                                 \
                auto labels_ptr    = (long*)  labels.request().ptr;                                                    \
                                                                                                                       \
                self.search(n, data_ptr, k, distances_ptr, labels_ptr);                                                \
                return std::make_tuple(distances, labels);                                                             \
            },                                                                                                         \
            "data"_a, "k"_a,                                                                                           \
            "Search data vectors for k closest vectors stored in database");                                           \
                                                                                                                       \
    obj.def("reset",                                                                                                   \
            &ClassName::reset,                                                                                         \
            "Reset the state of the Index");                                                                           \
}
