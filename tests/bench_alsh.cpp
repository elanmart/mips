#include "../src/common.h"
#include "../src/bench.h"
#include "../src/alsh.h"

size_t m = 3; // additional vector dimensions
int augtype; // augmentation type
size_t L; // number of hash tables
size_t K; // number of hash functions in one hash table
float r; // hash function parameter
float U; // vector scaling coefficient

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    MipsAugmentation* aug;
    size_t dim = xt.vector_length;
    switch (augtype) {
    case 0: aug = new MipsAugmentationNeyshabur(dim); break;
    case 1: aug = new MipsAugmentationShrivastava(dim, m, U); break;
    case 2: aug = new MipsAugmentationNone(dim); break;
    default: exit(1);
    }
    faiss::Index* index = new IndexALSH(dim, L, K, r, aug);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Arguments missing, terminating.\n");
    } else {
        L = atoi(argv[1]);
        K = atoi(argv[2]);
        sscanf(argv[3], "%f", &r);
        augtype = atoi(argv[4]);
        sscanf(argv[5], "%f", &U);
        
        faiss::Index* index = bench_train(get_trained_index);
        bench_add(index);
        bench_query(index);
    }
}
