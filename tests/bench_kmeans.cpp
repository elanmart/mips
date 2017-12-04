#include "../src/common.h"
#include "../src/bench.h"
#include "../src/kmeans.h"

// arguments guide (OT = opened_trees)
//argc=  0            1         2     3     4     5
// bench_kmeans layers_count augtype 
//                              1     U    OT1   OT2  ...
//                             0/2   -1    OT1   OT2  ...
// First value of OT will be used for training.

size_t m = 3; // additional vector dimensions
float U; // vector scaling coefficient
size_t layers_count;
size_t opened_trees;
int augtype; // augmentation type

faiss::Index* get_trained_index(const FloatMatrix& xt) {
    MipsAugmentation* aug;
    size_t dim = xt.vector_length;
    switch (augtype) {
    case 0: aug = new MipsAugmentationNeyshabur(dim); break;
    case 1: aug = new MipsAugmentationShrivastava(dim, m, U); break;
    case 2: aug = new MipsAugmentationNone(dim); break;
    default: exit(1);
    }
    faiss::Index* index = new IndexHierarchicKmeans(dim, layers_count, opened_trees, aug);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        printf("Arguments missing, terminating.\n");
    } else {
        layers_count = atoi(argv[1]);
        augtype = atoi(argv[2]);
        sscanf(argv[3], "%f", &U);
        opened_trees = atoi(argv[4]);

        faiss::Index* index = bench_train(get_trained_index);

        if (1) {
            bench_add(index);
            ((IndexHierarchicKmeans*) index)->save("/tmp/saved");
        }
        else {
            ((IndexHierarchicKmeans*) index)->load("/tmp/saved");
        }

        for (int i = 4; i < argc; i++) {
            printf("Querying using opened_trees = %d\n", atoi(argv[i]));
            ((IndexHierarchicKmeans*) index)->opened_trees = atoi(argv[i]);
            bench_query(index);
        }
    }
}
