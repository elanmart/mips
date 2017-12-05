#include "../src/common.h"
#include "../src/bench.h"
#include "../src/kmeans.h"

// Arguments (and usual values):
// 1. Layer count (1 - 4)
// 2. Augmentation type (0 - 2, see switch below)
// 3. U, vector scaling coefficient (0.83 as per paper, or anything if unused)
// 4. Use branch and bound? (0 or 1)
// 5+. Number of opened trees, a.k.a. nprobe (50 - 500)
// 
// First number of opened treees will be used for training,
// deciding the exact size of layers. This does not forbid you from
// querying using other value of this parameter - it might be less
// efficient though.

size_t m = 3; // additional vector dimensions
float U; // vector scaling coefficient
size_t layers_count;
size_t opened_trees;
int bb;
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
    faiss::Index* index = new IndexHierarchicKmeans(dim, layers_count, opened_trees, aug, bb);
    index->train(xt.vector_count(), xt.data.data());
    return index;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        printf("Arguments missing, terminating.\n");
    } else {
        layers_count = atoi(argv[1]);
        augtype = atoi(argv[2]);
        sscanf(argv[3], "%f", &U);
        bb = atoi(argv[4]);
        opened_trees = atoi(argv[5]);

        faiss::Index* index = bench_train(get_trained_index);

        if (1) {
            bench_add(index);
            ((IndexHierarchicKmeans*) index)->save("/tmp/saved_1");
        }
        else {
            ((IndexHierarchicKmeans*) index)->load("/tmp/saved_1");
        }

        for (int i = 5; i < argc; i++) {
            printf("Querying using opened_trees = %d\n", atoi(argv[i]));
            ((IndexHierarchicKmeans*) index)->opened_trees = atoi(argv[i]);
            bench_query(index);
        }
    }
}
