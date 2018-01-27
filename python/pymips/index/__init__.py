from ._pymips import MipsAugmentationShrivastava, IndexHierarchicKmeans
import faiss
from typing import Any


class IndexBase:

    index = NotImplemented  # type: Any

    def add(self, data):
        self.index.add(data)

    def train(self, data):
        self.index.train(data)

    def reset(self):
        self.index.reset()

    def search(self, data, k, *args, **kwargs):
        return self._search(data, k)

    def _search(self, data, k):

        D, I = self.index.search(data, k)
        D, I = D.reshape(-1, k), I.reshape(-1, k)

        return D, I


class IVFIndex(IndexBase):
    def __init__(self, d, size, nprobe):
        self.index = faiss.index_factory(d, f"IVF{size},Flat", faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = nprobe


class FlatIndex(IndexBase):
    def __init__(self, d):
        self.index = faiss.IndexFlatIP(d)


class KMeansIndex(IndexBase):
    def __init__(self, d, layers, nprobe, m, U, bnb, spherical):
        self.aug   = MipsAugmentationShrivastava(d, m, U)
        self.index = IndexHierarchicKmeans(d, layers, nprobe, self.aug, bnb, spherical)

    def search(self, data, k, nprobe):
        self.index.set_opened_trees(nprobe)
        return self._search(data, k)

