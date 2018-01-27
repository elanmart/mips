import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.utils import check_array

import faiss


def _default_index(d):
    index = faiss.index_factory(d, "IVF2048,Flat", faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 256

    return index


class ApproximateClassifierMixin(LinearClassifierMixin):

    def decision_function(self, X):
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        self._train_index()

        X = check_array(X, accept_sparse=False)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        D, I = self.index_.search(X.astype(np.float32), 1)
        return D, I

    def _train_index(self):

        if not hasattr(self, 'index_'):
            self.index_ = _default_index(self.coef_.shape[1])
            self.coef_ = np.ascontiguousarray(self.coef_, dtype=np.float32)

            self.index_.train(self.coef_)
            self.index_.add(self.coef_)

        return self


def fast(cls):
    assert LinearClassifierMixin in cls.mro(), "Can only speed up linear classifiers"
    return type(cls.__name__, (ApproximateClassifierMixin,) + cls.__bases__, dict(cls.__dict__))