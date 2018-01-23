import numpy as np
from sklearn.base import ClassifierMixin

import faiss


class LinearClassifierMixin(ClassifierMixin):
    def decision_function(self, X):
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        X = check_array(X, accept_sparse = False)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = self.index.search(X)
        return scores.ravel() if scores.shape[1] == 1 else scores

    def _predict_proba_lr(self, X):

        prob = self.decision_function(X)
        prob *= -1
        np.exp(prob, prob)
        prob += 1
        np.reciprocal(prob, prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
    
    def train_internal_index(self):
        self.index = self._default_index(self.coef_.shape[1])
        w = self.coef_
        self.index.train(np.ascontiguousarray(w, dtype=np.float32))
        self.index.add(np.ascontiguousarray(w, dtype=np.float32))
        return self    
        
    def _default_index(self, d):
        index = faiss.index_factory(d, "IVF512,Flat", faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 256        
        return index


def fast(cls):
    assert ClassifierMixin in cls.__bases__, "Can only speed up linear classifiers"
    return type(cls.__name__, (LinearClassifierMixin, ) + cls.__bases__, dict(cls.__dict__))