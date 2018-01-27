import warnings

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import faiss


def _as_numpy(var):
    return np.copy(np.ascontiguousarray(var.data.cpu().numpy()))


def _default_index(d):
    index = faiss.index_factory(d, "IVF2048,Flat", faiss.METRIC_INNER_PRODUCT)
    index.nprobe = 256

    return index


class ApproximateLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, train_on_eval=True,
                 index_factory=_default_index):

        if bias:
            warnings.warn("bias argument is ignored in ApproximateLinear layer")

        self.train_on_eval = train_on_eval
        self.index = index_factory(in_features)
        self._cuda_resource = None

        super().__init__(in_features=in_features, out_features=out_features, bias=False)

    def cuda(self, device=None):
        super().cuda(device=device)

        if self._cuda_resource is None:
            self._cuda_resource = faiss.StandardGpuResources()

        self.index = faiss.index_cpu_to_gpu(self._cuda_resource, device, self.index)

        return self

    def cpu(self):
        super().cpu()
        self.index = faiss.index_gpu_to_cpu(self.index)

        return self

    def reset_parameters(self):
        super().reset_parameters()
        self.index.reset()

        return self

    def train(self, mode=True):
        super().train(mode)

        if (mode is False) and self.train_on_eval:
            w = _as_numpy(self.weight)
            self.index.train(w)
            self.index.add(w)

        return self

    def forward(self, x, k=1):

        if self.training or k is None:
            return super().forward(x)

        else:
            D, I = self.index.search(_as_numpy(x), k)
            D, I = Variable(torch.from_numpy(D).float()), Variable(torch.from_numpy(I).long())
            return D, I
