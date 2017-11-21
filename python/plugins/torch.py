import torch as th
from torch import nn

import faiss


class ApproximateLinear(nn.Module):
    return_type = namedtuple('TopK', ['distances', 'indices'])

    def __init__(self, in_features, out_features, bias=False, train_on_eval=True,
                 index_factory=None):

        if bias:
            warnings.warn("bias argument is ignored in ApproximateLinear layer")

        super().__init__(in_features=in_features, out_features=out_features, bias=False)

        self.train_on_eval = train_on_eval

        self.index_factory = index_factory or self._default_index()
        self.index = self.index_factory(self.in_features)

        self._cuda_resource = None

    def _default_index(self, d)
        index = faiss.index_factory(d, "IVF4096,Flat")
        index.nprobe = 256

        return index

    def cuda(self):
        super().cuda()

        device_no = th.cuda.current_device()

        if self._cuda_resource is None:
            self._cuda_resource = faiss.StandardGpuResources()

        self.index = faiss.index_cpu_to_gpu(self._cuda_resource, device_no, self.index)

        return self

    def cpu(self):
        super().cpu()
        self.index = faiss.index_gpu_to_cpu(self.index)

        return self

    def reset_parameters(self):
        super().reset_parameters()
        self.index.reset()

        return self

    def train(self, flag=True):
        super().train(flag)

        if not flag:
            self.index.train(self.weight)
            self.index.add(self.weight)

        return self

    def forward(self, x, k=None):
        if self.training or k is None:
            D, I = super().forward(x), None
        else:
            D, I = self.index.search(x)

        return ApproximateLinear.return_type(D, I)
