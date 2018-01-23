from collections import namedtuple

import numpy as np
import torch as th
from torch import nn

import faiss


class ApproximateLinear(nn.Linear):
    return_type = namedtuple('TopK', ['distances', 'indices'])
    
    def __init__(self, in_features, out_features, bias=False, train_on_eval=True,
                 index_factory=None):
        
        if bias:
            warnings.warn("bias argument is ignored in ApproximateLinear layer")
            
        self.train_on_eval  = train_on_eval
        self._cuda_resource = None
        self.index          = None        
        
        self.index_factory = index_factory or self._default_index
        self.index = self.index_factory(in_features)
        
        super().__init__(in_features=in_features, out_features=out_features, bias=False)
        
    def _default_index(self, d):
        index = faiss.index_factory(d, "IVF512,Flat", faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 256
        
        return index
    
    def _as_np(self, x):
        return np.copy(np.ascontiguousarray(x.data.cpu().numpy()))
        
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
        
    def train(self, mode=True):
        super().train(mode)
        
        if (mode is False) and self.train_on_eval:
            w = self._as_np(self.weight)
            self.index.train(w)
            self.index.add(w)
            
        return self
            
    def forward(self, x, k=None):
        if self.training or k is None:
            D, I = super().forward(x), None
        else:
            D, I = self.index.search(self._as_np(x), k)
        
        return ApproximateLinear.return_type(D, I)
