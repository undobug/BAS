# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import torch

class Recorder():
    def __init__(self,dtype=None,device=torch.device('cpu')):
        self.dtype = dtype
        self.device = device
        self.reset()
    def reset(self):
        self._data = torch.tensor([], dtype=self.dtype, device=self.device)
        self.count = torch.tensor(0, dtype=torch.int, device=self.device)

    @torch.no_grad()
    def record(self, new_data):
        self._data = torch.cat((self._data, new_data), dim=0)
        self.count += 1

    @property
    def data(self):
        return self._data
        

def distributed_concat(tensor):
    output_tensors = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat
