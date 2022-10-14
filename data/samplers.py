# --------------------------------------------------------
#!/usr/bin/env python
# Licensed under The MIT License [see LICENSE for details]
# Written by fyb
# --------------------------------------------------------


import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import TypeVar, Optional, Iterator
from collections import defaultdict, namedtuple


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


T_co = TypeVar('T_co', covariant=True)
class DistributedSpeakerSampler(torch.utils.data.sampler.Sampler[T_co]):
    def __init__(self, batch_size, dataset_domains: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self._dataset_domains = dataset_domains
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size
        self._class_to_samples = defaultdict(set)
        self.num_ = 30
        for i, c in enumerate(self._dataset_domains):
            self._class_to_samples[c].add(i)              # eg:{0:{1,2,3,4}, 2:{5,6,7}, ..., c:{i-2,i-1,i}}   c:class_index,i:sample_index
        self.spklist = list(key for key in self._class_to_samples.keys())
    def __iter__(self) -> Iterator[T_co]:
        for i in range(self.num_):
            #choice one speaker
            #print(self.spklist)
            spk_classes = self.spklist[i%len(self.spklist)]
            #print(i,spk_classes)
            batch_samples=[]

            mm = np.random.randint(1,1000000)
            random.seed(mm)

            class_samples = random.sample(
                    self._class_to_samples[spk_classes], self.batch_size)

            for sample in class_samples:
                batch_samples.append(sample)

            random.shuffle(batch_samples)
            for sample in batch_samples:
                yield sample

    def __len__(self) -> int:
        return self.num_

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch