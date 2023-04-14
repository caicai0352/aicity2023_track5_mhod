# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

# based on:
# https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py


import itertools
from typing import Optional, List, Callable

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

from fastreid.utils import comm


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        data_source: a list of data items
        size: number of samples to draw
    """

    def __init__(self, data_source: List, size: int = None, seed: Optional[int] = None,
                 callback_get_label: Callable = None):
        self.data_source = data_source
        # consider all elements in the dataset
        self.indices = list(range(len(data_source)))
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self._size = len(self.indices) if size is None else size
        self.callback_get_label = callback_get_label
        label_to_count=np.zeros(94)
        # distribution of classes in the dataset
        for _,label in self.data_source:
        
            label_to_count +=label
        # print(label_to_count)
        weights=[]
        for idx in self.indices:
            label_to_count_sum=0
            for i,label in enumerate( self._get_label(data_source, idx)):
                if label:
                    label_to_count_sum+=label_to_count[i]
            weights.append(1.0/label_to_count_sum)
        # print(weights[1],len(weights))
        
            


        # weight for each sample
        # weights = [1.0 / label_to_count[self._get_label(data_source, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

    def _get_label(self, dataset, idx):
        if self.callback_get_label:
            return self.callback_get_label(dataset, idx)
        else:
            return dataset[idx][1]

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        while True:
            for i in torch.multinomial(self.weights, self._size, replacement=True):
                yield self.indices[i]
