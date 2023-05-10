import numpy as np

from typing import Tuple

from ..dataset import Jsb16thSeparatedDataset

class IsingModel:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        nodes = np.prod(self.shape)
        self.edges = np.zeros((nodes, nodes, 2), dtype=int)
        self.nodes = np.zeros((nodes, 2), dtype=int)

    def fit(self, dataset: Jsb16thSeparatedDataset):
        for epoch in range(10):
            for i in range(len(dataset)):
                sample = dataset[i]
                assert sample.dtype == bool, "samples must be binary"
                assert sample.shape == self.shape, "all samples must have the same shape"
                self._update(sample)
        self.edges += self.edges.T
        self.edges += 1
    
    def _update(self, sample: np.ndarray):
        sample = sample.reshape(-1).astype(int)
        self.nodes[np.arange(len(sample)), sample] += 1