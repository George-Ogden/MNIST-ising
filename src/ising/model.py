from tqdm import trange
import numpy as np

from typing import Tuple

from ..dataset import Jsb16thSeparatedDataset

class IsingModel:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        nodes = np.prod(self.shape)
        self.edges = np.zeros((nodes, nodes), dtype=int)
        self.nodes = np.zeros((nodes), dtype=int)
        self.count = 0

    def fit(self, dataset: Jsb16thSeparatedDataset, epochs: int = 10):
        for epoch in trange(epochs, desc="Fitting Ising Model"):
            for i in range(len(dataset)):
                sample = dataset[i]
                assert sample.dtype == bool, "samples must be binary"
                assert sample.shape == self.shape, "all samples must have the same shape"
                self._update(sample)
    
    def _update(self, sample: np.ndarray):
        sample = sample.reshape(-1)
        self.count += 1
        self.nodes[sample] += 1
        self.edges[sample.reshape(1, -1) == sample.reshape(-1, 1)] += 1