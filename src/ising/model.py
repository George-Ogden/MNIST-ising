from tqdm import trange
import numpy as np

from typing import Tuple

from ..dataset import Dataset

class IsingModel:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
        nodes = np.prod(self.shape)
        self.edges = np.zeros((nodes, nodes), dtype=int)
        self.nodes = np.zeros((nodes), dtype=int)
        self.count = 0

    def fit(self, dataset: Dataset, epochs: int = 10):
        for epoch in trange(epochs, desc="Fitting Ising Model"):
            for sample in dataset:
                assert sample.dtype == bool, "samples must be binary"
                assert sample.shape == self.shape, "all samples must have the same shape"
                self._update(sample)
    
    def _update(self, sample: np.ndarray):
        sample = sample.reshape(-1)
        self.count += 1
        self.nodes[sample] += 1
        self.edges[sample.reshape(1, -1) == sample.reshape(-1, 1)] += 1
    
    def generate(self, iterations: int = 100) -> np.ndarray:
        sample = np.sign(np.random.randn(np.prod(self.shape)))
        sample[sample == 0] = 1
        # $H(\sigma) = -J \sum_{i,j} \sigma_i \sigma_j - h \sum_i \sigma_i$
        J = (self.edges / self.count) * 2 - 1
        J[np.eye(len(J), dtype=bool)] = 0
        h = (self.nodes / self.count) * 2 - 1
        for _ in range(iterations):
            indices = np.random.permutation(len(sample))
            for index in indices:
                delta_energy = h[index] * sample[index] + J[index] * sample * sample[index]
                if (delta_energy).sum() <= 0:
                    sample[index] *= -1
        return sample.reshape(self.shape) > 0