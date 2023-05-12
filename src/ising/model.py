from tqdm import tqdm, trange
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

    def fit(self, dataset: Dataset):
        for sample in tqdm(dataset, desc="Fitting Ising model"):
            assert sample.dtype == bool, "samples must be binary"
            assert sample.shape == self.shape, "all samples must have the same shape"
            self._update(sample)
    
    def prune(self) -> Tuple[np.ndarray, np.ndarray]:
        h = (self.nodes / self.count) * 2 - 1
        J = ((self.edges / self.count) * 2 - 1)
        J[np.eye(len(J), dtype=bool)] = 0
        for _ in trange(int(len(h) ** 1.25), desc="Pruning Ising model"):
            cumulative = np.abs(J).sum(axis=-1)
            weights = np.exp(cumulative)
            i = np.random.choice(np.arange(len(J)), p=weights / weights.sum())
            weights = np.exp(J[i] * np.sign(cumulative[i]))
            j = np.random.choice(np.arange(len(J)), p=weights / weights.sum())
            cumulative[i] -= J[i,j]
            cumulative[j] -= J[i,j]
            J[i,j] = J[j,i] = 0
        return h, J
    
    def _update(self, sample: np.ndarray):
        sample = sample.reshape(-1)
        self.count += 1
        self.nodes[sample] += 1
        self.edges[sample.reshape(1, -1) == sample.reshape(-1, 1)] += 1
    
    def generate(self) -> np.ndarray:
        sample = np.sign(np.random.randn(np.prod(self.shape)))
        sample[sample == 0] = 1
        # $H(\sigma) = -J \sum_{i,j} \sigma_i \sigma_j - h \sum_i \sigma_i$
        h, J = self.prune()
        change = True
        while change:
            change = False
            indices = np.random.permutation(len(sample))
            for index in indices:
                delta_energy = h[index] * sample[index] + J[index] @ sample * sample[index]
                if delta_energy < 0:
                    sample[index] *= -1
                    change = True
        return sample.reshape(self.shape) > 0