from functools import cached_property
import numpy as np
import os.path

from .dataset import Dataset

path = os.path.join(os.path.dirname(__file__), "mnist.npz")

class MNISTDataset(Dataset):
    ...

class MNISTDatasetFactory:
    """create datasets from a json file"""
    def __init__(self, path: str = path):
        self.data = np.load(path)

    @cached_property
    def train_dataset(self) -> MNISTDataset:
        return MNISTDataset(self.data["train"])

    @cached_property
    def test_dataset(self):
        return MNISTDataset(self.data["test"])