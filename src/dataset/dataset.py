import numpy as np

from typing import List, Union

class Dataset:
    def __init__(self, data: List[List[List[int]]]):
        self.data = data
    
    def augment(self, data: np.ndarray):
        return data

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        if isinstance(idx, slice):
            return Dataset(self.data[idx])
        # return a random crop of the piece
        return self.augment(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)