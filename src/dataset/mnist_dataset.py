from PIL import Image
import numpy as np

from functools import cached_property
import os.path

from .dataset import Dataset

path = os.path.join(os.path.dirname(__file__), "mnist.npz")

class MNISTDataset(Dataset):
    @staticmethod
    def save_image(image: np.ndarray, filename: str):
        """save image to png file

        Args:
            image (np.ndarray): numpy array of shape (28, 28)
        """
        image = Image.fromarray(image)
        image.save(filename)

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