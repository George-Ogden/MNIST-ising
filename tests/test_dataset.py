from pytest import mark
import numpy as np

from src.data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory

factory = Jsb16thSeparatedDatasetFactory()
datasets = mark.parametrize("dataset", [factory.train_dataset, factory.val_dataset, factory.test_dataset])

def test_datasets():
    train_dataset = factory.train_dataset
    assert isinstance(train_dataset, Jsb16thSeparatedDataset)

    val_dataset = factory.val_dataset
    assert isinstance(val_dataset, Jsb16thSeparatedDataset)

    test_dataset = factory.test_dataset
    assert isinstance(test_dataset, Jsb16thSeparatedDataset)

@datasets
def test_iteration(dataset):
    # check lengths of the dataset
    dataset = factory.train_dataset
    for i in range(len(dataset)):
        dataset[i]

@datasets
def test_data_samples(dataset):
    sample = dataset[np.random.randint(0, len(dataset))]
    assert sample.shape == (64, 46, 4)