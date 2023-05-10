import random

from src.data.dataset import Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory

factory = Jsb16thSeparatedDatasetFactory()

def test_datasets():
    train_dataset = factory.train_dataset
    assert isinstance(train_dataset, Jsb16thSeparatedDataset)

    val_dataset = factory.val_dataset
    assert isinstance(val_dataset, Jsb16thSeparatedDataset)

    test_dataset = factory.test_dataset
    assert isinstance(test_dataset, Jsb16thSeparatedDataset)

def test_iteration():
    # check lengths of all datasets
    train_dataset = factory.train_dataset
    for i in range(len(train_dataset)):
        train_dataset[i]

    val_dataset = factory.val_dataset
    for i in range(len(val_dataset)):
        val_dataset[i]

    test_dataset = factory.test_dataset
    for i in range(len(test_dataset)):
        test_dataset[i]
    
def test_data_shape():
    train_dataset = factory.train_dataset
    val_dataset = factory.val_dataset
    test_dataset = factory.test_dataset

    assert train_dataset[random.randint(0, len(train_dataset))].shape == (64, 46, 4)
    assert val_dataset[random.randint(0, len(val_dataset))].shape == (64, 46, 4)
    assert test_dataset[random.randint(0, len(test_dataset))].shape == (64, 46, 4)