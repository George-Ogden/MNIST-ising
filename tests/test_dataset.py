from pytest import mark
import numpy as np

from src.dataset import Dataset, JSBDatasetInfo, Jsb16thSeparatedDataset, Jsb16thSeparatedDatasetFactory, MNISTDataset, MNISTDatasetFactory

jsb_factory = Jsb16thSeparatedDatasetFactory()
jsb_datasets = [jsb_factory.train_dataset, jsb_factory.val_dataset, jsb_factory.test_dataset]

mnist_factory = MNISTDatasetFactory()
mnist_datasets = [mnist_factory.train_dataset, mnist_factory.test_dataset]

jsb_datasets_test = mark.parametrize("dataset", jsb_datasets)
mnist_datasets_test = mark.parametrize("dataset", mnist_datasets)
datasets_test = mark.parametrize("dataset", jsb_datasets + mnist_datasets)

def test_slicing():
    dataset = Dataset(list(range(10)))
    sliced_dataset = dataset[2:5]
    assert isinstance(sliced_dataset, Dataset)
    assert sliced_dataset.data == [2, 3, 4]

def test_dataset_factories():
    train_dataset = jsb_factory.train_dataset
    assert isinstance(train_dataset, Jsb16thSeparatedDataset)

    val_dataset = jsb_factory.val_dataset
    assert isinstance(val_dataset, Jsb16thSeparatedDataset)

    test_dataset = jsb_factory.test_dataset
    assert isinstance(test_dataset, Jsb16thSeparatedDataset)

def test_jsb_config_propagates():
    info = JSBDatasetInfo(
        piece_length=16,
        min_pitch=4,
        max_pitch=125,
        resolution=4,
    )

    factory = Jsb16thSeparatedDatasetFactory(
        info=info
    )
    dataset = factory.train_dataset
    assert dataset.info == info
    assert dataset.min_pitch == 4
    assert dataset.max_pitch == 125
    assert dataset.resolution == 4

    sample = dataset[0]
    assert sample.shape == (16, 122)

@datasets_test
def test_dataset_iteration(dataset):
    # check lengths of the dataset
    dataset = jsb_factory.train_dataset
    for i in range(len(dataset)):
        dataset[i]

@jsb_datasets_test
def test_jsb_data_samples(dataset):
    sample = dataset[np.random.randint(0, len(dataset))]
    assert sample.shape == (64, 46)
    assert sample.dtype == bool

@mnist_datasets_test
def test_mnist_data_samples(dataset):
    sample = dataset[np.random.randint(0, len(dataset))]
    assert sample.shape == (28, 28)
    assert sample.dtype == bool

def test_dataset_random_crop():
    piece = [[i] for i in range(128)]
    
    cropping_dataset = Jsb16thSeparatedDataset(
        [piece],
        JSBDatasetInfo(
            piece_length=32,
            min_pitch=0,
            max_pitch=127
        )
    )

    for _ in range(100):
        piece = cropping_dataset[0]
        for notes in zip(piece[:-1], piece[1:]):
            assert notes[0].argmax() + 1 == notes[1].argmax()

def test_instrument_merging():
    piece = [[i, i + 64] for i in range(64)]
    
    dataset = Jsb16thSeparatedDataset(
        [piece],
        JSBDatasetInfo(
            piece_length=32,
            min_pitch=0,
            max_pitch=127
        )
    )

    for _ in range(100):
        piece = dataset[0]
        for note in piece:
            lower_note = note.argmax()
            assert note[lower_note] and note[lower_note + 64]