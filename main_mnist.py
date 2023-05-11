from src.dataset import MNISTDatasetFactory
from src.ising import IsingModel

if __name__ == "__main__":
    factory = MNISTDatasetFactory()
    dataset = factory.train_dataset

    model = IsingModel(dataset[0].shape)
    model.fit(dataset)

    sample = model.generate()
    dataset.save_image(sample, "sample.png")    