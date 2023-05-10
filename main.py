from src.dataset import Jsb16thSeparatedDatasetFactory
from src.ising import IsingModel

if __name__ == "__main__":
    factory = Jsb16thSeparatedDatasetFactory()
    dataset = factory.train_dataset

    model = IsingModel(dataset[0].shape)
    model.fit(dataset, epochs=5)

    sample = model.generate()
    dataset.info.save_pianoroll(sample, "sample.mid")