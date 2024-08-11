import lightning
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .config import DataModuleConfig
import torchvision

class MNIST_DataModule(lightning.LightningDataModule):
    def __init__(self, config: DataModuleConfig):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        self.train = torchvision.datasets.MNIST(
            root=self.config.dataset_path,
            train=True,
            transform=self.transform,
            download=True
        )

        self.test = torchvision.datasets.MNIST(
            root=self.config.dataset_path,
            train=False,
            transform=self.transform,
            download=True
        )

    def get_image_shape(self):
        return self.train[0][0].shape

    def setup(self, stage):
        '''Splitting dataset into train, valid, test'''
        if stage == 'fit':
            self.train, self.valid = random_split(
                dataset=self.train, 
                lengths=self.config.train_valid_ratio
            )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config.train_batch,
            num_workers=self.config.num_worker,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid,
            batch_size=self.config.valid_batch,
            num_workers=self.config.num_worker,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.config.test_batch,
            num_workers=self.config.num_worker,
            shuffle=False
        )

if __name__ == "__main__":
    dataset = MNIST_DataModule(
        config=DataModuleConfig(
            dataset_path="./storage/dataset",
            train_valid_ratio=[0.9, 0.1],
            train_batch=100,
            valid_batch=300,
            test_batch=300,
            num_worker=2
        )
    )

    print(list(dataset.get_image_shape()))