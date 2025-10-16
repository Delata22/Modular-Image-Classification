from typing import Optional
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class MNISTDataModule(LightningDataModule):

    # data_dir: path where the MNIST dataset will be stored
    # batch_size: size of the batches created by the DataLoaders
    # num_workers: number of processes created by the CPU to transfer the saved dataset data to the GPU while training
    def __init__(self, data_dir: str = "data/", batch_size: int = 32, num_workers: int = 4, transform: transforms.Compose = None) :
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081, ))
        ])
    
    # Download the MNIST dataset and put it in the directory indicated by self.data_dir
    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)


    # Get the MNIST dataset from disk and split it between the train, val and test dataset and store them in the class
    def setup(self, stage: Optional[str] = None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    
    # Functions to get the DataLoaders of the train, val and test dataset
    def train_dataloader(self):
        out = DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=self.num_workers)
        return out
    
    def test_dataloader(self):
        out = DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
        return out
    
    def val_dataloader(self):
        out = DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)
        return out
    
    