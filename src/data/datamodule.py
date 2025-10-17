# File: src/data/datamodule.py

from pathlib import Path
import pandas as pd
import yaml
from PIL import Image

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

class ParquetDataset(Dataset):
    """Dataset that reads data from a pandas DataFrame."""
    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose = None):
        self.metadata = metadata
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # Get the path and label from the DataFrame
        filepath = self.metadata.filepath.iloc[index]
        label_code = self.metadata.label_code.iloc[index]
        
        # Opens the image and ensures a 3-channel (RGB) conversion
        # to be compatible with most pre-trained models.
        image = Image.open(filepath).convert("RGB")
        
        # Apply transformations (Resize, ToTensor, Normalize, etc.)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_code, dtype=torch.long), filepath

class ParquetDataModule(LightningDataModule):
    """Generic DataModule that reads metadata in Parquet format."""
    def __init__(self, data_root_dir: str, metadata_dir: str, batch_size: int, num_workers: int, transform: transforms.Compose = None):
        super().__init__()
        # Saves hyperparameters (batch_size, etc.) to make them accessible
        self.save_hyperparameters(logger=False)

        self.transform = transform
        
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Reads the info file to discover dataset properties (num_classes, etc.)."""
        info_filepath = Path(self.hparams.data_root_dir) / self.hparams.metadata_dir / "data_info.yaml"
        try:
            with open(info_filepath, 'r') as f:
                info_data = yaml.safe_load(f)
            self.num_classes = info_data['num_classes']
            self.class_map = info_data['class_map']
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {info_filepath} not found. "
                "Did you run the data preparation script beforehand?"
            )

    def setup(self, stage: str = None):
        """Loads the Parquet files and creates Dataset instances."""
        root_path = Path(self.hparams.data_root_dir) / self.hparams.metadata_dir
        self.data_train = ParquetDataset(pd.read_parquet(root_path / "train.parquet"), self.transform)
        self.data_val = ParquetDataset(pd.read_parquet(root_path / "val.parquet"), self.transform)
        self.data_test = ParquetDataset(pd.read_parquet(root_path / "test.parquet"), self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)