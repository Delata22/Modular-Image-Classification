# Fichier: src/data/datamodule.py

from pathlib import Path
import pandas as pd
import yaml
from PIL import Image

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

class ParquetDataset(Dataset):
    """Dataset qui lit les données à partir d'un DataFrame pandas."""
    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose = None):
        self.metadata = metadata
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # Récupère le chemin et le label depuis le DataFrame
        filepath = self.metadata.filepath.iloc[index]
        label_code = self.metadata.label_code.iloc[index]
        
        # Ouvre l'image et garantit une conversion en 3 canaux (RGB)
        # pour être compatible avec la plupart des modèles pré-entraînés.
        image = Image.open(filepath).convert("RGB")
        
        # Applique les transformations (Resize, ToTensor, Normalize, etc.)
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label_code, dtype=torch.long), filepath

class ParquetDataModule(LightningDataModule):
    """DataModule générique qui lit des métadonnées au format Parquet."""
    def __init__(self, data_root_dir: str, metadata_dir: str, batch_size: int, num_workers: int, transform: transforms.Compose = None):
        super().__init__()
        # Sauvegarde les hyperparamètres (batch_size, etc.) pour les rendre accessibles
        self.save_hyperparameters(logger=False)

        self.transform = transform
        
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        """Lit le fichier d'info pour découvrir les propriétés du dataset (num_classes, etc.)."""
        info_filepath = Path(self.hparams.data_root_dir) / self.hparams.metadata_dir / "data_info.yaml"
        try:
            with open(info_filepath, 'r') as f:
                info_data = yaml.safe_load(f)
            self.num_classes = info_data['num_classes']
            self.class_map = info_data['class_map']
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Le fichier {info_filepath} est introuvable. "
                "Avez-vous bien lancé le script de préparation des données avant ?"
            )

    def setup(self, stage: str = None):
        """Charge les fichiers Parquet et crée les instances de Dataset."""
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