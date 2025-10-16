# Fichier: src/data/irm_datamodule.py

from pathlib import Path
import pandas as pd
import yaml
from PIL import Image
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

# --- Le Dataset qui lit les Parquet ---
class IRMDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, transform: transforms.Compose = None):
        self.metadata = metadata
        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        filepath = self.metadata.filepath.iloc[index]
        label_code = self.metadata.label_code.iloc[index]
        # On force la conversion en RGB pour être compatible avec les modèles pré-entraînés
        image = Image.open(filepath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_code

# --- Le DataModule qui orchestre tout ---
class IRMDataModule(LightningDataModule):
    def __init__(self, data_root_dir: str, metadata_dir: str, batch_size: int, num_workers: int, transform: transforms.Compose = None):
        super().__init__()
        self.save_hyperparameters(logger=False) # Sauvegarde les hyperparamètres
        self.transform = transform
        
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def prepare_data(self):
        # Lit le fichier d'info pour découvrir les propriétés du dataset
        info_filepath = Path(self.hparams.data_root_dir) / self.hparams.metadata_dir / "data_info.yaml"
        with open(info_filepath, 'r') as f:
            info_data = yaml.safe_load(f)
        self.num_classes = info_data['num_classes']
        self.class_map = info_data['class_map']

    def setup(self, stage: str = None):
        # Charge les DataFrames et crée les Datasets
        train_df = pd.read_parquet(f"{self.hparams.data_root_dir}/{self.hparams.metadata_dir}/train.parquet")
        val_df = pd.read_parquet(f"{self.hparams.data_root_dir}/{self.hparams.metadata_dir}/val.parquet")
        test_df = pd.read_parquet(f"{self.hparams.data_root_dir}/{self.hparams.metadata_dir}/test.parquet")
        
        self.data_train = IRMDataset(train_df, self.transform)
        self.data_val = IRMDataset(val_df, self.transform)
        self.data_test = IRMDataset(test_df, self.transform)

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)