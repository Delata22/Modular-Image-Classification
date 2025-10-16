# Fichier: src/test.py (Version finale et robuste)

import hydra
import lightning as L
from omegaconf import DictConfig
import torch

# On a toujours besoin des imports pour que Hydra connaisse les classes de base,
# mais surtout pour pouvoir appeler LitMNISTModel.load_from_checkpoint
from src.data.mnist_datamodule import MNISTDataModule
from src.models.mnist_lit_model import LitMNISTModel


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def test(cfg: DictConfig) -> None:
    """
    Fonction principale pour le TEST, utilisant les meilleures pratiques.
    """
    torch.set_float32_matmul_precision('high')
    
    print("--> Instantiating DataModule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    print(f"--> Loading model from checkpoint: {cfg.experiment.ckpt_path}")
    model = LitMNISTModel.load_from_checkpoint(checkpoint_path=cfg.experiment.ckpt_path)

    print("--> Instantiating Trainer")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    print("--> Starting Testing!")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()