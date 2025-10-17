# Fichier: src/train.py (Version finale, propre et dynamique)

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
import torch
from rich import print
import sys
from src.utils.transforms_resolver import build_dynamic_transforms
from src.data.prep_dataset import prepare_dataset
# Imports nécessaires pour que Hydra puisse trouver les classes cibles
# même si elles ne sont pas directement appelées dans ce fichier.
from src.data.datamodule import ParquetDataModule
from src.models.lit_model import LitModel
from transformers import AutoConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Fonction principale d'entraînement, pilotée par Hydra.
    """

    print(f"Using dataset :[bold blue] {cfg.ds} [/bold blue] ")
    print(f"Using model '{cfg.model.model_name}' ")

    # Active les optimisations pour les Tensor Cores des GPUs modernes
    if torch.cuda.is_available():
        print("GPU detected, using it to full capabilities")
        torch.set_float32_matmul_precision('high')
    else:
        print("Using CPU")

    print("--> Creating metadata")
    prepare_dataset(dataset_name=cfg.ds)
    # Fixe la graine aléatoire pour garantir la reproductibilité des expériences
    L.seed_everything(cfg.seed)

    # --- 1. Instanciation dynamique des composants ---
    full_transform = build_dynamic_transforms(
        static_transforms_cfg=cfg.data.transform,
        model_name=cfg.model.model_name
    )

    # 2. On instancie le DataModule en lui passant la transformation complète
    print("--> Instanciation du DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data, transform=full_transform)
    
    # On exécute 'prepare_data' manuellement pour forcer la lecture du fichier data_info.yaml
    # et découvrir le nombre de classes avant d'instancier le modèle.
    datamodule.prepare_data()

    # On met à jour la configuration du modèle avec le nombre de classes découvert
    print(f"--> Discovered {datamodule.num_classes} classes from data_info.yaml")
    cfg.model.num_classes = datamodule.num_classes
    
    print("--> Instantiating Model")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    print("--> Instantiating Callbacks")
    callbacks_list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            callbacks_list.append(hydra.utils.instantiate(cb_conf))
    
    print("--> Instantiating Logger")
    logger_list = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            logger_list.append(hydra.utils.instantiate(lg_conf))

    print("--> Instantiating Trainer")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks_list, 
        logger=logger_list
    )

    # --- 2. Lancement de l'entraînement et du test ---

    print("--> Starting Training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    print("--> Starting Testing!")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()