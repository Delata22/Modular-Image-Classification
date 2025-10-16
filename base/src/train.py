# Fichier: src/train.py (Version dynamique finale)

import hydra
import lightning as L
from omegaconf import DictConfig
import torch

from src.data.irm_datamodule import IRMDataModule # <--- Mettre à jour l'import
from src.models.mnist_lit_model import LitMNISTModel

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')
    L.seed_everything(cfg.seed)

    # 1. On instancie le DataModule D'ABORD
    print("--> Instantiating DataModule")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # 2. On exécute sa préparation pour qu'il découvre les infos du dataset
    datamodule.prepare_data()

    # 3. On récupère le nombre de classes et on met à jour la config du modèle
    print(f"--> Discovered {datamodule.num_classes} classes from data_info.yaml")
    cfg.model.num_classes = datamodule.num_classes
    
    # 4. MAINTENANT, on peut instancier le modèle avec la bonne config
    print("--> Instantiating Model")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Le reste ne change pas...
    print("--> Instantiating Callbacks & Logger")
    callbacks_list = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            callbacks_list.append(hydra.utils.instantiate(cb_conf))
    
    logger_list = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            logger_list.append(hydra.utils.instantiate(lg_conf))

    print("--> Instantiating Trainer")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks_list, logger=logger_list)

    print("--> Starting Training!")
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()