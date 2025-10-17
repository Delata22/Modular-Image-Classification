# File: src/train.py (Final, clean, and dynamic version)

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf
import torch
from rich import print
import sys
from src.utils.transforms_resolver import build_dynamic_transforms
from src.data.prep_dataset import prepare_dataset

# Imports needed for Hydra to find the target classes
# even if they are not directly called in this file.
from src.data.datamodule import ParquetDataModule
from src.models.lit_model import LitModel
from transformers import AutoConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """
    Main training function, driven by Hydra.
    """

    print(f"Using dataset: [bold blue]{cfg.ds}[/bold blue]")
    print(f"Using model: '{cfg.model.model_name}'")

    # Enables optimizations for modern GPU Tensor Cores
    if torch.cuda.is_available():
        print("GPU detected, using it to full capabilities")
        torch.set_float32_matmul_precision('high')
    else:
        print("Using CPU")

    print("--> Creating metadata")
    prepare_dataset(dataset_name=cfg.ds)
    # Sets the random seed to ensure experiment reproducibility
    L.seed_everything(cfg.seed)

    # --- 1. Dynamic component instantiation ---
    full_transform = build_dynamic_transforms(
        static_transforms_cfg=cfg.data.transform,
        model_name=cfg.model.model_name
    )

    # 2. We instantiate the DataModule, passing it the complete transformation
    print("--> Instantiating DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data, transform=full_transform)
    
    # We manually run 'prepare_data' to force reading the data_info.yaml file
    # and discover the number of classes before instantiating the model.
    datamodule.prepare_data()

    # We update the model configuration with the discovered number of classes
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

    # --- 2. Launching training and testing ---

    print("--> Starting Training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    print("--> Starting Testing!")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()