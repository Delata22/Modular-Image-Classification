# Fichier: src/utils/transforms_factory.py

from omegaconf import DictConfig
import hydra
from transformers import AutoConfig
import torchvision.transforms as T
from rich import print

def get_image_size_from_model_name(model_name: str) -> int:
    """
    Charge la config d'un modèle Hugging Face et retourne la taille d'image attendue.
    """
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        image_size = model_config.image_size
        print(f"--> Taille d'image attendue par le modèle '{model_name}': [yellow]{image_size}x{image_size}[/yellow]")
        return image_size
    except Exception:
        print(f"[yellow]AVERTISSEMENT : Impossible de déterminer la taille d'image pour '{model_name}'. Utilisation de 224 par défaut.[/yellow]")
        return 224

def build_dynamic_transforms(static_transforms_cfg: DictConfig, model_name: str) -> T.Compose:
    """
    Construit la chaîne de transformation complète en ajoutant dynamiquement le Resize.
    
    Args:
        static_transforms_cfg: La partie de la config Hydra contenant les transformations statiques.
        model_name: Le nom du modèle Hugging Face pour déterminer la taille de l'image.
        
    Returns:
        Un objet torchvision.transforms.Compose complet.
    """
    # 1. On détermine la taille de l'image
    image_size = get_image_size_from_model_name(model_name)

    # 2. On instancie la partie STATIQUE des transformations depuis la config
    static_transforms = hydra.utils.instantiate(static_transforms_cfg)
    
    # 3. On construit la chaîne de transformation COMPLÈTE en Python
    full_transform = T.Compose([
        T.Resize((image_size, image_size)), # Le Resize dynamique
        *static_transforms.transforms     # On déballe la liste des transforms statiques
    ])
    
    return full_transform