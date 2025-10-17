# File: src/utils/transforms_factory.py

from omegaconf import DictConfig
import hydra
from transformers import AutoConfig
import torchvision.transforms as T
from rich import print

def get_image_size_from_model_name(model_name: str) -> int:
    """
    Loads the config of a Hugging Face model and returns the expected image size.
    """
    try:
        model_config = AutoConfig.from_pretrained(model_name)
        image_size = model_config.image_size
        print(f"--> Expected image size for model '{model_name}': [yellow]{image_size}x{image_size}[/yellow]")
        return image_size
    except Exception:
        print(f"[yellow]WARNING: Could not determine image size for '{model_name}'. Defaulting to 224.[/yellow]")
        return 224

def build_dynamic_transforms(static_transforms_cfg: DictConfig, model_name: str) -> T.Compose:
    """
    Builds the complete transformation chain by dynamically adding the Resize transform.
    
    Args:
        static_transforms_cfg: The part of the Hydra config containing the static transformations.
        model_name: The name of the Hugging Face model to determine the image size.
        
    Returns:
        A complete torchvision.transforms.Compose object.
    """
    # 1. Determine the image size
    image_size = get_image_size_from_model_name(model_name)

    # 2. Instantiate the STATIC part of the transformations from the config
    static_transforms = hydra.utils.instantiate(static_transforms_cfg)
    
    # 3. Build the COMPLETE transformation chain in Python
    full_transform = T.Compose([
        T.Resize((image_size, image_size)), # The dynamic Resize
        *static_transforms.transforms     # Unpack the list of static transforms
    ])
    
    return full_transform