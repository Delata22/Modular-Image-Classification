# Fichier: src/predict.py (Version finale refactorisée)

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import yaml
import sys
from rich import print

from src.models.lit_model import LitModel
from src.utils.transforms_resolver import build_dynamic_transforms

# --- FONCTION 1 : Le "Loader" ---
def load_components(model_dir_path: str) -> dict:
    """Charge tous les composants nécessaires à partir d'un dossier de modèle."""
    print("--> Chargement des composants du modèle...")
    model_dir = Path(model_dir_path)
    print(model_dir)
    if not model_dir.is_dir():
        print(f"[bold red]❌ ERREUR : Le dossier du modèle '{model_dir}' est introuvable.[/bold red]")
        sys.exit(1)

    # A) Charger la configuration d'origine
    original_config_path = model_dir / ".hydra" / "config.yaml"
    if not original_config_path.exists():
        print(f"[bold red]❌ ERREUR : Fichier '.hydra/config.yaml' introuvable dans '{model_dir}'.[/bold red]")
        sys.exit(1)
    train_cfg = OmegaConf.load(original_config_path)

    # B) Trouver le checkpoint
    try:
        ckpt_path = next(model_dir.glob("checkpoints/*.ckpt"))
    except StopIteration:
        print(f"[bold red]❌ ERREUR : Aucun fichier .ckpt trouvé dans '{model_dir}/checkpoints/'.[/bold red]")
        sys.exit(1)
    
    # C) Charger le mappage des classes
    info_path = model_dir / "train_info/data_info.yaml"
    if not info_path.exists():
        print(f"[bold red]❌ ERREUR : Fichier 'data_info.yaml' introuvable dans '{model_dir}'.[/bold red]")
        sys.exit(1)
    with open(info_path, 'r') as f:
        class_map = yaml.safe_load(f)['class_map']

    # D) Charger le modèle et les transformations
    model = LitModel.load_from_checkpoint(checkpoint_path=ckpt_path)
    model.eval()
    transform = build_dynamic_transforms(train_cfg.data.transform, train_cfg.model.model_name)
    
    return {"model": model, "transform": transform, "class_map": class_map}

# --- FONCTION 2 : Le "Moteur d'Inférence" ---
def predict_single_image(image_path: Path, components: dict, device) -> dict:
    """Effectue une prédiction sur une seule image et retourne un dictionnaire de résultats."""
    model = components['model']
    transform = components['transform']
    class_map = components['class_map']

    image = Image.open(image_path).convert("RGB")
    input_batch = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
    top3_confidences, top3_indices = torch.topk(probabilities, 3)

    top3_confidences_list = top3_confidences.cpu().tolist()
    top3_indices_list = top3_indices.cpu().tolist()
    top3_labels_list = [class_map.get(i, "classe_inconnue") for i in top3_indices_list]

    return {
        "file": str(image_path.resolve()),
        "top_3_predicted_labels": top3_labels_list,
        "top_3_predicted_codes": top3_indices_list,
        "top_3_probabilities": top3_confidences_list
    }

# --- FONCTION 3 : Le "Chef d'Orchestre" ---
@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def predict(cfg: DictConfig) -> None:
    if torch.cuda.is_available():
        print("GPU detected, using it to full capabilities")
        torch.set_float32_matmul_precision('high')
    else:
        print("Using CPU")


    # 1. On charge tous nos outils une seule fois
    components = load_components(model_dir_path=cfg.md)
    
    # 2. On détermine la liste des images à traiter (dossier OU fichier unique)
    if cfg.file:
        image_files = [Path(cfg.file)]
        if not image_files[0].is_file():
            print(f"[bold red]❌ ERREUR : Le fichier '{cfg.file}' est introuvable.[/bold red]")
            sys.exit(1)
    else:
        input_dir = Path(cfg.predict.input_dir)
        image_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    
    output_dir = Path(cfg.predict.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components['model'].to(device)

    print(f"--> {len(image_files)} image(s) à prédire. Les résultats seront sauvegardés dans '{output_dir}'.")

    # 3. On boucle et on appelle le moteur d'inférence
    for image_path in tqdm(image_files, desc="Prédiction sur les images"):
        result_dict = predict_single_image(image_path, components, device)
        
        json_filename = image_path.stem + ".json"
        output_filepath = output_dir / json_filename
        with open(output_filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
    
    print(f"\n[green]✅ Terminé ! {len(image_files)} fichiers JSON sauvegardés dans {output_dir}[/green]")

if __name__ == "__main__":
    predict()