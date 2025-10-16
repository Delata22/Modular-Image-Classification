# Fichier: src/predict.py (Version dynamique finale)

import hydra
from omegaconf import DictConfig
import torch
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm

from src.models.mnist_lit_model import LitMNISTModel

@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def predict(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision('high')

    # 1. Chargement du modèle (inchangé)
    print(f"--> Loading model from checkpoint: {cfg.experiment.ckpt_path}")
    model = LitMNISTModel.load_from_checkpoint(checkpoint_path=cfg.experiment.ckpt_path)
    model.eval()

    # 2. Préparation des transformations (inchangé)
    transform = hydra.utils.instantiate(cfg.data.transform)
    
    # =========================================================================
    # --- LA MODIFICATION CLÉ EST ICI ---
    # =========================================================================
    # On construit le chemin vers le fichier de mappage généré par prepare_dataset.py
    # en utilisant les informations de la configuration 'data'.
    map_filepath = Path("/app/data/") / cfg.data.metadata_dir / "label_codes.json"
    
    print(f"--> Loading class mapping from: {map_filepath}")
    try:
        with open(map_filepath, 'r') as f:
            # On charge le contenu du fichier JSON
            loaded_map = json.load(f) 
            # On s'assure que les clés du dictionnaire sont bien des entiers
            label_map = {int(k): v for k, v in loaded_map.items()}
            print(f"--> Found {len(label_map)} classes in mapping file.")
    except FileNotFoundError:
        print(f"ERREUR : Fichier de mappage '{map_filepath}' non trouvé.")
        print("Avez-vous bien lancé le script de préparation des données avant ?")
        return
    # =========================================================================
    # --- FIN DE LA MODIFICATION ---
    # =========================================================================
    
    # Le reste du script utilise 'label_map' comme avant, mais il a été chargé dynamiquement
    input_dir = Path(cfg.experiment.input_dir)
    output_dir = Path(cfg.experiment.output_dir)
    # ... (le reste du code pour la boucle de prédiction et la sauvegarde est identique)
    # ...
    # (Je remets la boucle ci-dessous pour être complet)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for image_path in tqdm(image_files, desc="Predicting images"):
        image = Image.open(image_path).convert("RGB")
        input_batch = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_batch)
        
        probabilities = torch.nn.functional.softmax(logits, dim=1).squeeze()
        top3_confidences, top3_indices = torch.topk(probabilities, 3)
        top3_confidences_list = top3_confidences.cpu().tolist()
        top3_indices_list = top3_indices.cpu().tolist()
        
        # On utilise le 'label_map' chargé dynamiquement
        top3_labels_list = [label_map.get(i, "classe_inconnue") for i in top3_indices_list]

        result_dict = {
            "file": str(image_path.resolve()),
            "predicted_label_code": top3_indices_list[0],
            "predicted_label": top3_labels_list[0],
            "prediction_confidence": top3_confidences_list[0],
            "top_3_predicted_label_code": top3_indices_list,
            "top_3_predicted_labels": top3_labels_list,
            "top_3_prediction_confidences": top3_confidences_list
        }

        json_filename = image_path.stem + ".json"
        output_filepath = output_dir / json_filename
        with open(output_filepath, 'w') as f:
            json.dump(result_dict, f, indent=4)
    
    print(f"\n--> Done! {len(image_files)} JSON files saved in {output_dir}")

if __name__ == "__main__":
    predict()