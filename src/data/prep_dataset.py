# Fichier: src/data_preparation.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import sys
from rich import print

# --- CONFIGURATION (peut être déplacée dans un fichier config plus tard si besoin) ---
RAW_DATA_ROOT = Path("/app/data/raw_data")
METADATA_ROOT = Path("/app/data/metadata")
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
RANDOM_SEED = 42

def prepare_dataset(dataset_name: str) -> dict:
    """
    Vérifie si un dataset est déjà préparé. Si non, le prépare.
    Retourne les informations du dataset (num_classes, class_map).
    """
    dataset_path = RAW_DATA_ROOT / dataset_name
    output_path = METADATA_ROOT / dataset_name
    info_filepath = output_path / "data_info.yaml"

    if not dataset_path.is_dir():
        print(f"[bold red]ERREUR : Le dossier de données brutes '{dataset_path}' est introuvable.[/bold red]")
        sys.exit(1)

    # Découverte des classes actuelles
    current_class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    if not current_class_names:
        print(f"[bold red]ERREUR : Aucun sous-dossier de classe trouvé dans {dataset_path}.[/bold red]")
        sys.exit(1)
    current_class_map = {i: name for i, name in enumerate(current_class_names)}

    # Vérification de l'existant
    if info_filepath.exists():
        print(f"--> Un 'data_info.yaml' existe déjà pour '{dataset_name}'. Vérification...")
        with open(info_filepath, 'r') as f:
            existing_info = yaml.safe_load(f)
        
        if existing_info.get('class_map') == current_class_map:
            print("[green]✅ Le dataset est déjà préparé et à jour.[/green]")
            return existing_info # On retourne les infos existantes
        else:
            print("[bold red]❌ ERREUR DE CONCORDANCE : Le contenu du dataset a changé ![/bold red]")
            print("[yellow]   Si c'est intentionnel, veuillez utiliser un nouveau nom de dataset (ex: `ds={dataset_name}_v2`)[/yellow]")
            sys.exit(1)

    # Si on arrive ici, c'est une première préparation
    print(f"--> Première préparation pour le dataset '{dataset_name}'. Création des métadonnées...")
    
    # Scan des fichiers et création du DataFrame
    class_map_inv = {name: i for i, name in enumerate(current_class_names)}
    records = [{'filepath': str(p.resolve()), 'label_code': class_map_inv[p.parent.name], 'label': p.parent.name}
               for p in dataset_path.glob('**/*') if p.suffix in ['.jpg', '.png']]
    master_df = pd.DataFrame(records)
    
    # Split
    train_df, temp_df = train_test_split(master_df, test_size=(1.0 - SPLIT_RATIOS['train']), stratify=master_df['label'], random_state=RANDOM_SEED)
    relative_test_size = SPLIT_RATIOS['test'] / (SPLIT_RATIOS['val'] + SPLIT_RATIOS['test'])
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, stratify=temp_df['label'], random_state=RANDOM_SEED)
    
    # Sauvegarde
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.drop('label', axis=1).to_parquet(output_path / 'train.parquet')
    val_df.drop('label', axis=1).to_parquet(output_path / 'val.parquet')
    test_df.drop('label', axis=1).to_parquet(output_path / 'test.parquet')

    info_data = {'num_classes': len(current_class_map), 'class_map': current_class_map}
    with open(info_filepath, 'w') as f:
        yaml.dump(info_data, f, default_flow_style=False)
        
    print(f"[green]✅ Préparation terminée avec succès pour le dataset '{dataset_name}'.[/green]")