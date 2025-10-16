import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import yaml
import glob
import os
from sklearn.model_selection import train_test_split

RAW_DATA_ROOT = Path("/app/raw_data/")

# Le dossier où seront sauvegardés les .parquet ET le fichier d'info
OUTPUT_DIR = Path("/app/data/metadata")

SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
RANDOM_SEED = 42

def create_dataset_metadata():
    print(f"--- Démarrage de l'inspection du dossier : {RAW_DATA_ROOT} ---")
    if not RAW_DATA_ROOT.is_dir():
        print(f"ERREUR : Dossier non trouvé !")
        return

    # 1. DÉCOUVERTE AUTOMATIQUE DES CLASSES
    class_names = sorted([d.name for d in RAW_DATA_ROOT.iterdir() if d.is_dir()])
    if not class_names:
        print("ERREUR : Aucun sous-dossier de classe trouvé.")
        return
        
    num_classes = len(class_names)
    print(f"--> {num_classes} classes découvertes : {class_names}")

    # 2. CRÉATION AUTOMATIQUE DU MAPPAGE
    class_map = {i: name for i, name in enumerate(class_names)}
    class_map_inv = {name: i for i, name in enumerate(class_names)}

    # 3. SCAN DES FICHIERS ET CRÉATION DU DATAFRAME
    records = []
    for class_name in class_names:
        label_code = class_map_inv[class_name]
        class_path = RAW_DATA_ROOT / class_name
        
        image_paths = list(class_path.glob('**/*.jpg')) + list(class_path.glob('**/*.png'))

        for image_path in image_paths:
            records.append({
                'filepath': str(image_path.resolve()),
                'label_code': label_code
            })
            
    master_df = pd.DataFrame(records)
    print(f"--> {len(master_df)} images trouvées au total.")

    # 4. DIVISION ET SAUVEGARDE DES FICHIERS PARQUET (inchangé)
    master_df = master_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    # ... (copiez ici la logique de split et de sauvegarde en .parquet du script précédent)

    data = [{'label': class_label, 'filepath': glob.glob(os.path.join(RAW_DATA_ROOT, class_label, '*'))} for class_label in class_names]
    df = pd.DataFrame(data).explode('filepath')
    df = df.reset_index(drop = True)


    df['categories'] = pd.Categorical(df['label'])
    df['label_code'] = df.categories.cat.codes

    print(df)


    df_train, df_test = train_test_split(df, stratify=df.label_code, random_state = RANDOM_SEED, test_size = 0.35)
    df_val, df_test = train_test_split(df_test, stratify=df_test.label_code, random_state = RANDOM_SEED, test_size = 0.8)


    output_metadata_dir = 'metadata_detailed'

    df_train.to_parquet(Path('/app/data/' + output_metadata_dir + '/train.parquet'))
    df_val.to_parquet(Path('/app/data/' + output_metadata_dir + '/val.parquet'))
    df_test.to_parquet(Path('/app/data/' + output_metadata_dir + '/test.parquet'))

    actual_json_name = 'label_codes.json'
    mapping = df[['label', 'label_code']].drop_duplicates().set_index('label_code').to_dict()['label']

    with open('/app/data/metadata/' + actual_json_name, 'w') as f:
        json.dump(mapping, f)

    # 5. SAUVEGARDE DES INFORMATIONS DÉCOUVERTES
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    info_data = {
        'num_classes': num_classes,
        'class_map': class_map
    }
    info_filepath = OUTPUT_DIR / "data_info.yaml"
    with open(info_filepath, 'w') as f:
        yaml.dump(info_data, f, default_flow_style=False)
        
    print(f"--> Fichier d'information sauvegardé ici : {info_filepath}")
    print("--- Préparation terminée ! ---")


if __name__ == "__main__":
    create_dataset_metadata()