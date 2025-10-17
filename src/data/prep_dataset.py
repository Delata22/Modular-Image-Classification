# File: src/data/prep_dataset.py

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import yaml
from sklearn.model_selection import train_test_split
import sys
from rich import print

# --- CONFIGURATION (can be moved to a config file later if needed) ---
RAW_DATA_ROOT = Path("/app/data/raw_data")
METADATA_ROOT = Path("/app/data/metadata")
SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
RANDOM_SEED = 42

def prepare_dataset(dataset_name: str) -> dict:
    """
    Checks if a dataset is already prepared. If not, prepares it.
    Returns the dataset information (num_classes, class_map).
    """
    dataset_path = RAW_DATA_ROOT / dataset_name
    output_path = METADATA_ROOT / dataset_name
    info_filepath = output_path / "data_info.yaml"

    if not dataset_path.is_dir():
        print(f"[bold red]ERROR: Raw data directory '{dataset_path}' not found.[/bold red]")
        sys.exit(1)

    # Discover current classes
    current_class_names = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    if not current_class_names:
        print(f"[bold red]ERROR: No class subfolders found in {dataset_path}.[/bold red]")
        sys.exit(1)
    current_class_map = {i: name for i, name in enumerate(current_class_names)}

    # Check if it already exists
    if info_filepath.exists():
        print(f"--> A 'data_info.yaml' already exists for '{dataset_name}'. Checking...")
        with open(info_filepath, 'r') as f:
            existing_info = yaml.safe_load(f)
        
        if existing_info.get('class_map') == current_class_map:
            print("[green]✅ Dataset is already prepared and up to date.[/green]")
            return existing_info # Return existing info
        else:
            print("[bold red]❌ MISMATCH ERROR: The dataset content has changed![/bold red]")
            print("[yellow]   If this is intentional, please use a new dataset name (e.g., `ds={dataset_name}_v2`)[/yellow]")
            sys.exit(1)

    # If we get here, it's a first-time preparation
    print(f"--> First-time preparation for dataset '{dataset_name}'. Creating metadata...")
    
    # Scan files and create DataFrame
    class_map_inv = {name: i for i, name in enumerate(current_class_names)}
    records = [{'filepath': str(p.resolve()), 'label_code': class_map_inv[p.parent.name], 'label': p.parent.name}
               for p in dataset_path.glob('**/*') if p.suffix in ['.jpg', '.png']]
    master_df = pd.DataFrame(records)
    
    # Split
    train_df, temp_df = train_test_split(master_df, test_size=(1.0 - SPLIT_RATIOS['train']), stratify=master_df['label'], random_state=RANDOM_SEED)
    relative_test_size = SPLIT_RATIOS['test'] / (SPLIT_RATIOS['val'] + SPLIT_RATIOS['test'])
    val_df, test_df = train_test_split(temp_df, test_size=relative_test_size, stratify=temp_df['label'], random_state=RANDOM_SEED)
    
    # Save
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.drop('label', axis=1).to_parquet(output_path / 'train.parquet')
    val_df.drop('label', axis=1).to_parquet(output_path / 'val.parquet')
    test_df.drop('label', axis=1).to_parquet(output_path / 'test.parquet')

    info_data = {'num_classes': len(current_class_map), 'class_map': current_class_map}
    with open(info_filepath, 'w') as f:
        yaml.dump(info_data, f, default_flow_style=False)
        
    print(f"[green]✅ Preparation completed successfully for dataset '{dataset_name}'.[/green]")
    return info_data