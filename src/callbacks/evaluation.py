# Fichier: src/callbacks/evaluation.py

from lightning import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics import ConfusionMatrix
from pathlib import Path
import shutil
from rich import print

class TestEvaluationCallback(Callback):
    """
    Callback qui s'active à la fin du test pour générer :
    1. Une image de la matrice de confusion.
    2. Un fichier CSV listant toutes les images mal classées.
    3. Une copie du fichier d'information du dataset.
    """

    @rank_zero_only  # Assure que ce code ne s'exécute que sur le processus principal
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        print("\n--> Génération du rapport d'évaluation final...")
        
        outputs = pl_module.test_step_outputs
        if not outputs:
            print("Aucune sortie de test à analyser.")
            return

        # 1. Rassembler toutes les prédictions, étiquettes et chemins de fichiers
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        filepaths = [path for batch_output in outputs for path in batch_output['filepaths']]
        
        # 2. Récupérer les informations utiles depuis le Trainer et le DataModule
        class_map = trainer.datamodule.class_map
        class_names = [class_map.get(i, str(i)) for i in range(pl_module.hparams.num_classes)]
        save_dir = Path(trainer.logger.log_dir).parent / "train_info"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 3. Appeler les fonctions pour générer les rapports
        self.plot_confusion_matrix(preds, labels, class_names, save_dir)
        self.save_misclassified_as_csv(preds, labels, filepaths, class_map, save_dir)
        self.copy_data_info(trainer, save_dir)
        
        print(f"Your model as been saved in '{str(save_dir.parent)}'")
        # 4. Vider la mémoire pour le prochain run
        pl_module.test_step_outputs.clear()

    def plot_confusion_matrix(self, preds, labels, class_names, save_dir):
        """Calcule et sauvegarde la matrice de confusion."""
        print("--> Génération de la matrice de confusion...")
        confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
        cm_tensor = confmat(preds.cpu(), labels.cpu())

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_tensor.numpy(), annot=True, ax=ax, fmt='g', 
                    xticklabels=class_names, yticklabels=class_names)
        
        ax.set_xlabel('Labels Prédits')
        ax.set_ylabel('Vrais Labels')
        ax.set_title('Matrice de Confusion')
        plt.tight_layout()

        save_path = save_dir / "matrice_de_confusion.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Matrice de confusion sauvegardée dans : '{save_path}'")

    def save_misclassified_as_csv(self, preds, labels, filepaths, class_map, save_dir):
        """Identifie toutes les images mal classées et les sauvegarde dans un fichier CSV."""
        print("--> Recherche des erreurs de classification pour le rapport CSV...")
        
        misclassified_records = []
        for i in range(len(preds)):
            pred_code = preds[i].item()
            true_code = labels[i].item()
            
            if pred_code != true_code:
                record = {
                    'filepath': filepaths[i],
                    'true_label': class_map.get(true_code, 'unknown'),
                    'predicted_label': class_map.get(pred_code, 'unknown')
                }
                misclassified_records.append(record)
        
        if not misclassified_records:
            print("--> Aucune image mal classée trouvée. Excellent !")
            return
            
        df = pd.DataFrame(misclassified_records)
        save_path = save_dir / "erreurs_de_classification.csv"
        df.to_csv(save_path, index=False)
        
        print(f"--> {len(df)} erreurs trouvées. Rapport sauvegardé dans : '{save_path}'")

    def copy_data_info(self, trainer: Trainer, save_dir: Path):
        """Copie le fichier data_info.yaml dans le dossier du modèle pour l'auto-suffisance."""
        source_info_path = Path(trainer.datamodule.hparams.data_root_dir) / trainer.datamodule.hparams.metadata_dir / "data_info.yaml"
        dest_info_path = save_dir / "data_info.yaml"

        if source_info_path.exists():
            shutil.copy(source_info_path, dest_info_path)
            print(f"--> Fichier d'information du dataset copié dans : '{dest_info_path}'")
        else:
            print(f"[yellow]AVERTISSEMENT: Impossible de trouver '{source_info_path}' pour le copier.[/yellow]")