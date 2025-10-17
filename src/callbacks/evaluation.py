# File: src/callbacks/evaluation.py

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
    Callback that activates at the end of the test phase to generate:
    1. An image of the confusion matrix.
    2. A CSV file listing all misclassified images.
    3. A copy of the dataset's information file.
    """

    @rank_zero_only  # Ensures this code only runs on the main process
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        print("\n--> Generating final evaluation report...")
        
        outputs = pl_module.test_step_outputs
        if not outputs:
            print("No test outputs to analyze.")
            return

        # 1. Gather all predictions, labels, and filepaths
        preds = torch.cat([x['preds'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        filepaths = [path for batch_output in outputs for path in batch_output['filepaths']]
        
        # 2. Retrieve useful information from the Trainer and DataModule
        class_map = trainer.datamodule.class_map
        class_names = [class_map.get(i, str(i)) for i in range(pl_module.hparams.num_classes)]
        save_dir = Path(trainer.logger.log_dir).parent / "train_info"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 3. Call the functions to generate reports
        self.plot_confusion_matrix(preds, labels, class_names, save_dir)
        self.save_misclassified_as_csv(preds, labels, filepaths, class_map, save_dir)
        self.copy_data_info(trainer, save_dir)
        
        print(f"Your model has been saved in the '{str(save_dir.parent)}' directory")
        # 4. Clear memory for the next run
        pl_module.test_step_outputs.clear()

    def plot_confusion_matrix(self, preds, labels, class_names, save_dir):
        """Calculates and saves the confusion matrix."""
        print("--> Generating confusion matrix...")
        confmat = ConfusionMatrix(task="multiclass", num_classes=len(class_names))
        cm_tensor = confmat(preds.cpu(), labels.cpu())

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(cm_tensor.numpy(), annot=True, ax=ax, fmt='g', 
                    xticklabels=class_names, yticklabels=class_names)
        
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()

        save_path = save_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to: '{save_path}'")

    def save_misclassified_as_csv(self, preds, labels, filepaths, class_map, save_dir):
        """Identifies all misclassified images and saves them to a CSV file."""
        print("--> Searching for classification errors for the CSV report...")
        
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
            print("--> No misclassified images found. Excellent!")
            return
            
        df = pd.DataFrame(misclassified_records)
        save_path = save_dir / "classification_errors.csv"
        df.to_csv(save_path, index=False)
        
        print(f"--> Found {len(df)} errors. Report saved to: '{save_path}'")

    def copy_data_info(self, trainer: Trainer, save_dir: Path):
        """Copies the data_info.yaml file into the model's directory for self-sufficiency."""
        source_info_path = Path(trainer.datamodule.hparams.data_root_dir) / trainer.datamodule.hparams.metadata_dir / "data_info.yaml"
        dest_info_path = save_dir / "data_info.yaml"

        if source_info_path.exists():
            shutil.copy(source_info_path, dest_info_path)
            print(f"--> Dataset information file copied to: '{dest_info_path}'")
        else:
            print(f"[yellow]WARNING: Could not find '{source_info_path}' to copy.[/yellow]")