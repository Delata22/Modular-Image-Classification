# Fichier: src/models/lit_model.py (Version spécialisée pour Hugging Face)

import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from transformers import AutoModelForImageClassification, AutoConfig

class LitModel(LightningModule):
    """LightningModule générique pour n'importe quel modèle de classification d'images de Hugging Face."""
    def __init__(self, model_name: str, num_classes: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. On charge le modèle Hugging Face
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes, # On dit à HF de créer une tête pour le bon nombre de classes
            ignore_mismatched_sizes=True # Indispensable pour remplacer la tête
        ).train()

    
        # Initialisation des métriques et des listes de sortie
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # La sortie de HF est un objet, on en extrait les logits
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        # On déballe le batch (sans l'étoile)
        images, labels, filepaths = batch
        
        # On s'assure que les labels sont du bon type
        labels = labels.long()
        
        # On calcule les logits et la perte
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        # On calcule les prédictions et met à jour l'accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        
        # On loggue en spécifiant la taille du batch pour la perte
        batch_size = images.size(0)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, filepaths = batch
        labels = labels.long()
        
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        
        batch_size = images.size(0)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels, filepaths = batch
        labels = labels.long()
        
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        
        batch_size = images.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)
        
        # On stocke tout ce dont on a besoin pour le callback d'évaluation
        self.test_step_outputs.append({
            'preds': preds.cpu(), 
            'labels': labels.cpu(),
            'filepaths': filepaths
        })
        return loss

    def configure_optimizers(self):
        # L'optimiseur ne prendra en compte que les paramètres avec requires_grad=True
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer