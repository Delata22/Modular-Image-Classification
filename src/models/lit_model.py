# File: src/models/lit_model.py (Specialized version for Hugging Face)

import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import Accuracy, F1Score
from transformers import AutoModelForImageClassification, AutoConfig

class LitModel(LightningModule):
    """Generic LightningModule for any image classification model from Hugging Face."""
    def __init__(self, model_name: str, num_classes: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. Load the Hugging Face model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes, # Tell HF to create a head for the correct number of classes
            ignore_mismatched_sizes=True # Essential for replacing the head
        ).train()

        
        # Initialize metrics and output lists
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The HF output is an object, we extract the logits from it
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        # Unpack the batch (without the star)
        images, labels, filepaths = batch
        
        # Ensure labels are of the correct type
        labels = labels.long()
        
        # Compute logits and loss
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        
        # Compute predictions and update accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        
        # Log by specifying the batch size for the loss
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
        
        # Store everything needed for the evaluation callback
        self.test_step_outputs.append({
            'preds': preds.cpu(), 
            'labels': labels.cpu(),
            'filepaths': filepaths
        })
        return loss

    def configure_optimizers(self):
        # The optimizer will only consider parameters with requires_grad=True
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer