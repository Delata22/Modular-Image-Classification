import torch
from lightning import LightningModule
from torch.nn import functional as F
from torchmetrics import Accuracy
from transformers import AutoModelForImageClassification, AutoConfig

class LitMNISTModel(LightningModule):

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", num_classes: int = 10, lr: float = 0.001):
        super().__init__()

        self.save_hyperparameters()

        config = AutoConfig.from_pretrained(model_name)

        self.model = AutoModelForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)

        in_features = self.model.classifier.in_features
        self.model.classifier = torch.nn.Linear(in_features, num_classes)

        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        out = self.model(x).logits
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long() 

        logits = self(images)

        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long() 

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.long() 
        
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)

        # On loggue les métriques de test
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer