from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch import nn
import torch

class EmotionClassification(LightningModule):
    def __init__(self):
        super(EmotionClassification, self).__init__()
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        pass

    def cross_entropy_loss(self, output, labels):
        self.parameters()
        return nn.CrossEntropyLoss()(output, labels)

    def prepare_data(self):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass
