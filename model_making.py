from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch
import pandas as pd

class EmotionClassification(LightningModule):
    def __init__(self):
        super(EmotionClassification, self).__init__()

        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.batch_size = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        pass

    def cross_entropy_loss(self, output, labels):
        self.parameters()
        return torch.nn.CrossEntropyLoss()(output, labels)

    def prepare_data(self):
        raw_train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_test = pd.read_csv("./data/test.txt", sep="\t", encoding="utf-8", index_col=0)

        # TODO tokenize

        self.train_set = TensorDataset(raw_train["data"], raw_train["label"])
        self.val_set = TensorDataset(raw_val["data"], raw_val["label"])
        self.test_set = TensorDataset(raw_test["data"], raw_test["label"])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        pass
