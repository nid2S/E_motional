import setuptools
from torch.utils.data import TensorDataset, DataLoader
import torch
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from konlpy.tag import Hannanum
import pandas as pd
import re

class EmotionClassification(LightningModule):
    def __init__(self):
        super(EmotionClassification, self).__init__()

        self.RANDOM_SEED = 7777
        self.train_set = None
        self.val_set = None
        self.test_set = None
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)

        self.batch_size = 32
        self.input_dim = 55
        self.num_labels = 7

        self.tokenizer = Hannanum()
        self.vocab = dict([(key, index) for _, (key, index) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8").iterrows()])
        self.id_to_token = dict([(index, key) for key, index in self.vocab.items()])
        self.vocab_size = len(self.vocab) + 1  # 76066
        self.pad_token_id = self.vocab['<pad>']  # 0

        self.fc1 = torch.nn.Linear(self.input_dim, 64)
        self.LSTM = torch.nn.LSTM((self.batch_size, 64), 32, num_layers=3, batch_first=True)
        self.h_0 = torch.randn((self.batch_size, 32))
        self.fc2 = torch.nn.Linear(32, self.num_labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = self.fc1(x)
        x = self.LSTM(x, self.h_0)
        y = self.fc2(x)
        return y

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels, ignore_index=self.pad_token_id)

    def accuracy(self, output, labels):
        pass

    def prepare_data(self):
        raw_train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_test = pd.read_csv("./data/test.txt", sep="\t", encoding="utf-8", index_col=0)

        raw_train["data"] = raw_train["data"].apply(lambda x: self.tokenize(x))
        raw_train["label"] = raw_train["label"].apply(lambda x: torch.LongTensor([x]))
        raw_val["data"] = raw_val["data"].apply(lambda x: self.tokenize(x))
        raw_val["label"] = raw_val["label"].apply(lambda x: torch.LongTensor([x]))
        raw_test["data"] = raw_test["data"].apply(lambda x: self.tokenize(x))
        raw_test["label"] = raw_test["label"].apply(lambda x: torch.LongTensor([x]))

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
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        logs = {'val_loss': mean_loss}
        return {'avg_val_loss': mean_loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        logs = {'test_loss': mean_loss}
        return {'avg_test_loss': mean_loss, 'log': logs}

    def tokenize(self, text):
        # N - 체언 | P - 용언 | F - 외국어
        text = re.sub(r"\W", r" ", text).strip()
        text = [token for (token, tag) in self.tokenizer.pos(text) if ('N' in tag) or ('P' in tag) or ('F' in tag)]
        for i, token in enumerate(text):
            text[i] = self.vocab[token] if token in self.vocab else self.vocab['<oov>']
        text = [text + [0] * (self.input_dim - len(text))][:self.input_dim]
        return torch.FloatTensor(text)
