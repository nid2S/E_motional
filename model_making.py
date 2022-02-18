import setuptools
from torch.utils.data import TensorDataset, DataLoader
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from konlpy.tag import Hannanum
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument('-hf', type=bool, default=False, dest="use_hf", help='condition of using HuggingFace Model')
args = parser.parse_args()

class EmotionClassification(LightningModule):
    def __init__(self):
        super(EmotionClassification, self).__init__()

        self.USE_HF = args.use_hf
        self.RANDOM_SEED = 7777
        self.train_set = None
        self.val_set = None
        self.test_set = None
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        pl.seed_everything(self.RANDOM_SEED)

        self.batch_size = 32
        self.input_dim = 55
        self.num_labels = 7

        self.tokenizer = Hannanum()
        self.vocab = dict([(token, index) for _, (token, index) in pd.read_csv("./data/vocab.txt", sep="\t", index_col=0, encoding="utf-8").iterrows()])
        self.id_to_token = dict([(index, token) for token, index in self.vocab.items()])
        self.vocab_size = len(self.vocab) + 1  # 76066
        self.pad_token_id = self.vocab['<pad>']  # 1

        self.fc1 = torch.nn.Linear(self.input_dim, 64)
        self.LSTM = torch.nn.LSTM(64, 32, num_layers=3)
        self.h_0 = torch.randn((self.batch_size, 32))
        self.c_0 = torch.randn((self.batch_size, 32))
        self.fc2 = torch.nn.Linear(32, self.num_labels)
        self.model = None

        if self.USE_HF:
            self.model = BertForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli", num_labels=self.num_labels)
            self.tokenizer = BertTokenizerFast.from_pretrained("Huffon/klue-roberta-base-nli")
            self.pad_token_id = self.tokenizer.pad_token_id
            self.input_dim = 125

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        if self.USE_HF:
            output = self.model(x)
            y = output.logits
        else:
            x = self.fc1(x)
            x, (h_n, c_n) = self.LSTM(x, (self.h_0, self.c_0))
            y = self.fc2(x)
        return y

    def cross_entropy_loss(self, output, labels):
        return torch.nn.CrossEntropyLoss()(output, labels, ignore_index=self.pad_token_id)

    def accuracy(self, output, labels):
        self.parameters()
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        raw_train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        raw_test = pd.read_csv("./data/test.txt", sep="\t", encoding="utf-8", index_col=0)


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
        accuracy = self.accuracy(y_pred, y)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        logs = {'val_loss': loss, 'val_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()
        logs = {'val_loss': mean_loss, 'val_acc': mean_acc}
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc, 'log': logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.cross_entropy_loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        logs = {'test_loss': loss, 'test_acc': accuracy}
        return {'loss': loss, 'accuracy': accuracy, 'log': logs}

    def test_epoch_end(self, outputs):
        mean_loss = torch.stack([output['test_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['test_acc'] for output in outputs]).mean()
        logs = {'test_loss': mean_loss, 'test_acc': mean_acc}
        return {'avg_test_loss': mean_loss, 'avg_test_acc': mean_acc, 'log': logs}

    def tokenize(self, text):
        # N - 체언 | P - 용언 | F - 외국어
        text = re.sub(r"\W", r" ", text).strip()
        text = [token for (token, tag) in self.tokenizer.pos(text) if ('N' in tag) or ('P' in tag) or ('F' in tag)]
        for i, token in enumerate(text):
            text[i] = self.vocab[token] if token in self.vocab else self.vocab['<oov>']
        text = text + [0] * (self.input_dim - len(text))
        text = text[:self.input_dim]
        return torch.FloatTensor(text)


epochs = 4 if args.use_hf else 100
dir_name = "hf_model" if args.use_hf else "pl_model"
model = EmotionClassification()
trainer = Trainer(max_epochs=epochs, gpus=torch.cuda.device_count(),
                  callbacks=[ModelCheckpoint("./model/"+dir_name+"/check_point/", verbose=True, save_top_k=3, monitor="val_acc", mode="max"),
                             EarlyStopping(monitor="val_loss", mode="min", patience=3)])

trainer.fit(model)
torch.save(model.state_dict(), "./model/" + dir_name + "/torch_model.pt")
trainer.save_checkpoint("./model/" + dir_name + "/pl_model.ptl")
