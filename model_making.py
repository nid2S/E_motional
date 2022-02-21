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
        self.RANDOM_SEED = 7777
        torch.manual_seed(self.RANDOM_SEED)
        torch.cuda.manual_seed(self.RANDOM_SEED)
        pl.seed_everything(self.RANDOM_SEED)

        self.USE_HF = args.use_hf
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.learning_rate = 0.1
        self.num_layers = 2
        self.num_labels = 7
        self.batch_size = 32
        self.input_dim = 55
        self.embedding_size = 128
        self.hidden_size = 64

        self.tokenizer = Hannanum()
        self.vocab = dict([(token, index) for _, (token, index) in pd.read_csv("./data/vocab.txt", sep="\t", index_col=0, encoding="utf-8").iterrows()])
        self.id_to_token = dict([(index, token) for token, index in self.vocab.items()])
        self.vocab_size = len(self.vocab) + 1
        self.pad_token_id = self.vocab['<pad>']  # 0

        self.embedding = torch.nn.Embedding(self.input_dim, self.embedding_size, padding_idx=self.pad_token_id)
        self.LSTM = torch.nn.LSTM(self.embedding_size, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.3)
        self.fc = torch.nn.Linear(self.hidden_size, self.num_labels)
        self.model = None

        if self.USE_HF:
            self.model = BertForSequenceClassification.from_pretrained("Huffon/klue-roberta-base-nli", num_labels=self.num_labels)
            self.tokenizer = BertTokenizerFast.from_pretrained("Huffon/klue-roberta-base-nli")
            self.pad_token_id = self.tokenizer.pad_token_id
            self.learning_rate = 3e-5
            self.input_dim = 125

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        if self.USE_HF:
            output = self.model(x)
            y = output.logits
        else:
            x = self.embedding(x)
            h_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, requires_grad=True)
            c_0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size, requires_grad=True)
            x, (h_n, c_n) = self.LSTM(x, (h_0, c_0))
            y = self.fc(x)
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

        train_Y = torch.LongTensor(raw_train["label"].to_list())
        val_Y = torch.LongTensor(raw_val["label"].to_list())
        test_Y = torch.LongTensor(raw_test["label"].to_list())

        if self.USE_HF:
            train_x = self.tokenizer.batch_encode_plus(raw_train["data"].to_list(), return_tensors="pt",
                                                       max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
            val_x = self.tokenizer.batch_encode_plus(raw_val["data"].to_list(), return_tensors="pt",
                                                     max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
            test_x = self.tokenizer.batch_encode_plus(raw_test["data"].to_list(), return_tensors="pt",
                                                      max_length=self.max_len, padding="max_length", truncation=True)["input_ids"]
        else:
            train_x = torch.FloatTensor(raw_train["data"].apply(lambda x: self.tokenize(x, return_tensor=False)).to_list())
            val_x = torch.FloatTensor(raw_val["data"].apply(lambda x: self.tokenize(x, return_tensor=False)).to_list())
            test_x = torch.FloatTensor(raw_test["data"].apply(lambda x: self.tokenize(x, return_tensor=False)).to_list())

        self.train_set = TensorDataset(train_x, train_Y)
        self.val_set = TensorDataset(val_x, val_Y)
        self.test_set = TensorDataset(test_x, test_Y)

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

    def tokenize(self, text, return_tensor: bool = True):
        if self.USE_HF:
            self.tokenizer.encode(text, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="pt")
        else:
            # N - 체언 | P - 용언 | F - 외국어
            text = re.sub(r"\W", r" ", text).strip()
            text = [token for (token, tag) in self.tokenizer.pos(text) if ('N' in tag) or ('P' in tag) or ('F' in tag)]
            for i, token in enumerate(text):
                text[i] = self.vocab[token] if token in self.vocab else self.vocab['<oov>']
            text = text + [self.pad_token_id] * (self.input_dim - len(text))
            text = text[:self.input_dim]
            return torch.FloatTensor(text) if return_tensor else text


epochs = 4 if args.use_hf else 100
dir_name = "hf_model" if args.use_hf else "pl_model"
model = EmotionClassification()
trainer = Trainer(max_epochs=epochs, gpus=torch.cuda.device_count(),
                  callbacks=[ModelCheckpoint("./model/"+dir_name+"/check_point/", verbose=True, save_top_k=3, monitor="val_acc", mode="max"),
                             EarlyStopping(monitor="val_loss", mode="min", patience=3)])

trainer.fit(model)
torch.save(model.state_dict(), "./model/" + dir_name + "/torch_model.pt")
trainer.save_checkpoint("./model/" + dir_name + "/pl_model.ptl")
