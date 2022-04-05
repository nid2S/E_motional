import setuptools
import re
import os
import math
import torch
import argparse
import pandas as pd
import pytorch_lightning as pl
from torch.functional import F
from hgtk.text import decompose
from typing import List, Tuple, Optional
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.mobile_optimizer import optimize_for_mobile
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

MAX_LEN = 415  # train - 415, val - 364, test - 289
VOCAB_SIZE = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class charDataset(torch.utils.data.Dataset):
    def __init__(self, x: List[str], Y: List[int]):
        super(charDataset, self).__init__()
        self.vocab = dict((token, id) for _, (token, id) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
        self.pad_token_id = 0
        self.oov_token_id = 1
        self.space_token_id = 146

        self.x = x
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.Tensor]:
        x = self.encoding_list(self.x[index])
        return torch.LongTensor(x).to(DEVICE), torch.scalar_tensor(self.Y[index], dtype=torch.long).to(DEVICE)

    def encoding_list(self, sent: str) -> List[int]:
        result_list = []
        for word in decompose(sent, compose_code="_").split():
            for char in word.split("_"):
                for i, c in enumerate(list(char)):
                    try:
                        if i == 0:
                            result_list.append(self.vocab[c])
                        else:
                            result_list.append(self.vocab["##" + c])
                    except KeyError:
                        result_list.append(self.oov_token_id)
            result_list.append(self.space_token_id)

        result_list = [0] * (MAX_LEN - len(result_list)) + result_list
        result_list = result_list[:MAX_LEN]
        return result_list

class PositionalEncoding(torch.nn.Module):
    def __init__(self, input_dim, model_dim, dropout_rate):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)

        pos_encoding = torch.zeros(input_dim, model_dim)
        position_list = torch.arange(input_dim, dtype=torch.float).view(-1, 1)  # == unsqueez(1), shape -> (input_dim, 1)
        division_term = torch.exp(torch.arange(0, model_dim, 2).float()*(-math.log(10000)/model_dim))

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position_list * division_term)
        self.pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1).to(DEVICE)
        try:
            self.register_buffer("pos_encoding", pos_encoding)
        except KeyError:
            pass

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])

class EmotionClassifier(LightningModule):
    def __init__(self, hparams: Optional[argparse.Namespace]):
        super(EmotionClassifier, self).__init__()

        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.pad_token_id = 0
        if hparams is None:
            self.batch_size = 32
            self.embedding_size = 512
            self.hidden_size = 256
            self.num_heads = 8
            self.num_layers = 6
            self.dropput_rate = 0.1
            self.gamma = 0.9
            self.patience = 5
            self.lr = 0.001
        else:
            self.batch_size = hparams.batch_size
            self.embedding_size = hparams.embedding_size
            self.hidden_size = hparams.hidden_size
            self.num_heads = hparams.num_heads
            self.num_layers = hparams.num_layers
            self.dropput_rate = hparams.dropout_rate
            self.gamma = hparams.gamma
            self.patience = hparams.patience
            self.lr = hparams.lr
        self.num_labels = 7
        self.input_dim = MAX_LEN
        self.vocab_size = VOCAB_SIZE
        self.num_worker = os.cpu_count() if DEVICE == "cpu" else torch.cuda.device_count()

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=self.pad_token_id)
        encoder_layer = torch.nn.TransformerEncoderLayer(self.embedding_size, self.num_heads, dropout=self.dropput_rate, device=DEVICE,
                                                         dim_feedforward=2048, activation=F.elu, batch_first=True)
        self.transformer_encoder = torch.nn.Sequential(
            PositionalEncoding(self.input_dim, self.embedding_size, self.dropput_rate),
            torch.nn.TransformerEncoder(encoder_layer, self.num_layers, norm=torch.nn.LayerNorm(self.embedding_size))
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.hidden_size, device=DEVICE),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.hidden_size//2, device=DEVICE),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size//2, self.hidden_size, device=DEVICE),
            torch.nn.ELU(),
            torch.nn.LayerNorm(self.hidden_size),
            torch.nn.Linear(self.hidden_size, self.num_labels),
            torch.nn.Softmax(dim=-1)
        )

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), self.lr, weight_decay=0.1)
        optim_N = torch.optim.NAdam(self.parameters(), self.lr, weight_decay=0.1, momentum_decay=0.1)
        lr_scheduler = ExponentialLR(optim, gamma=self.gamma)
        lr_scheduler_N = ExponentialLR(optim_N, gamma=self.gamma)

        return [optim, optim_N], [lr_scheduler, lr_scheduler_N]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint("./model/model_ckp/", monitor="val_acc", save_last=True)
        early_stopping = EarlyStopping(monitor="val_acc", patience=self.patience)
        lr_monitor = LearningRateMonitor(log_momentum=True)

        return [model_checkpoint, early_stopping, lr_monitor]

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer_encoder(x)
        output = self.output_layer(x)
        return torch.mean(output, dim=1)

    def loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(output, labels)

    def accuracy(self, output, labels):
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        train["data"] = train["data"].apply(lambda x: re.sub("[^가-힣0-9a-z.,?!]", "", x.lower()))
        train["data"] = train["data"].apply(lambda x: x if x != "" else None)
        train.dropna(inplace=True)

        val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        val["data"] = val["data"].apply(lambda x: re.sub("[^가-힣0-9a-z.,?!]", "", x.lower()))
        val["data"] = val["data"].apply(lambda x: x if x != "" else None)
        val.dropna(inplace=True)

        test = pd.read_csv("./data/test.txt", sep="\t", encoding="utf-8", index_col=0)
        test["data"] = test["data"].apply(lambda x: re.sub("[^가-힣0-9a-z.,?!]", "", x.lower()))
        test["data"] = test["data"].apply(lambda x: x if x != "" else None)
        test.dropna(inplace=True)

        self.train_set = charDataset(train["data"].to_list(), train["label"].to_list())
        self.val_set = charDataset(val["data"].to_list(), val["label"].to_list())
        self.test_set = charDataset(test["data"].to_list(), test["label"].to_list())

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=True, num_workers=8)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log('loss', loss)
        self.log('acc', accuracy, prog_bar=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        accuracy = self.accuracy(y_pred, y)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}

    def predict(self, x):
        model.eval()
        encoded_x = self.tokenize(x).to(DEVICE)
        with torch.no_grad:
            pred = self(encoded_x)
        return torch.argmax(pred, dim=1)

    def tokenize(self, sent: str) -> torch.Tensor:
        vocab = dict((token, id) for _, (token, id) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
        oov_token_id = 1
        space_token_id = 146

        encoded_sent = []
        for word in decompose(sent, "_").split():
            for char in word.split("_"):
                for i, c in enumerate(list(char)):
                    try:
                        if i == 0:
                            encoded_sent.append(vocab[c])
                        else:
                            encoded_sent.append(vocab["##" + c])
                    except KeyError:
                        encoded_sent.append(oov_token_id)
            encoded_sent.append(space_token_id)
        return torch.LongTensor(encoded_sent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, default=50, dest="epochs", help="epochs")
    parser.add_argument("-batch_size", type=int, default=32, dest="batch_size", help="batch_size")
    parser.add_argument("-lr", type=int, default=0.001, dest="lr", help="learning rate")
    parser.add_argument("-embedding-size", type=int, default=512, dest="embedding_size", help="size of embedding layer")
    parser.add_argument("-hidden_size", type=int, default=256, dest="hidden_size", help="size of hidden layer")
    parser.add_argument("-num-heads", type=int, default=8, dest="num_heads", help="num of attention heads")
    parser.add_argument("-num-layers", type=int, default=6, dest="num_layers", help="num of attention layers")
    parser.add_argument("-dropout-rate", type=int, default=0.1, dest="dropout_rate", help="rate of dropout")
    parser.add_argument("-gamma", type=int, default=0.9, dest="gamma", help="rate of multiplied with lr for each epoch")
    parser.add_argument("-patience", type=int, default=5, dest="patience", help="num of times monitoring metric can be reduced")

    args = parser.parse_args()
    model = EmotionClassifier(args)
    trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(), logger=TensorBoardLogger("./model/tensorboardLog/"))
    trainer.fit(model)
    torch.save(model.state_dict(), "./model/emotion_classifier_state.pt")

    example_input = model.tokenize("이건 트레이싱을 위한 예시 입력입니다.")
    model = torch.quantization.convert(model)
    model = torch.jit.trace(model, example_input, strict=False)
    opt_model = optimize_for_mobile(model)
    opt_model._save_for_lite_interpreter("./model/emotion_classifier.ptl")
