import setuptools
import torch
import re
import math
import argparse
import pandas as pd
import pytorch_lightning as pl
from torch.functional import F
from konlpy.tag import Okt
from torch.utils import mobile_optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import MobileBertForSequenceClassification, MobileBertTokenizerFast

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=50, dest="epochs", help="num of epochs")
parser.add_argument("-b", type=int, default=32, dest="batch_size", help="size of each batch")
parser.add_argument("-hd", type=int, default=256, dest="hidden_size", help="size of hidden_state")
parser.add_argument("-l", type=int, default=6, dest="num_layers", help="num of transfomer model encoder layers")
parser.add_argument("-p", type=int, default=5, dest="patience", help="number of check with no improved")
parser.add_argument("-lr", type=float, default=0.1, dest="learning_rate", help="learning rate")
parser.add_argument("-dr", type=float, default=0.1, dest="dropout_rate", help="dropout rate")
parser.add_argument("-gamma", type=float, default=0.9, dest="gamma", help="decay rate of learning_rate on each epoch")
parser.add_argument("-train", type=bool, default=False, dest="is_train", help="is_train")
parser.add_argument("--embedding-size", type=int, default=512, dest="embedding_size", help="size of embedding vector")
parser.add_argument("--rnn-layer", type=int, default=2, dest="rnn_layers", help="rnn layers")

class PositionalEncoding(torch.nn.Module):
    def __init__(self, model_dim, input_dim, dropout_rate):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        pos_encoding = torch.zeros(input_dim, model_dim)
        position_list = torch.arange(input_dim, dtype=torch.float).view(-1, 1)  # unsqueez(1)
        division_term = torch.exp(torch.arange(0, model_dim, 2).float()*(-math.log(10000)/model_dim))
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])

class EmotionClassifier(LightningModule):
    def __init__(self, hparams):
        super(EmotionClassifier, self).__init__()
        self.RANDOM_SEED = 7777
        pl.seed_everything(self.RANDOM_SEED)

        self.epochs = hparams.epochs
        self.batch_size = hparams.batch_size
        self.embedding_size = hparams.embedding_size
        self.hidden_size = hparams.hidden_size
        self.num_layers = hparams.num_layers
        self.rnn_layers = hparams.rnn_layers
        self.patience = hparams.patience
        self.gamma = hparams.gamma
        self.learning_rate = hparams.learning_rate
        self.dropout_rate = hparams.dropout_rate
        self.is_train = hparams.is_train

        self.num_labels = 7
        # self.input_dim = 125  # train-125, val-107, test-91
        self.input_dim = 320  # train-311, val-275, test-209

        self.label_dict = {0: "기쁨", 1: "분노", 2: "슬픔", 3: "불안", 4: "놀람", 5: "혐오", 6: "중립"}
        self.train_set = None  # 78262
        self.val_set = None  # 32020
        self.test_set = None  # 12254

        self.model = MobileBertForSequenceClassification.from_pretrained("google/mobilebert-uncased", num_labels=self.num_labels).to(self.device)
        self.tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        # self.tokenizer = Okt()
        # self.vocab = dict((token, index) for token, index in pd.read_csv('./data/vocab.txt', sep="\t", encoding="utf-8", index_col=0).values)
        # self.pad_token_id = self.vocab['<pad>']
        #
        # self.embedding_layer = torch.nn.Sequential(
        #     torch.nn.Embedding(len(self.vocab), self.embedding_size, self.pad_token_id),
        #     torch.nn.LayerNorm(self.embedding_size, eps=1e-5)
        # )
        # self.gru_layer = torch.nn.GRU(self.embedding_size, self.hidden_size, batch_first=True, num_layers=self.rnn_layers, dropout=self.dropout_rate)
        # self.output_layer = torch.nn.Sequential(
        #     torch.nn.Linear(self.hidden_size, self.hidden_size * 2),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(self.hidden_size * 2, self.hidden_size),
        #     torch.nn.Tanh(),
        #     torch.nn.LayerNorm(self.hidden_size, eps=1e-5),
        #     torch.nn.Linear(self.hidden_size, self.num_labels)
        # )

        if self.is_train:
            self.model.training = True
            for params in self.model.parameters():
                params.require_grad = True

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.1)

        lr_scheduler = ExponentialLR(optim, gamma=self.gamma)
        return [optim], [lr_scheduler]

    def configure_callbacks(self):
        model_checkpoint = ModelCheckpoint(dirpath="./model/model_ckp/", filename='{epoch:02d}_{loss:.2f}', verbose=True, save_last=True, monitor='val_loss', mode='min')
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=self.patience)
        lr_monitor = LearningRateMonitor(logging_interval="step")
        return [model_checkpoint, early_stopping, lr_monitor]

    def forward(self, x):
        # x = self.embedding_layer(x)
        # h_0 = torch.zeros(self.rnn_layers, self.batch_size, self.hidden_size).to(self.device)
        # x, h = self.gru_layer(x, h_0)
        # output = self.output_layer(x)
        #
        # output = torch.sum(output, 1)
        # output = F.softmax(output, 1)
        output = self.model(x)
        return output.logits

    def loss(self, output, labels):
        return torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)(output, labels)

    def accuracy(self, output, labels):
        output = torch.argmax(output, dim=1)
        return torch.sum(output == labels) / output.__len__() * 100  # %(Precentage)

    def prepare_data(self):
        train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        test = pd.read_csv("./data/test.txt", sep="\t", encoding="utf-8", index_col=0)
        data_list = []
        for data in [train, val, test]:
            # x = []
            # for sent in data['data'].values:
            #     sent = re.sub(r"[^가-힣ㄱ-ㅎa-zA-z0-9.,?! ]", "", sent).strip()
            #     temp_x = []
            #     for token in self.tokenizer.morphs(sent, norm=True, stem=True):
            #         if token in self.vocab.keys():
            #             temp_x.append(self.vocab[token])
            #         else:
            #             temp_x.append(self.vocab['oov'])
            #     temp_x = temp_x + [self.pad_token_id] * (self.input_dim - len(temp_x))
            #     temp_x = temp_x[:self.input_dim]
            #     x.append(temp_x)
            # x = torch.LongTensor(x).to(self.device)
            # Y = torch.LongTensor(data['label']).to(self.device)
            # data_list.append((x, Y))
            x = self.tokenizer.batch_encode_plus(data['data'].to_list(), max_length=self.input_dim, padding="max_length", return_tensors="pt").to(self.device)
            Y = torch.LongTensor(data['label']).to(self.device)
            data_list.append((x["input_ids"], x["token_type_ids"], x["attention_mask"], Y))
        self.train_set = TensorDataset(*data_list[0])
        self.val_set = TensorDataset(*data_list[1])
        self.test_set = TensorDataset(*data_list[2])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # y_pred = self(x)
        # loss = self.loss(y_pred, y)
        # accuracy = self.accuracy(y_pred, y)

        input_ids, token_type_ids, attention_mask, labels = batch
        output = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        accuracy = self.accuracy(output.logits, labels)

        self.log('loss', loss, on_step=True)
        self.log('acc', accuracy, prog_bar=True, on_step=True)
        return {'loss': loss, 'acc': accuracy}

    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_pred = self(x)
        # loss = self.loss(y_pred, y)
        # accuracy = self.accuracy(y_pred, y)

        input_ids, token_type_ids, attention_mask, labels = batch
        output = self(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels).logits
        loss = output.loss
        accuracy = self.accuracy(output.logits, labels)

        self.log_dict({'val_loss': loss, 'val_acc': accuracy}, prog_bar=True)
        return {'val_loss': loss, 'val_acc': accuracy}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        mean_acc = torch.stack([output['val_acc'] for output in outputs]).mean()

        self.log_dict({'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}, on_epoch=True, prog_bar=True)
        return {'avg_val_loss': mean_loss, 'avg_val_acc': mean_acc}

    def tokenize(self, sent: str) -> torch.FloatTensor:
        # sent = re.sub(r"[^가-힣ㄱ-ㅎa-zA-z0-9.,?! ]", "", sent).strip()
        # temp_x = []
        # for token in self.tokenizer.morphs(sent, norm=True, stem=True):
        #     if token in self.vocab.keys():
        #         temp_x.append(self.vocab[token])
        #     else:
        #         temp_x.append(self.vocab['oov'])
        # sent = temp_x + [self.pad_token_id] * (self.input_dim - len(sent))
        # sent = sent[:self.input_dim]
        # return torch.FloatTensor(sent).to(self.device)
        return self.tokenizer.encode(sent, max_length=self.input_dim, padding="max_length", return_tensors="pt")


args = parser.parse_args()
model = EmotionClassifier(args)
trainer = Trainer(max_epochs=args.epochs, gpus=torch.cuda.device_count(), logger=TensorBoardLogger("./model/tensorboardLog/"))
trainer.fit(model)

torch.save(model.state_dict(), "./model/model_state.pt")
example_input = model.tokenize("이건 트레이싱을 위한 예시 입력입니다.")
model = torch.quantization.convert(model)
model = torch.jit.trace(model, example_input)
opt_model = mobile_optimizer.optimize_for_mobile(model)
opt_model.save_for_lite_interpreter("./model/label_classifier.ptl")
