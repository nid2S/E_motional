import setuptools
import re
import os
import math
import torch
import logging
import argparse
import pandas as pd
from torch.functional import F
from hgtk.text import decompose
from typing import List, Optional
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.mobile_optimizer import optimize_for_mobile

MAX_LEN = 415  # train - 415, val - 364, test - 289
VOCAB_SIZE = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 7777
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class charDataset(torch.utils.data.Dataset):
    def __init__(self, x: List[str], Y: List[int]):
        super(charDataset, self).__init__()
        self.vocab = dict((token, idx) for _, (token, idx) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
        self.pad_token_id = 0
        self.oov_token_id = 1
        self.space_token_id = 146

        self.x = [self.encoding_list(sent) for sent in x]
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index) -> (torch.LongTensor, torch.Tensor):
        return torch.LongTensor(self.x[index]).to(DEVICE), torch.scalar_tensor(self.Y[index], dtype=torch.long).to(DEVICE)

    def encoding_list(self, sent: str) -> List[int]:
        sent = re.sub(" (\W)", r"\1", decompose(sent, compose_code=" "))
        is_subword = False
        encoded_sent = []

        for char in sent:
            if char == " ":
                char = "[SPACE]"
                is_subword = False
            elif not is_subword:
                is_subword = True
            else:
                char = "##" + char

            try:
                encoded_sent.append(self.vocab[char])
            except KeyError:
                encoded_sent.append(self.oov_token_id)

        encoded_sent = [0] * (MAX_LEN - len(encoded_sent)) + encoded_sent
        encoded_sent = encoded_sent[:MAX_LEN]
        return encoded_sent

class PositionalEncoding(torch.nn.Module):
    def __init__(self, input_dim, model_dim, dropout_rate, device: Optional[str]):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        if device is None:
            device = DEVICE

        pos_encoding = torch.zeros(input_dim, model_dim)
        position_list = torch.arange(input_dim, dtype=torch.float).view(-1, 1)  # == unsqueez(1), shape -> (input_dim, 1)
        division_term = torch.exp(torch.arange(0, model_dim, 2).float()*(-math.log(10000)/model_dim))

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(position_list * division_term)
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(position_list * division_term)
        self.pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1).to(device)
        try:
            self.register_buffer("pos_encoding", pos_encoding)
        except KeyError:
            pass

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.dropout(x + self.pos_encoding[:x.size(0)])

class EmotionClassifier(torch.nn.Module):
    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 **kwargs):
        super(EmotionClassifier, self).__init__()
        num_labels = 7
        pad_token_id = 0
        input_dim = MAX_LEN
        vocab_size = VOCAB_SIZE

        self.embeddingLayer = torch.nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id, device=DEVICE)
        self.linearLayer = torch.nn.Linear(embedding_size, hidden_size, device=DEVICE)
        encoder_layer = torch.nn.TransformerEncoderLayer(hidden_size, num_heads, dropout=dropout_rate, device=DEVICE,
                                                         dim_feedforward=2048, activation=F.gelu, batch_first=True)
        self.transformerEncoder = torch.nn.Sequential(
            PositionalEncoding(input_dim, hidden_size, dropout_rate, device=DEVICE),
            torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=torch.nn.LayerNorm(hidden_size, device=DEVICE))
        )
        self.outputLayer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, num_labels, device=DEVICE),
            torch.nn.Softmax(dim=-1)
        )
        torch.nn.init.xavier_uniform_(self.linearLayer.weight)

    def forward(self, x):
        x = self.embeddingLayer(x)
        x = torch.mean(x, dim=1)
        x = self.linearLayer(x)
        x = self.transformerEncoder(x)
        output = self.outputLayer(x)
        return output

    def tokenize(self, sent: str) -> torch.Tensor:
        vocab = dict((token, idx) for _, (token, idx) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
        oov_token_id = 1
        is_subword = False

        encoded_sent = []
        sent = re.sub("[^가-힣a-z0-9.,?!]", "", sent.lower())
        sent = re.sub(" (\W)", r"\1", decompose(sent, compose_code=" "))
        for char in sent:
            if char == " ":
                char = "[SPACE]"
                is_subword = False
            elif not is_subword:
                is_subword = True
            else:
                char = "##"+char

            try:
                encoded_sent.append(vocab[char])
            except KeyError:
                encoded_sent.append(oov_token_id)
        return torch.LongTensor(encoded_sent).to(DEVICE)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')  # for using GPU
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)

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
    num_worker = os.cpu_count() if DEVICE == "cpu" else torch.cuda.device_count()
    num_worker = num_worker if DEVICE == "cpu" or num_worker <= 2 else 2
    pad_token_id = 0
    num_labels = 7

    # prepare datasets
    train_data = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
    train_data["data"] = train_data["data"].apply(lambda x: re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower())
                                                  if re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower()) != "" else None)
    train_data.dropna(inplace=True)
    val_data = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
    val_data["data"] = val_data["data"].apply(lambda x: re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower())
                                              if re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower()) != "" else None)
    val_data.dropna(inplace=True)

    # define model, optim, lr_scehduler, accuracy
    args = parser.parse_args()
    model = EmotionClassifier(**args.__dict__)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.dropout_rate)
    lr_scheduler = ExponentialLR(optim, gamma=args.gamma)
    accuracy = Accuracy(num_classes=num_labels, ignore_index=pad_token_id).to(DEVICE)
    # make dataloader
    train_set = DataLoader(charDataset(train_data["data"].to_list(), train_data["label"].to_list()),
                           batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)
    val_set = DataLoader(charDataset(val_data["data"].to_list(), val_data["label"].to_list()),
                         batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=num_worker)

    last_metric = 0
    best_metric = 0
    patience_cnt = 0
    logger.info(model)
    logger.info("start Training.")
    for i in range(args.epochs):
        # train step
        for j, (train_x, train_Y) in enumerate(train_set):
            optim.zero_grad()
            pred = model(train_x)
            loss = F.cross_entropy(pred, train_Y, ignore_index=pad_token_id)
            acc = accuracy(torch.argmax(pred, dim=1), train_Y)

            if j % 10 == 0:
                logger.info(f"Epoch {i} - loss : %.4f, acc : %.2f | progress : {j}/{len(train_set)}" % (float(loss), acc))
                logger.debug(f"pred : {torch.argmax(pred, dim=1)}")

            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            loss.backward()
            optim.step()

        # validation step
        loss_list = []
        acc_list = []
        for j, (val_x, val_Y) in enumerate(val_set):
            with torch.no_grad():
                pred = model(val_x)
                loss = F.cross_entropy(pred, val_Y)
                acc = accuracy(torch.argmax(pred, dim=1), val_Y)

                loss_list.append(loss)
                acc_list.append(acc)
                if j % 10 == 0:
                    logger.info(f"Epoch {i} - val_loss : %.4f, val_acc : %.2f | progress : {j}/{len(val_set)}" % (float(loss), acc))
                    logger.debug(f"pred : {torch.argmax(pred, dim=1)}")
        logger.info(f"Epoch {i} - avg_val_loss : %.4f, avg_val_acc : %.2f" % (sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)))

        # Ealry Stopping
        avg_val_acc = sum(acc_list)/len(acc_list)
        if last_metric > avg_val_acc:
            patience_cnt += 1
            if patience_cnt > args.patience:
                logger.info(f"metrics was not improved at {args.patience} times. stop training.")
                break
        else:
            logger.info("metrics was improved.")
            # ModelCheckpoint(SaveBestOnly)
            if best_metric < avg_val_acc:
                best_metric = avg_val_acc
                logger.info(f"avg_val_acc has achived to best({avg_val_acc}). save model state.")
                torch.save(model.state_dict(), "./model/best_state.pt")
            patience_cnt = 0
    torch.save(model.state_dict(), "./model/emotion_classifier_state.pt")

    example_input = model.tokenize("이건 트레이싱을 위한 예시 입력입니다.")
    model = torch.quantization.convert(model)
    model = torch.jit.trace(model, example_input, strict=False)
    opt_model = optimize_for_mobile(model)
    opt_model._save_for_lite_interpreter("./model/emotion_classifier.ptl")
