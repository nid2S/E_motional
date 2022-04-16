from torch.utils.data import Dataset
from typing import Optional, List
from hgtk.text import decompose
import torch
import pandas as pd
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt=None, style='$'))
logger.addHandler(handler)
MAX_LEN = 415

def tokenize(sent: str, device: str) -> torch.Tensor:
    vocab = dict((token, idx) for _, (token, idx) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
    sent = re.sub(" (\W)", r"\1", decompose(sent, compose_code=" "))
    is_subword = False
    encoded_sent = []
    oov_token_id = 1

    for char in sent:
        if char == " ":
            char = "[SPACE]"
            is_subword = False
        elif not is_subword:
            is_subword = True
        else:
            char = "##" + char

        try:
            encoded_sent.append(vocab[char])
        except KeyError:
            encoded_sent.append(oov_token_id)

    encoded_sent = [0] * (MAX_LEN - len(encoded_sent)) + encoded_sent
    encoded_sent = encoded_sent[:MAX_LEN]
    return torch.LongTensor(encoded_sent).to(device)

class charDataset(Dataset):
    def __init__(self, x, Y, device: Optional[str]):
        super(charDataset, self).__init__()
        self.vocab = dict((token, idx) for _, (token, idx) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8", index_col=0).iterrows())
        self.device = device
        self.pad_token_id = 0
        self.oov_token_id = 1
        self.space_token_id = 146

        self.x = [self.encoding_list(sent) for sent in x]
        self.Y = Y

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, index) -> (torch.LongTensor, torch.Tensor):
        return torch.LongTensor(self.x[index]).to(self.device), torch.scalar_tensor(self.Y[index], dtype=torch.long).to(self.device)

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

    @staticmethod
    def prepare_data():
        train_data = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8", index_col=0)
        train_data["data"] = train_data["data"].apply(lambda x: re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower())
                                                      if re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower()) != "" else None)
        train_data.dropna(inplace=True)

        val_data = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8", index_col=0)
        val_data["data"] = val_data["data"].apply(lambda x: re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower())
                                                  if re.sub("[^가-힣a-z0-9.,?! ]", "", x.lower()) != "" else None)
        val_data.dropna(inplace=True)

        return train_data, val_data
