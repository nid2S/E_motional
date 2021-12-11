from transformers import BertTokenizerFast
from konlpy.tag import Hannanum
from typing import Union, List
import tensorflow as tf
import pandas as pd
import random
import json
import re
import os

def make_dataset():
    # train - 86958, val - 35578 | sentimental => T - 40827, V - 5122 | multimodal => T - 46131, V - 30456
    train = pd.DataFrame(columns=["data", "label"])
    val = pd.DataFrame(columns=["data", "label"])
    hist = []

    print('making label dict')
    all_label = pd.read_csv("./data/label.txt", encoding="utf-8", names=["label", "s_label", "m_label"])
    sentimental_label = dict([(s_label[0], label) for _, (label, s_label, _) in all_label.iterrows() if int(label) < 6])
    multimodal_label = dict([(m_label, label) for _, (label, _, m_label) in all_label.iterrows() if pd.notna(m_label)])

    print('start making sentimental conversation dataset.')
    sentimental_T = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json', 'r+', encoding='utf-8'))
    sentimental_V = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json', 'r+', encoding='utf-8'))
    for conv in sentimental_T:
        train = train.append(pd.DataFrame([[conv['talk']['content']['HS01'],
                                          sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]]], columns=["data", "label"]))
    for conv in sentimental_V:
        val = val.append(pd.DataFrame([[conv['talk']['content']['HS01'],
                                      sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]]], columns=["data", "label"]))

    print('start making multimodal video dataset')
    for fpath in os.listdir("./data/멀티모달_영상"):
        for fname in os.listdir("./data/멀티모달_영상/"+fpath):
            try:
                temp_mm = json.load(open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='utf-8'))
            except UnicodeDecodeError:
                temp_mm = json.load(open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='949'))

            for conv in temp_mm['data'].values():  # repeat for all data in this file
                for person in conv.keys():
                    if 'text' not in conv[person].keys():  # find text data
                        continue
                    # if conv[person]['text']['script'] == hist[-1]:  # skip duplicate sentence
                    if conv[person]['text']['script'] in hist:  # skip duplicate sentence
                        continue

                    hist.append(conv[person]['text']['script'])
                    if random.randint(1, 10) > 4:  # train val split in random (7:3)
                        train = train.append(pd.DataFrame([[conv[person]['text']['script'],
                                                          multimodal_label[conv[person]['emotion']['text']['emotion']]]], columns=["data", "label"]))
                    else:
                        val = val.append(pd.DataFrame([[conv[person]['text']['script'],
                                                      multimodal_label[conv[person]['emotion']['text']['emotion']]]], columns=["data", "label"]))
            print(f"multi_modal {fname[5:]} ended")

    train.to_csv('./data/train.txt', sep='\t', encoding='utf-8')
    val.to_csv('./data/val.txt', sep='\t', encoding='utf-8')

    print('making dataset finished.')

def make_vocab():
    hist = [""]
    vocab = dict()
    tokenizer = Hannanum()

    vocab['<pad>'] = 0
    vocab['<oov>'] = 1

    print('start sentimental conversation dataset.')
    sentimental_T1 = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json', 'r+', encoding='utf-8'))
    sentimental_T2 = json.load(open('./data/감성대화/감성대화말뭉치(원천데이터)_Training/감성대화말뭉치(원시데이터)_Training.json', 'r+', encoding='utf-8'))
    sentimental_V1 = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json', 'r+', encoding='utf-8'))
    sentimental_V2 = json.load(open('./data/감성대화/감성대화말뭉치(원천데이터)_Validation/감성대화말뭉치(원시데이터)_Validation.json', 'r+', encoding='utf-8'))
    for dataset in [sentimental_T1, sentimental_T2, sentimental_V1, sentimental_V2]:
        for conv in dataset:
            for sentence in conv['talk']['content'].values():
                for token in [t for (t, tag) in tokenizer.pos(re.sub(r"\W", r" ", sentence)) if ('N' in tag) or ('P' in tag) or ('F' in tag)]:
                    if token not in vocab:
                        vocab[token] = len(vocab)
        print("one of dataset ended")

    print('start multimodal video dataset')
    for fpath in os.listdir("./data/멀티모달_영상"):
        for fname in os.listdir("./data/멀티모달_영상/" + fpath):
            try:
                temp_mm = json.load(
                    open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='utf-8'))
            except UnicodeDecodeError:
                temp_mm = json.load(
                    open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='949'))

            for conv in temp_mm['data'].values():  # repeat for all data in this file
                for person in conv.keys():
                    if 'text' not in conv[person].keys():  # find text data
                        continue
                    if conv[person]['text']['script'] == hist[-1]:  # skip duplicate sentence
                        continue
                    hist.append(conv[person]['text']['script'])
                    for token in [t for (t, tag) in tokenizer.pos(re.sub(r'\W', r" ", hist[-1]).strip()) if ('N' in tag) or ('P' in tag) or ('F' in tag)]:
                        if token not in vocab:
                            vocab[token] = len(vocab)
            print(f"multi_modal {fname[5:]} ended")

    pd.DataFrame((token, index) for token, index in vocab.items()).to_csv("./data/vocab.txt", sep="\t", encoding="utf-8", header=False, index=False)
    print('making vocabulary finished.')

class Preprocesser:
    def __init__(self, use_HF=True):
        self.use_HF = use_HF
        self.MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        self.SEED = 1000
        self.vocab = dict([(key, index) for _, (key, index) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8").iterrows()])
        self.id_to_token = dict([(index, key) for _, (key, index) in pd.read_csv("./data/vocab.txt", sep="\t", encoding="utf-8").iterrows()])
        if not use_HF:
            self.tokenizer = Hannanum()
            self.input_dim = None
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
            self.input_dim = 100  # train max : 99, val max : 82
        self.output_dim = 11

        # hyper parameter
        self.embed_dim = 128
        self.batch_size = 16

    def getTrainDataset(self) -> tf.data.Dataset:
        if self.use_HF:
            train_set = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
            train_X = train_set["data"].apply(lambda data: re.sub(r"[은는이가을를에게께]", r"", re.sub(r"\W", r" ", data)))
            train_X = self.tokenizer.batch_encode_plus(train_X.to_list(), max_length=self.input_dim,
                                                       padding="max_length", truncation=True, return_tensors="tf")
            en_train_X = dict()
            for key, item in train_X.items():
                en_train_X[key] = item

            train_y = train_set["label"].apply(lambda data: int(data))
            en_train_y = tf.keras.utils.to_categorical(train_y)

            return tf.data.Dataset.from_tensor_slices((en_train_X, en_train_y)).batch(self.batch_size).shuffle(256, seed=self.SEED)
        else:
            train_set = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
            train_X = train_set["data"].apply(lambda x: self.tokenize(x))
            train_y = tf.keras.utils.to_categorical(train_set["label"].apply(lambda data: int(data)))

            return tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(self.batch_size).shuffle(256, seed=self.SEED)

    def getValidationDataset(self) -> tf.data.Dataset:
        if self.use_HF:
            val_set = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
            val_X = val_set["data"].apply(lambda data: re.sub(r"[은는이가을를에게께]", r"", re.sub(r"\W", r" ", data)))
            val_X = self.tokenizer.batch_encode_plus(val_X.to_list(), max_length=self.input_dim,
                                                     padding="max_length", truncation=True, return_tensors="tf")
            en_val_X = dict()
            for key, item in val_X.items():
                en_val_X[key] = item

            val_y = val_set["label"].apply(lambda data: int(data))
            en_val_y = tf.keras.utils.to_categorical(val_y)

            return tf.data.Dataset.from_tensor_slices((en_val_X, en_val_y)).batch(self.batch_size).shuffle(256, seed=self.SEED)
        else:
            val_set = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
            val_X = val_set["data"].apply(lambda x: self.tokenize(x))
            val_y = tf.keras.utils.to_categorical(val_set["label"].apply(lambda data: int(data)))

            return tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(self.batch_size).shuffle(256, seed=self.SEED)

    def tokenize(self, text: str, return_tensor: bool = True) -> Union[tf.Tensor, List[int]]:
        if self.use_HF:
            text = re.sub(r"\W", r" ", text)
            text = re.sub(r"[은는이가을를에게께]", "", text)
            return self.tokenizer.encode(text, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="tf")
        else:
            # N - 체언 | P - 용언 | F - 외국어
            text = re.sub(r"\W", r" ", text)
            text = [token for (token, tag) in self.tokenizer.pos(text) if ('N' in tag) or ('P' in tag) or ('F' in tag)]
            for i, token in enumerate(text):
                text[i] = self.vocab[token] if token in self.vocab else self.vocab['<oov>']

            text = (text + [0] * (self.input_dim - len(text)))[:self.input_dim]
            if return_tensor:
                return tf.convert_to_tensor(text)
            else:
                return text
