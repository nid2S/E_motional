from transformers import ElectraTokenizerFast
from tokenizers.models import BPE
from string import punctuation
from typing import Union
import tensorflow as tf
import pandas as pd
import tokenizers
import random
import json
import re
import os

def make_dataset():
    # sentimental => T - 40827 , V - 5122 | multimodal => T - ?, V - ? (7:3)
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

class Preprocesser:
    def __init__(self, use_HF=True):
        self.use_HF = use_HF
        self.MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
        self.SEED = 1000
        if not use_HF:
            self.tokenizer = tokenizers.Tokenizer(BPE())
            self.set_tokenizer()
        else:
            self.tokenizer = ElectraTokenizerFast.from_pretrained(self.MODEL_NAME)

        # hyper parameter
        self.batch_size = 16
        self.input_dim = 0
        self.output_dim = 11
        self.embed_dim = 128

    def set_tokenizer(self):
        pass

    def getTrainDataset(self) -> tf.data.Dataset:
        # 데이터 max_length 확인 | 개수 확인 | 데이터셋 질 확인
        train_set = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
        train_set["data"] = train_set["data"].apply(lambda data: self.tokenize(data))
        train_set["label"] = train_set["label"].apply(lambda data: tf.constant(int(data)))

        return tf.data.Dataset.from_tensor_slices((train_set["data"].tolist(), train_set["label"].tolist()))\
            .batch(self.batch_size).shuffle(256, seed=self.SEED)

    def getValidationDataset(self) -> tf.data.Dataset:
        val_set = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8").drop(['Unnamed: 0'], axis=1)
        val_set["data"] = val_set["data"].apply(lambda data: self.tokenize(data))
        val_set["label"] = val_set["label"].apply(lambda data: tf.constant(int(data)))

        return tf.data.Dataset.from_tensor_slices((val_set["data"].tolist(), val_set["label"].tolist())) \
            .batch(self.batch_size).shuffle(256, seed=self.SEED)

    def tokenize(self, text: str) -> Union[tf.Tensor, list[int]]:
        if self.use_HF:
            text = re.sub("\W", " ", text)
            text = re.sub(f"[{punctuation}]", "", text)
            return self.tokenizer.encode(text, max_length=self.input_dim, padding="max_length", truncation=True, return_tensors="tf")
        else:
            pass