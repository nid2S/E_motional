from transformers import BertTokenizerFast
from tokenizers.models import BPE
import tokenizers
import tensorflow as tf
import pandas as pd
import random
import json
import os

def make_dataset():
    # sentimental => T - 40827 , V - 5122 | multimodal => T - ?, V - ? (7:3)
    train = pd.DataFrame(columns=["data", "label"])
    val = pd.DataFrame(columns=["data", "label"])

    # print('making label dict')
    all_label = pd.read_csv("./data/label.txt", encoding="utf-8", names=["label", "s_label", "m_label"])
    sentimental_label = dict([(s_label[0], label) for _, (label, s_label, _) in all_label.iterrows() if int(label) < 6])
    multimodal_label = dict([(m_label, label) for _, (label, _, m_label) in all_label.iterrows() if pd.notna(m_label)])

    print('start making sentimental conversation dataset.')
    sentimental_T = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json', 'r+', encoding='utf-8'))
    sentimental_V = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json', 'r+', encoding='utf-8'))
    for conv in sentimental_T:
        train.append(pd.DataFrame([[conv['talk']['content']['HS01'],
                                   sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]]], columns=["data", "label"]))
    for conv in sentimental_V:
        val.append(pd.DataFrame([[conv['talk']['content']['HS01'],
                                 sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]]], columns=["data", "label"]))
    hist = []

    print('start making multimodal video dataset')
    for fpath in os.listdir("./data/멀티모달_영상"):
        for fname in os.listdir("./data/멀티모달_영상/"+fpath):
            temp_mm = json.load(open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='utf-8'))
            for conv in temp_mm['data'].values():  # repeat for all data in this file
                for person in conv.keys():
                    if 'text' not in conv[person].keys():  # find text data
                        continue
                    if conv[person]['text']['script'] in hist:  # skip duplicate sentence
                        continue
                    hist.append(conv[person]['text']['script'])
                    if random.randint(1, 10) > 4:  # train val split in random (7:3)
                        train.append(pd.DataFrame([[conv[person]['text']['script'],
                                                  multimodal_label[conv[person]['emotion']['text']['emotion']]]], columns=["data", "label"]))
                    else:
                        val.append(pd.DataFrame([[conv[person]['text']['script'],
                                                multimodal_label[conv[person]['emotion']['text']['emotion']]]], columns=["data", "label"]))
    train.to_csv('./data/train.txt', sep='\t', encoding='utf-8')
    val.to_csv('./data/val.txt', sep='\t', encoding='utf-8')

    print('making dataset finished.')

class Preprocesser:
    def __init__(self, use_HF=False):
        self.MODEL_NAME = "skt/kobert-base-v1"
        if use_HF:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
        else:
            self.tokenizer = tokenizers.Tokenizer(BPE())
            self.set_tokenizer()

        # hyper parameter
        self.batch_size = 16
        self.input_dim = 0
        self.output_dim = 0
        self.embed_dim = 0

    def set_tokenizer(self):
        pass

    def getTrainDataset(self):
        pass

    def getValidationDataset(self):
        pass

make_dataset()
