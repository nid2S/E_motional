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

    print('making label dict')
    label = pd.read_csv("./data/label.txt", encoding="utf-8", names=["label", "s_label", "m_label"])
    sentimental_label = dict([(s_label[0], label) for _, (label, s_label, _) in label.iterrows() if label < 6])
    multimodal_label = dict([(m_label, label) for _, (label, _, m_label) in label.iterrows() if pd.notna(m_label)])

    print('start making sentimental conversation dataset.')
    sentimental_T = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Training/감성대화말뭉치(최종데이터)_Training.json', 'r+', encoding='utf-8'))
    sentimental_V = json.load(open('./data/감성대화/감성대화말뭉치(최종데이터)_Validation/감성대화말뭉치(최종데이터)_Validation.json', 'r+', encoding='utf-8'))
    for conv in sentimental_T:
        train.append(pd.DataFrame([conv['talk']['content']['HS01'], sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]], columns=["data", "label"]))
    for conv in sentimental_V:
        val.append(pd.DataFrame([conv['talk']['content']['HS01'], sentimental_label[conv['profile']['emotion']['emotion-id'][-2]]], columns=["data", "label"]))

    train.to_csv('./data/senimental_T.txt', sep="\t", encoding="utf-8")
    val.to_csv('./data/senimental_V.txt', sep="\t", encoding="utf-8")
    del train, val
    train = pd.DataFrame(columns=["data", "label"])
    val = pd.DataFrame(columns=["data", "label"])
    hist = []

    print('start making multimodal video dataset')
    for fpath in os.listdir("./data/멀티모달_영상"):
        for fname in os.listdir("./data/멀티모달_영상/"+fpath):
            temp_mm = json.load(open("./data/멀티모달_영상/" + fpath + "/" + fname + "/" + fname + ".json", 'r+', encoding='utf-8'))
            for key in temp_mm['data']:
                for conv in temp_mm['data'][key]:
                    # 1. 문장이 중복된게 있음. -> 따로 리스트를 만들어서 그쪽을 검사시키자
                    # 2. 화자에 따라 데이터 저장 장소가 다름 -> keys의 항목을 보고 결정
                    # 3. 데이터가 큼 -> 중간중간 저장/초기화 or 내 메모리를 믿고 시도

                    # train.append(pd.DataFrame([conv['1']['text']['script'], multimodal_label[conv['1']]], columns=["data", "label"]))
                    pass

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
