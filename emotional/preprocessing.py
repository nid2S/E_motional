from transformers import BertTokenizerFast
from tokenizers.models import BPE
import tokenizers
import tensorflow as tf
import pandas as pd
import json

def make_dataset():
    pass

class Preprocesser:
    def __init__(self):
        # self.MODEL_NAME = ""
        # self.tokenizer = BertTokenizerFast.from_pretrained(self.MODEL_NAME)
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
