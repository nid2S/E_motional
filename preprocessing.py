from sklearn.model_selection import train_test_split
from konlpy.tag import Hannanum
import pandas as pd
import re

def change_dataset():
    train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8").drop(["Unnamed: 0"], axis=1)
    val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8").drop(["Unnamed: 0"], axis=1)
    label_dict = {0: [0], 1: [1], 2: [2], 3: [3, 4, 8], 4: [5, 6], 5: [7, 9], 6: [10]}
    for new_label, past_labels in label_dict.items():
        train["label"] = train["label"].apply(lambda x: new_label if x in past_labels else x)
        val["label"] = val["label"].apply(lambda x: new_label if x in past_labels else x)
    train.to_csv("./data/train.txt", sep="\t", encoding="utf-8")
    val.to_csv("./data/val.txt", sep="\t", encoding="utf-8")

def make_testset():
    train = pd.read_csv("./data/train.txt", sep="\t", encoding="utf-8").drop(["Unnamed: 0"], axis=1)
    val = pd.read_csv("./data/val.txt", sep="\t", encoding="utf-8").drop(["Unnamed: 0"], axis=1)

    train, test_1 = train_test_split(train, train_size=0.9, shuffle=True)
    val, test_2 = train_test_split(val, train_size=0.9, shuffle=True)

    train = train.reset_index().drop(["index"], axis=1)
    val = val.reset_index().drop(["index"], axis=1)
    test = test_1.append(test_2).reset_index().drop(["index"], axis=1)

    train.to_csv("./data/train.txt", sep="\t", encoding="utf-8")
    val.to_csv("./data/val.txt", sep="\t", encoding="utf-8")
    test.to_csv("./data/test.txt", sep="\t", encoding="utf-8")

def make_vocab():
    CUTTING_RATE = 0.7

    tokenizer = Hannanum()
    vocab = dict()

    train = pd.read_csv("./data/train.txt", sep="\t", index_col=0, encoding="utf-8")
    val = pd.read_csv("./data/val.txt", sep="\t", index_col=0, encoding="utf-8")
    test = pd.read_csv("./data/test.txt", sep="\t", index_col=0, encoding="utf-8")
    data = pd.concat([train, val, test], axis=0)

    for text in data["data"].values:
        text = re.sub(r"\W", r" ", text).strip()
        tokens = [token for (token, tag) in tokenizer.pos(text) if ('N' in tag) or ('P' in tag) or ('F' in tag)]
        for token in tokens:
            try:
                vocab[token] += 1
            except KeyError:
                vocab[token] = 1

    vocab_df = pd.DataFrame.from_dict({'token': vocab.keys(), 'count': vocab.values()})
    vocab_df.sort_values(["count"], inplace=True, ignore_index=True)
    vocab_df = vocab_df.iloc[:len(vocab_df)*CUTTING_RATE]
    vocab_df.to_csv("./data/vocab.txt", sep="\t", encoding='utf-8')
