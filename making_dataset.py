from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
import pandas as pd
import zipfile
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
    print("making vocab started")
    CUTTING_RATE = 0.7

    tokenizer = Okt()
    token_dict = dict()

    train = pd.read_csv("./data/train.txt", sep="\t", index_col=0, encoding="utf-8")
    val = pd.read_csv("./data/val.txt", sep="\t", index_col=0, encoding="utf-8")
    test = pd.read_csv("./data/test.txt", sep="\t", index_col=0, encoding="utf-8")
    data = pd.concat([train, val, test], axis=0)

    with zipfile.ZipFile("data/vocab_data/kowiki.txt.zip") as z:
        with z.open('kowiki.txt') as f:
            wiki_data = [line.decode('utf-8').strip() for line in f]
    data = data.append(pd.DataFrame(wiki_data, columns=["data"]))

    for text in data["data"].values:
        text = re.sub(r"[^가-힣ㄱ-ㅎa-zA-z0-9.,?! ]", "", text).strip()
        for token in tokenizer.morphs(text, norm=True, stem=True):
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
    print("tokenizing ended")

    vocab = pd.DataFrame.from_dict({'token': token_dict.keys(), 'count': token_dict.values()})
    vocab.sort_values(["count"], inplace=True, ignore_index=True)
    vocab = vocab.iloc[:int(len(vocab)*CUTTING_RATE)]
    print("sort and cutting ended")

    vocab = vocab.drop(["count"], axis=1)
    vocab["index"] = range(2, len(vocab)+2)
    vocab = pd.concat([pd.DataFrame([["<pad>", 0], ["<oov>", 1]], columns=["token", "index"]), vocab], ignore_index=True)
    vocab.to_csv("./data/vocab.txt", sep="\t", encoding='utf-8')
    print("making vocab finished")
