from sklearn.model_selection import train_test_split
import pandas as pd

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
