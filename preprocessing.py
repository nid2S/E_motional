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
