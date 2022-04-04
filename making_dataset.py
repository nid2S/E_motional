from sklearn.model_selection import train_test_split
from transformers import MobileBertTokenizerFast
from hgtk.text import compose
from konlpy.tag import Okt
import pandas as pd
import logging
import zipfile
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt=None, style='$'))
logger.addHandler(handler)

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
    logger.info("making vocab started")
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
    data_num = len(data)
    logger.info('making raw dataset ended | data_num: '+str(data_num))

    for i, text in enumerate(data["data"].values):
        text = re.sub(r"[^가-힣ㄱ-ㅎa-zA-z0-9.,?! ]", "", text).strip()
        for token in tokenizer.morphs(text, norm=True, stem=True):
            try:
                token_dict[token] += 1
            except KeyError:
                token_dict[token] = 1
        if i % 1000 == 0:
            logger.info(f"progress : {i}/{data_num} ({i/data_num*100}%)")
    logger.info("tokenizing ended")

    vocab = pd.DataFrame.from_dict({'token': token_dict.keys(), 'count': token_dict.values()})
    vocab.sort_values(["count"], inplace=True, ignore_index=True, ascending=False)
    vocab = vocab.iloc[:int(len(vocab)*CUTTING_RATE)]
    logger.info("sort and cutting ended")

    vocab = vocab.drop(["count"], axis=1)
    vocab["index"] = range(2, len(vocab)+2)
    vocab = pd.concat([pd.DataFrame([["<pad>", 0], ["<oov>", 1]], columns=["token", "index"]), vocab], ignore_index=True)
    vocab.to_csv("./data/vocab.txt", sep="\t", encoding='utf-8')
    logger.info("making vocab finished")

def make_char_vocab():
    initial_consonants = ["ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㄸ", "ㅃ", "ㅆ", "ㅉ"]
    middle_consonants = ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅛ", "ㅡ", "ㅣ", "ㅐ", "ㅒ", "ㅔ", "ㅖ", "ㅘ", "ㅙ", "ㅝ", "ㅞ", "ㅚ", "ㅟ", "ㅢ"]
    last_consonants = ["", "ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㄲ", "ㅆ", "ㄵ", "ㄶ", "ㄼ", "ㄽ", "ㄿ", "ㅀ", "ㅄ", "ㄿ"]
    other_chars = [".", ",", "?", "!", '"', "'", "^", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    keys = initial_consonants + list(map(lambda x: "_"+x, middle_consonants)) + list(map(lambda x: "__"+x, last_consonants)) + other_chars
    keys.remove("__")

    tokenizer = MobileBertTokenizerFast.from_pretrained("google/mobilebert-uncased")
    char_vocab = dict((k, "") for k in keys)
    logger.info(f"char_num : {len(char_vocab)}")

    for init in initial_consonants:
        for mid in middle_consonants:
            for last in last_consonants:
                turn_char = compose(init+mid+last+"_", compose_code="_")
                encoded_char = tokenizer.encode(turn_char, add_special_tokens=False)
                if encoded_char[0] != tokenizer.unk_token_id:
                    logger.info(f"{turn_char} -> {tokenizer.convert_ids_to_tokens(encoded_char)}")
                    char_vocab[init] = encoded_char[0]
                    char_vocab["_" + mid] = encoded_char[1]
                    if last != "":
                        char_vocab["__"+last] = encoded_char[2]
    for c in other_chars:
        encoded_char = tokenizer.encode(c, add_special_tokens=False)
        if encoded_char[0] != tokenizer.unk_token_id:
            char_vocab[c] = encoded_char[0]

    logger.info(f'non_encoded_char = {[k for k, v in char_vocab.items() if v == ""]}')
    pd.DataFrame(char_vocab.items(), columns=["char", "id"]).to_csv("./data/char_vocab.txt", sep="\t", encoding="utf-8")
