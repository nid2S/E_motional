import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

def imdb_preprocessing(new_sentence, vocab_size=10000, max_len=2000):
    """eng only"""

    # 알파벳과 숫자 제외 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub(r'[^0-9a-zA-Z ]', r'', new_sentence).lower()

    # 정수 인코딩
    encoded = []
    word_to_index = imdb.get_word_index()
    for word in new_sentence.split():
        # 단어 집합의 크기를 vocab_size(10000)으로 제한.
        try:
            if word_to_index[word] <= vocab_size:
                encoded.append(word_to_index[word] + 3)
            else:
                # 10,000 이상의 숫자는 <unk> 토큰으로 취급.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 취급.
        except KeyError:
            print("없는 단어 : "+word)
            encoded.append(2)

    # max_len(2000)으로 길이를 맞춰 패딩 후 반환
    return pad_sequences([encoded], maxlen=max_len)  # 패딩

pass
