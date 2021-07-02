import sklearn
import re
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

vocap_size = 10000
max_len = 2000
embedding_dim = 100

def imdb_preprocessing(new_sentence):
    # 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
    new_sentence = re.sub('[^0-9a-zA-Z ]', '', new_sentence).lower()

    # 정수 인코딩
    encoded = []
    for word in new_sentence.split():
        # 단어 집합의 크기를 10,000으로 제한.
        word_to_index = imdb.get_word_index()
        try:
            if word_to_index[word] <= vocap_size:
                encoded.append(word_to_index[word] + 3)
            else:
                # 10,000 이상의 숫자는 <unk> 토큰으로 취급.
                encoded.append(2)
        # 단어 집합에 없는 단어는 <unk> 토큰으로 취급.
        except KeyError:
            encoded.append(2)

    return pad_sequences([encoded], maxlen=max_len)  # 패딩


# 정제(Nomalize) + 토큰화 + 정수화 | 0-부정, 1-긍정
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocap_size)  # 빈도수 10000위 까지의 단어만 가져옴

# 각 문장의 길이는 2000으로 고정
X_train = pad_sequences(X_train, maxlen=max_len)  # (25000, 2000)
X_test = pad_sequences(X_test, maxlen=max_len)

model = Sequential()
# model.add(layers.Input(max_len))                      # (각 문장 길이). (25000, 2000)으로 사용해도 구동은 되나 생략가능.
model.add(layers.Embedding(vocap_size, embedding_dim))  # (문장에서 등장하는(정수인코딩 된)단어 총수, 임베딩 벡터 차원).
model.add(layers.GRU(128))                              # RNN의 일종인 GRU로 예측
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test),
          callbacks=[EarlyStopping(monitor="loss", patience=3), ModelCheckpoint("./imdb.h5")])

# prediction = model.predict(X_test)
# acc = sklearn.metrics.accuracy_score(prediction, Y_test)
# print(acc)
