import sklearn
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 정제(Nomalize) + 토큰화 + 정수화 | 0-부정, 1-긍정
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=10000)  # 빈도수 10000위 까지의 단어만 가져옴

# 각 문장의 길이는 2000으로 고정
X_train = pad_sequences(X_train, maxlen=2000)  # (25000, 2000)
X_test = pad_sequences(X_test, maxlen=2000)

model = Sequential()
# model.add(layers.Input(2000))
model.add(layers.Embedding(10000, 100))
model.add(layers.GRU(128))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=100, callbacks=[EarlyStopping(monitor="loss", patience=3)])
prediction = model.predict(X_test)
acc = sklearn.metrics.accuracy_score(prediction, Y_test)

print(acc)


