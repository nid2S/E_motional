from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def imdb_modelMaking(vocab_size=10000, max_len=2000, embedding_dim=100):
    # 정제(Nomalize) + 토큰화 + 정수화 | 0-부정, 1-긍정
    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocab_size)  # 빈도수 vocab_size위 까지의 단어만 가져옴

    # 각 문장의 길이는 max_len(2000)으로 고정
    X_train = pad_sequences(X_train, maxlen=max_len)  # (25000, max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)

    model = Sequential()
    # model.add(layers.Input(max_len))                      # (각 문장 길이). (25000, 2000)으로 사용해도 구동은 되나 생략가능.
    model.add(layers.Embedding(vocab_size, embedding_dim))  # (문장에서 등장하는(정수인코딩 된)단어 총수, 임베딩 벡터 차원).
    model.add(layers.GRU(64))                              # RNN의 일종인 GRU로 예측
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test),
              callbacks=[EarlyStopping(monitor="val_loss", patience=3),
                         ModelCheckpoint('model/imdb.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)])

pass
