from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from transformers import TFMobileBertForSequenceClassification
from preprocessing import Preprocesser
import tensorflow as tf
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-hf', '--use-hf', type=bool, default=True, metavar='Bool', dest="use_HF", help='condition about using HF model')
# use_HF = parser.parse_args().use_HF
use_HF = True
p = Preprocesser(use_HF)

def HF_model():
    return TFMobileBertForSequenceClassification.from_pretrained(p.MODEL_NAME, from_pt=True, num_labels=p.output_dim)

def TF_model(use_rnn: bool = True) -> tf.keras.Model:
    if use_rnn:
        x = tf.keras.layers.Input(shape=(None, p.batch_size), batch_size=p.batch_size)
        x = tf.keras.layers.Embedding(input_dim=p.input_dim, output_dim=p.embed_dim)(x)

        x = tf.keras.layers.LSTM(64, activation="tanh", dropout=0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu', dropout=0.3)(x)

        y = tf.keras.layers.Dense(p.output_dim, activation='softmax')(x)
        return tf.keras.Model(x, y)
    else:
        pass


# hyper_param
if use_HF:
    epochs = 4
    lr = 3e-5
    p.batch_size = 32
    model = HF_model()
else:
    epochs = 50
    lr = 0.01
    p.batch_size = 16
    model = TF_model()
optim = tf.optimizers.Adam(learning_rate=lr)

# train
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optim, loss, "accuracy")
model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
          callbacks=[EarlyStopping(monitor='val_loss', patience=3), ModelCheckpoint("./model/emotion_classification",
                                                                                    monitor="val_accuracy", save_best_only=True)])
model.save("./model/emotion.h5")
