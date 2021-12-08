from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFMobileBertForSequenceClassification
from preprocessing import Preprocesser
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-hf', '--use-hf', type=bool, default=False, metavar='Bool', dest="use_HF", help='condition about using HF model')
use_HF = parser.parse_args().use_HF

p = Preprocesser(use_HF)

def HF_model():
    return TFMobileBertForSequenceClassification.from_pretrained(p.MODEL_NAME, from_pt=True, num_labels=p.output_dim)

def TF_model(use_rnn: bool = True, use_LSTM: bool = True) -> tf.keras.Model:
    if use_rnn:
        x = tf.keras.layers.Input(shape=(None, p.batch_size), batch_size=p.batch_size)
        x = tf.keras.layers.Embedding(input_dim=p.input_dim, output_dim=p.embed_dim)(x)
        if use_LSTM:
            x = tf.keras.layers.LSTM(64, activation="tanh", dropout=0.3)(x)
        else:
            x = tf.keras.layers.GRU(64, activation="tanh", dropout=0.3)(x)
        x = tf.keras.layers.Dense(32, activation='relu', dropout=0.3)(x)

        y = tf.keras.layers.Dense(p.output_dim, activation='softmax')(x)
        return tf.keras.Model(x, y)
    else:
        pass


# hyper_param
if use_HF:
    epochs = 4
    p.batch_size = 32
    from_logits = True
    model = HF_model()
else:
    epochs = 100
    p.batch_size = 32
    from_logits = False
    model = TF_model()

def lr_scheduler(epoch, lr):
    if use_HF:
        if epoch < 2:
            return 5e-5
        elif epoch < 4:
            return 3e-5
        else:
            return 1e-5
    else:
        if epoch < 10:
            return 0.01
        else:
            return 0.001 * tf.exp(-0.1)


# train
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
if use_HF:
    model.compile("adam", loss, "accuracy")
    hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3), LearningRateScheduler(lr_scheduler),
                                ModelCheckpoint("./model/emotion_classification", monitor="val_accuracy", save_best_only=True)])
    model.save("./model/emotion.h5")
else:
    history = ""
    for optim in ["adam", "rmsprop", "nadam"]:
        for order_RNN in ["LSTM", "GRU"]:
            model = TF_model(use_LSTM=(order_RNN == "LSTM"))
            model.compile(optim, loss, "accuracy")
            hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                             callbacks=[EarlyStopping(monitor='val_loss', patience=3), LearningRateScheduler(lr_scheduler),
                                        ModelCheckpoint("./model/emotion_"+order_RNN + "_" + optim, monitor="val_accuracy", save_best_only=True)])
            model.save("./model/emotion_"+order_RNN + "_" + optim+".h5")
            history += order_RNN + "_" + optim + ":" + str(hist) + "\n"
    open("./model/history.txt", "w+", encoding="utf-8").write(history)
