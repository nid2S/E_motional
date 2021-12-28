from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from transformers import TFMobileBertForSequenceClassification
from preprocessing import Preprocesser
import tensorflow as tf
import argparse
import datetime
import os

parser = argparse.ArgumentParser()
parser.add_argument('-hf', '--use-hf', type=bool, default=True, metavar='Bool', dest="use_HF", help='condition about using HF model')
use_HF = parser.parse_args().use_HF

p = Preprocesser(use_HF)

class HF_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(HF_model, self).__init__(*args, **kwargs)
        self.model = TFMobileBertForSequenceClassification.from_pretrained(p.MODEL_NAME, num_labels=p.output_dim, from_pt=True)

    def call(self, inputs, training=None, mask=None):
        output = self.model(inputs, return_dict=True)
        return output.logits

def TF_model(order_model: str = "LSTM", use_Bidirectional: bool = False) -> tf.keras.Model:
    assert order_model in ["LSTM", "CNN", "attention"]

    input_layer = tf.keras.layers.Input(shape=p.input_dim)
    x = tf.keras.layers.Embedding(input_dim=p.vocab_size, output_dim=p.embed_dim)(input_layer)
    # x = tf.keras.layers.LayerNormalization()(x)

    if order_model == "RNN":
        model_layer = tf.keras.layers.LSTM(64, activation="tanh", dropout=0.3)
        if use_Bidirectional:
            model_layer = tf.keras.layers.Bidirectional(model_layer)
    elif order_model == "CNN":
        x = tf.keras.layers.Conv1D(128, 4, activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64)(x)
        x = tf.keras.layers.Conv1D(32, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
        model_layer = tf.keras.layers.Dropout(0.3)
    else:
        model_layer = None

    x = model_layer(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)

    y = tf.keras.layers.Dense(p.output_dim, activation='softmax')(x)
    return tf.keras.Model(input_layer, y)


# hyper_param
if use_HF:
    epochs = 4
    p.batch_size = 16
    model = HF_model()
else:
    epochs = 150
    p.batch_size = 32
    model = TF_model()

def lr_scheduler(epoch, lr):
    if use_HF:
        if epoch < 2:
            return 0.1  # 5e-5
        elif epoch < 4:
            return 0.01  # 3e-5
        else:
            return 0.001  # 1e-5
    else:
        if epoch < 10:
            return 0.1
        elif epoch < 20:
            return 0.01
        else:
            return 0.001 * tf.exp(-0.1)


# train
log_dir = os.path.join('./logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
loss = tf.keras.losses.CategoricalCrossentropy()
if use_HF:
    model.compile("adam", loss, "accuracy")
    hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=3), LearningRateScheduler(lr_scheduler),
                                ModelCheckpoint("./model/emotion_classification", monitor="val_accuracy", save_best_only=True),
                                TensorBoard(log_dir=log_dir, write_graph=True, write_images=True, histogram_freq=1)])
else:
    pos = 1
    history = ""
    for model_order in ["LSTM", "CNN", "attention"]:
        for use_Bi in ["", "Bi"]:
            if model_order != "LSTM" and use_Bi == "Bi":
                continue
            if model_order == "attention":
                continue

            model = TF_model(order_model=model_order, use_Bidirectional=(use_Bi == "Bi"))
            model.compile("adam", loss, "accuracy")
            hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                             callbacks=[EarlyStopping(monitor='loss', mode="min", patience=5), LearningRateScheduler(lr_scheduler),
                                        ModelCheckpoint("./model/"+use_Bi+model_order, monitor="accuracy", save_best_only=True)])
            history += use_Bi + model_order + "\n"
            for key, item in hist.history.items():
                history += key + " : " + str(["%.4f" % figure for figure in item]) + "\n"
            history += "\n"
    open("./model/history.txt", "w+", encoding="utf-8").write(history)
