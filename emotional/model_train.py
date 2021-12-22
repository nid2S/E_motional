from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from transformers import TFMobileBertForSequenceClassification
from preprocessing import Preprocesser
import matplotlib.pyplot as plt
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-hf', '--use-hf', type=bool, default=False, metavar='Bool', dest="use_HF", help='condition about using HF model')
use_HF = parser.parse_args().use_HF

p = Preprocesser(use_HF)

def HF_model():
    return TFMobileBertForSequenceClassification.from_pretrained(p.MODEL_NAME, from_pt=True, num_labels=p.output_dim)

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
        x = tf.keras.layers.Conv1D(32, 4, activation="relu")(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2, strides=1)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

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
    p.batch_size = 32
    from_logits = True
    model = HF_model()
else:
    epochs = 150
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
            return 0.1
        elif epoch < 20:
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
else:
    pos = 1
    for model_order in ["LSTM", "CNN", "attention"]:
        for use_Bi in ["", "Bi"]:
            if model_order != "LSTM" and use_Bi == "Bi":
                continue
            if model_order == "attention":
                continue

            plt.subplot(3, 4, pos)
            plt.figure(figsize=(50, 50))
            pos += 1

            model = TF_model(order_model=model_order, use_Bidirectional=(use_Bi == "Bi"))
            model.compile("adam", loss, "accuracy")
            hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                             callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=7), LearningRateScheduler(lr_scheduler),
                                        ModelCheckpoint("./model/"+use_Bi+model_order+"/max_accuracy", monitor="val_accuracy", save_best_only=True),
                                        ModelCheckpoint("./model/"+use_Bi+model_order+"/min_loss", mode="min", monitor="val_loss", save_best_only=True)])

            plt.plot(range(1, len(hist.history["loss"])+1), hist.history["loss"], "r", label="loss")
            plt.plot(range(1, len(hist.history["loss"])+1), hist.history["accuracy"], "b", label="accuracy")
            plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_loss"], "g", label="val_loss")
            plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_accuracy"], "k", label="val_accuracy")
            plt.title(use_Bi+model_order)
            plt.text(1, 3, "max val_acc : %.1f" % max(hist.history["val_accuracy"]))
            plt.text(1, 6, "min val_loss : %.1f" % min(hist.history["val_loss"]))
            plt.xlabel("epoch")
            plt.ylabel("loss/accuracy")
            plt.xticks(range(1, len(hist.history["loss"])+1))
            plt.xlim(0.9, len(hist.history["loss"])+0.1)
        plt.legend()
        plt.savefig("./history.png")
