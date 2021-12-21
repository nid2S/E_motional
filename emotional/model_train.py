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

def TF_model(use_LSTM: bool = True, use_Bidirectional: bool = False) -> tf.keras.Model:
    input_layer = tf.keras.layers.Input(shape=p.input_dim)
    x = tf.keras.layers.Embedding(input_dim=p.vocab_size, output_dim=p.embed_dim)(input_layer)

    if use_LSTM:
        RNN_layer = tf.keras.layers.LSTM(64, activation="tanh", dropout=0.1)
    else:
        RNN_layer = tf.keras.layers.GRU(64, activation="tanh", dropout=0.1)
    if use_Bidirectional:
        RNN_layer = tf.keras.layers.Bidirectional(RNN_layer)

    x = RNN_layer(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)

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
    model.save("./model/emotion.h5")
else:
    pos = 1
    for optim in ["adam", "rmsprop", "nadam"]:
        for order_RNN in ["LSTM", "GRU"]:
            for use_Bi in ["", "Bi"]:
                plt.subplot(3, 4, pos)
                pos += 1

                model = TF_model(use_LSTM=(order_RNN == "LSTM"), use_Bidirectional=(use_Bi == "Bi"))
                model.compile(optim, loss, "accuracy")
                hist = model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
                                 callbacks=[EarlyStopping(monitor='val_loss', mode="min", patience=5), LearningRateScheduler(lr_scheduler),
                                            ModelCheckpoint("./model/"+use_Bi+order_RNN+"/"+optim+"_max_accuracy", monitor="val_accuracy", save_best_only=True),
                                            ModelCheckpoint("./model/"+use_Bi+order_RNN+"/"+optim+"_min_loss", mode="min", monitor="val_loss", save_best_only=True)])

                plt.plot(range(1, len(hist.history["loss"])+1), hist.history["loss"], "r", label="loss")
                plt.plot(range(1, len(hist.history["loss"])+1), hist.history["accuracy"], "b", label="accuracy")
                plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_loss"], "g", label="val_loss")
                plt.plot(range(1, len(hist.history["loss"])+1), hist.history["val_accuracy"], "k", label="val_accuracy")
                plt.title(optim+"_"+use_Bi+order_RNN)
                plt.text(5, 3, str(max(hist.history["val_accuracy"])))
                plt.xlabel("epoch")
                plt.ylabel("loss/accuracy")
                plt.xticks(range(1, len(hist.history["loss"])+1))
                plt.xlim(0.9, len(hist.history["loss"])+0.1)
                plt.legend()
    plt.savefig("./history.png")
