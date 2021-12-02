from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from transformers import TFBertForSequenceClassification
from preprocessing import Preprocesser
import tensorflow as tf

p = Preprocesser()

# def HF_model():
#     return TFBertForSequenceClassification.from_pretrained(p.MODEL_NAME)

def TF_model() -> tf.keras.Model:
    pass


# hyper_param
epochs = 50
lr = 0.01
p.batch_size = 16
optim = tf.optimizers.Adam(learning_rate=lr)
# train
model = TF_model()
model.compile(optim, "sparse_categorical_crossentropy", "accuracy")
model.fit(p.getTrainDataset(), validation_data=p.getValidationDataset(), batch_size=p.batch_size, epochs=epochs,
          callbacks=[EarlyStopping(monitor='val_loss', patience=4), ModelCheckpoint("./model/emotion_classification", monitor="val_accuracy", save_best_only=True)])
