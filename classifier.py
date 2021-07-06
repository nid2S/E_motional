from tensorflow.keras.models import load_model
from textClassifier import preprocessing
import make_model

# make_model.imdb_modelMaking()

model = load_model("model/imdb.h5")
test = preprocessing.imdb_preprocessing(input("긍정/부정 여부를 판단할 문장 입력(영어) : "))


prediction = model.predict(test)
if prediction:    # 1 in prediction:
    print("긍정")
else:
    print("부정")
