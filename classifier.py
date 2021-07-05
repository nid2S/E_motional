import sklearn
from tensorflow.keras.models import load_model

model = load_model("model/imdb.h5")

prediction = model.predict(X_test)
acc = sklearn.metrics.accuracy_score(prediction, Y_test)
print(acc)
