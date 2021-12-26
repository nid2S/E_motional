# Task
- 문장의 감정 분류.
- 메일이나 간단한 대화, SNS용 게시물은 물론 자소서등의 공적인 문서 또한 대상에 해당됨.
- 분야를 가리지 않고 데이터를 모아, 전처리를 통해 중요 토큰만 남겨 학습에 사용.

# Model
- HF : 허깅페이스의 Bert(엘렉트라) 모델을 사용 | 엘렉트라의 탓인지 TFLite로의 변환이 불가했음(TFMobile사용에도 불구).
  -> [snunlp/KR-Medium](https://huggingface.co/snunlp/KR-Medium) -> 느리고 정확도도 낮음
- TF : KoNLPy의 토크나이저 하나와 LSTM을 사용.

# Data
- [멀티모달 영상](https://aihub.or.kr/aidata/137)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978)

# notice
- 허깅페이스 등 커스텀 모델을 학습시키고 tf모델로 부른 상태라면 call함수로 사용할 때 훈련때와 똑같은 전처리가 필요함(당연히 반환값도 np.argmax(axis=-1)필요).

# error
- ValueError: Cannot reshape a tensor with 768 elements to shape [1,1,100,1] (100 elements) 에러 발생. Epoch 후, 에러코드를 보니 층 정규화 단계에서 에러발생.
- 해결 : 변경사항 1 -> toCategorical 후 Sparse에서 CategoricalCrossentropy로 바꿈 | 변경사항 2 : batch_encode_plus 후 dict형태로 전달.
- 어느쪽이 문제였을지 모르겠으나, Epochs 후 층 정규화 쪽에서 형상변환 오류가 생겼던 걸로 보아 Sparse 쪽이 문제가 아니였을까 함.

- vocab 생성 중 중간에 갑자기 교착상태에 빠지는 오류가 있었음. 확인결과 konlpy의 pos함수는 공백만 존재하는 문장을 입력하면 그런 상태가 되었었음.

- tensorflow.python.framework.errors_impl.InvalidArgumentError:  indices[22,2] = 4109 is not in [0, 55 | Errors may have originated from an input operation.
- input_layer = tf.keras.layers.Input(shape=p.input_dim) | x = tf.keras.layers.Embedding(input_dim=p.input_dim(55), output_dim=p.embed_dim)(input_layer)
  입력차원이 임베딩 레이어에 들어오는 값이 차원이 아니라 들어온 데이터의 원핫인코딩된(진짜)차원인 듯. vocab size와 동일하게 설정해 줘야 함.
  -> input_dim으로 설정했던 55가 vocab size로 인식되어, 입력 데이터 중 특정 인덱스(이경우엔 22, 2)의 데이터가 해당 범위를 넘자 생긴 오류. 
- ValueError: Input 0 of layer lstm is incompatible with the layer: expected ndim=3, found ndim=4. Full shape received: (None, None, 55, 128)
- input_layer = tf.keras.layers.Input(shape=(None, p.input_dim)) | x = tf.keras.layers.Embedding(input_dim=len(p.vocab), output_dim=p.embed_dim)(input_layer)
  shape는 batch를 포함하지 않아, (None, input_dim)로 입력하면 입력데이터의 차원이 그런줄 알아서, 저기에 batch_size로 None이 붙어 ndim이 4가 되니 생기는 오류.
- -> input_layer = tf.keras.layers.Input(shape=p.input_dim) | x = tf.keras.layers.Embedding(input_dim=len(p.vocab), output_dim=p.embed_dim)(input_layer)
  로 변경하니 결론적으로는 해결되었음. 

- 허깅페이스 모델 학습 과정에서 학습 완료 후 tflite로 변환시 변환이 되지 않는 오류가 있었음. -> TFMobileBert로 변경하는 등 다양한 노력이 있었으나, 결국 허깅페이스 사용을 포기함.
  -> "monologg/koelectra-base-v3-discriminator"모델에서 생긴 오류일거라 추측, 다른 모델로 바꿔 시도해봄. 

- 모델 학습 과정에서 val_accuracy가 0.3193에서 변하지 않는 현상이 발생했음. -> 추후 약간의 변화가 있긴 했으나, 0.001 안팎의 변화였음
  -> 모델의 파라미터가 갱신이 되지 않았다? -> val_loss나 loss, accuracy는 변화했음. | lr이나 모델 파라미터 수정, 추후 트랜스포머로의 교체등의 테스트와 텐서보드의 사용이 필요.
- val_loss 기준으로는 adam, LSTM모델과 BiLSTM모델이 2.0567로 최저치를 달성하였음.  허깅페이스 모델 훈련 중 확인결과 여기서도 0.3193. 이정도면 val데이터에 문제가 있는게 아닌가 함.
- 일단 accuracy기준으로 학습을 진행해보고, 이후 val_accuracy의 동태를 확인해 볼 예정. 
