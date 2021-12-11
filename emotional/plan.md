# Task
- 문장의 감정 분류.
- 메일이나 간단한 대화, SNS용 게시물은 물론 자소서등의 공적인 문서 또한 대상에 해당됨.
- 분야를 가리지 않고 데이터를 모아, 전처리를 통해 중요 토큰만 남겨 학습에 사용.

# Model
- HF : 허깅페이스의 Bert(엘렉트라) 모델을 사용 -> koBert의 토크나이저가 따로 설치가 필요했음 | 엘렉트라의 탓인지 TFLite로의 변환이 불가했음(TFMobile사용에도 불구).
- TF : KoNLPy의 토크나이저 하나와 LSTM을 사용.

# Data
- [멀티모달 영상](https://aihub.or.kr/aidata/137)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978)

# notice
- 감성대화 말뭉치 -> 데이터 감정이 SXX_DXX_EXX 식으로 되어있음. S와 D는 상황/질병이고, E만 감정인데, 첫번째 X는 대감정, 두번째는 소감정이였음. 소감정까지 사용시 주의필요.
- 멀터모달 영상 -> 기본적으로 데이터가 낱낱이 나뉘어 있고, 프레임 단위로 되어 있어 한 문장이 특정구간동안 이어지며, 화자에 따라 데이터가 들어있는 위치가 달라짐. 

# error
- ValueError: Cannot reshape a tensor with 768 elements to shape [1,1,100,1] (100 elements) 에러 발생. Epoch 후, 에러코드를 보니 층 정규화 단계에서 에러발생.
- 해결 : 변경사항 1 -> toCategorical 후 Sparse에서 CategoricalCrossentropy로 바꿈 | 변경사항 2 : batch_encode_plus 후 dict형태로 전달.
- 어느쪽이 문제였을지 모르겠으나, Epochs 후 층 정규화 쪽에서 형상변환 오류가 생겼던 걸로 보아 Sparse 쪽이 문제가 아니였을까 함.

- vocab 생성 중 중간에 갑자기 교착상태에 빠지는 오류가 있었음. 확인결과 konlpy의 pos함수는 공백만 존재하는 문장을 입력하면 그런 상태가 되었었음.
