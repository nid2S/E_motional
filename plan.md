# Task
- 문장의 감정 분류.
- 메일이나 간단한 대화, SNS용 게시물은 물론 자소서등의 공적인 문서 또한 대상에 해당됨.
- 분야를 가리지 않고 데이터를 모아, 전처리를 통해 중요 토큰만 남겨 학습에 사용.
- 이후 PyTorch Mobile로 변환하고, Cortex등을 이용해 API로 바꿔 배포까지 해봄.
- 레이블 : 기쁨, 분노, 슬픔, 불안, 놀람, 혐오, 중립

# Data
- [멀티모달 영상](https://aihub.or.kr/aidata/137)
- [감성대화 말뭉치](https://aihub.or.kr/aidata/7978)
- 전처리 : konlpy의 Hannaum 토크나이저를 사용, 허깅헤이스 사용시 모델에 맞는 토크나이저를 사용.

# Model
- 먼저 간단한 RNN기반 모델을 PL로 제작한 뒤, CNN기반, Attention + S2S, Transformer등도 제작해봄. 

# Problem
- softmax -> mean 방식으로 하자니 input_dim이 너무 길어 각 학습이 잘 적용되지 않을것 같음.
  -> 편차가 가장 큰(표준편차나 분산이 가장 큰)텐서를 고른다 -> grad_fn으로 추적이 안되서 학습이 안될듯
  -> input_dim을 줄인다 -> 이건 생각 좀 해봐야 할듯(굳이 이 문제 아니더라도 input dim이 길어서 좋을거 없으니까)
