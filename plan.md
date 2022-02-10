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

# TODO
- accuracy
- vocab 재구축(빈도수 낮은 단어 제거)
- 허깅페이스 사용