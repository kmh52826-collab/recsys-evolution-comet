# Evolution of Neural Collaborative Filtering: <br> Reproducing COMET and Proposed Improvements

> **Paper:** *COMET: Convolutional Dimension Interaction for Collaborative Filtering (ACM TIST, 2023)*

COMET(Convolutional Dimension Interaction for Collaborative Filtering)은 **2D 합성곱 신경망(CNN)** 을 활용하여 사용자의 상호작용 히스토리 내에 숨겨진 복잡한 패턴을 포착하는 차세대 협업 필터링 모델입니다. 본 프로젝트에서는 NeuMF와 ConvNCF를 거쳐 COMET으로 이어지는 모델의 진화 과정을 심층 분석했습니다.

### 🎯 Key Architectural Innovations
* 여기에다가는 이 프로젝트의 전체 내용을 함축한 것을 집어넣어야 함. 마지막에 하기

## 🖼️ Visual Guide: Understanding COMET
COMET 모델이 데이터를 어떻게 '시각적 패턴'으로 해석하여 정교한 추천을 수행하는지 설명합니다.

### 1. The Concept of Embedding Maps
> COMET은 사용자의 과거 이력을 단순한 수치가 아닌 하나의 **'그림(Picture)'** 으로 변환하여 처리합니다.

<img width="1390" height="427" alt="image" src="https://github.com/user-attachments/assets/0c7a94fe-7eba-415a-933d-d74cdcfa3b58" />

---
### 2. Multi-Dimensional Interaction Cube
> 단순한 2차원 관계를 넘어, 사용자(User), 아이템(Item), 그리고 특징(Feature)이 결합된 고차원적 관계를 모델링합니다.
<img width="2413" height="981" alt="image" src="https://github.com/user-attachments/assets/ab1c6467-822f-45de-ab56-a6fcc2094085" />

---
### 3. Full Architecture Overview
> 입력단에서부터 히스토리 데이터를 통합하여 고차원적인 특징을 추출하도록 설계된 전체 학습 프로세스입니다.
<img width="2556" height="1326" alt="image" src="https://github.com/user-attachments/assets/f16b9479-0eb6-43bf-a639-a59c1b3e368c" />

---

## 🔍 Detailed Implementation & Lab Report
> 본 실험에서는 **NeuMF**, **ConvNCF**, **COMET** 세 가지 모델을 비교 구현하여 데이터 환경에 따른 성능 특성을 검증했습니다.

* **Model Scalability**: NeuMF는 빠른 프로토타이핑에 적합하지만, 데이터 밀도가 높고 복잡한 시퀀스 패턴이 존재하는 환경에서는 COMET이 압도적인 정확도를 기록함을 확인했습니다.
* **Training Dynamics**: COMET과 같은 복잡한 아키텍처는 완전한 수렴을 위해 **50+ Epochs** 이상의 충분한 학습 횟수와 정교한 하이퍼파라미터 튜닝(Learning Rate, Dropout 등)이 필수적임을 도출했습니다.
* **Engineering Trade-offs**: 모델의 정확도 향상에 따른 추론 지연(Latency) 문제를 분석하였으며, 실무 적용을 위한 **모델 양자화(Quantization)** 및 **GPU 가속**의 필요성을 기술적으로 검토했습니다.

#### 데이터 전처리부터 PyTorch를 이용한 모델 구현, 하이퍼파라미터 최적화 및 상세 성능 지표(NDCG, Recall, Precision) 분석을 포함한 17개 장의 전체 실험 과정은 아래 리포트에서 확인하실 수 있습니다.

### 🔗 [Full Report: comet_deep_dive.ipynb](model-experiments/comet_deep_dive.ipynb)
*이 노트북은 모델의 배경 이론부터 최종 아키텍처 결론까지 아우르는 엄격한 기술 분석 내용을 담고 있습니다.*
