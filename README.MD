# Evolution of Neural Collaborative Filtering: <br> Reproducing COMET and Proposed Improvements

## 📑 Core Technical Summary: COMET
> **Paper:** *COMET: Convolutional Dimension Interaction for Collaborative Filtering (ACM TIST, 2023)*

COMET(Convolutional Dimension Interaction for Collaborative Filtering)은 **2D 합성곱 신경망(CNN)**을 활용하여 사용자의 상호작용 히스토리 내에 숨겨진 복잡한 패턴을 포착하는 차세대 협업 필터링 모델입니다. 본 프로젝트에서는 NeuMF와 ConvNCF를 거쳐 COMET으로 이어지는 모델의 진화 과정을 심층 분석했습니다.

### 🎯 Key Architectural Innovations
* **Dimensional Interaction Modeling**: 단순히 특징을 선형적으로 결합하는 기존 MLP 방식과 달리, **Outer Product(외적)**를 사용하여 임베딩 차원 간의 공간적 상관관계를 보존하는 2D 상호작용 지도를 생성합니다.
* **User History Embedding Map**: 사용자의 과거 행동 이력을 포괄적인 임베딩 맵으로 구성하여, 장기적인 취향 변화와 시퀀스 패턴을 모델이 직접 학습할 수 있도록 설계되었습니다.
* **CNN-based Feature Extraction**: 생성된 상호작용 지점들에 컨볼루션 필터를 적용함으로써, 사용자-아이템 간의 고차원적(High-order) 특징을 자동으로 추출합니다.



### 📊 Experimental Analysis & Insights
본 실험에서는 **NeuMF**, **ConvNCF**, **COMET** 세 가지 모델을 비교 구현하여 데이터 환경에 따른 성능 특성을 검증했습니다.

* **Model Scalability**: NeuMF는 빠른 프로토타이핑에 적합하지만, 데이터 밀도가 높고 복잡한 시퀀스 패턴이 존재하는 환경에서는 COMET이 압도적인 정확도를 기록함을 확인했습니다.
* **Training Dynamics**: COMET과 같은 복잡한 아키텍처는 완전한 수렴을 위해 **50+ Epochs** 이상의 충분한 학습 횟수와 정교한 하이퍼파라미터 튜닝(Learning Rate, Dropout 등)이 필수적임을 도출했습니다.
* **Engineering Trade-offs**: 모델의 정확도 향상에 따른 추론 지연(Latency) 문제를 분석하였으며, 실무 적용을 위한 **모델 양자화(Quantization)** 및 **GPU 가속**의 필요성을 기술적으로 검토했습니다.

---

## 🔍 Detailed Implementation & Lab Report
데이터 전처리부터 PyTorch를 이용한 모델 구현, 하이퍼파라미터 최적화 및 상세 성능 지표(NDCG, Recall, Precision) 분석을 포함한 17개 장의 전체 실험 과정은 아래 리포트에서 확인하실 수 있습니다.

### 🔗 [Full Report: comet_deep_dive.ipynb](model-experiments/comet_deep_dive.ipynb)
*이 노트북은 모델의 배경 이론부터 최종 아키텍처 결론까지 아우르는 엄격한 기술 분석 내용을 담고 있습니다.*
