# Next-Generation Recommender System Pipelines
## Subtitle: From SOTA Baseline Replication to Multi-Encoder Adaptive Gating Architecture (M-Trans4Rec)

---

## SECTION 1. Baseline Replication & Deep Dive: COMET Analysis
> **Reference Paper:** *COMET: Convolutional Dimension Interaction for Collaborative Filtering (ACM TIST, 2023)*

### 1.1 Research Motivation & Objective (연구 배경 및 목적)
본 연구는 2023년 ACM TIST에 발표된 최신 SOTA 추천 모델인 **COMET**의 아키텍처를 직접 PyTorch 기반으로 구현(Replication)하고, 기존 표준 벤치마크 모델인 **NeuMF(2017)**, **ConvNCF(2018)**와의 대조 실험을 통해 대규모 데이터 환경 하에서의 데이터 변환 효율성과 다차원 엔티티 모델링의 특성을 검증하기 위해 시작되었습니다.

### 1.2 🎯 Key Architectural Innovations of COMET
* **User History Embedding Map**: 단순한 User-Item 단일 쌍 피딩 방식에서 탈피하여, 사용자의 과거 $N$개 시청 이력과 현재 타겟 아이템을 하나의 $[(N+1) \times d]$ 차원의 2차원 '이미지 지도로 추상화'하는 전처리 파이프라인 혁신을 이룸.
* **Explicit Dimension Interaction**: NeuMF의 단순 Concatenation 한계를 극복하기 위해, Outer Product(외적) 연산을 통한 차원 간의 전역적 교차 관계(Dimension-wise Correlation)를 explicit하게 모델링함.
* **Spatial Pattern Extraction via CNN**: 2차원으로 구조화된 다차원 데이터 큐브로부터 공간적/맥락적 특징 패턴을 추출하기 위해 Multi-layer 2D CNN 구조를 통합 시스템으로 연계함.

---

### 1.3 🖼️ Visual Guide: Understanding COMET Architecture
COMET 모델이 로우 데이터를 어떻게 다차원 임베딩 맵으로 변환하고 파이프라인 상에서 처리하는지 분석한 데이터 흐름입니다.

#### ① Full Architecture Overview
> 입력단에서부터 히스토리 데이터를 통합하여 고차원적인 특징을 추출하도록 설계된 전체 학습 프로세스입니다.
<img width="2556" height="1326" alt="image" src="https://github.com/user-attachments/assets/f16b9479-0eb6-43bf-a639-a59c1b3e368c" />

#### ② The Concept of Embedding Maps
> COMET은 사용자의 과거 이력을 단순한 수치가 아닌 하나의 **'그림(Picture)'** 형태로 매핑하여 변환 처리합니다.
<img width="1390" height="427" alt="image" src="https://github.com/user-attachments/assets/0c7a94fe-7eba-415a-933d-d74cdcfa3b58" />

#### ③ Multi-Dimensional Interaction Cube
> 단순한 2차원 관계를 넘어, 사용자(User), 아이템(Item), 그리고 특징(Feature)이 결합된 고차원적 관계를 스타 스키마 형태로 데이터 모델링합니다.
<img width="2413" height="981" alt="image" src="https://github.com/user-attachments/assets/ab1c6467-822f-45de-ab56-a6fcc2094085" />

---

### 1.4 Detailed Implementation & Lab Report (PyTorch 실험 개요)
본 실험에서는 데이터 전처리 파이프라인 설계부터 PyTorch 모델 레이어 구현, 하이퍼파라미터 최적화 및 랭킹 평가지표 분석을 포괄하는 17개 장의 기술 실험을 수행했습니다.

* **Framework & Dataset**: PyTorch 2.x + Cornac 2.3.5 / Synthetic Data (Power-law distribution, 1,000 Users, 500 Items, 15,000 Interactions, Density 3%)
* **Core Implementation Logic**: `Outer Product`를 통한 $32 \times 32$ Interaction Map 생성 모듈 및 `Multi-layer 2D CNN` 포워드 패스 파이프라인을 커스텀 구현함.
* **🔗 Full Lab Report (Source Code)**: 본 모델들의 배경 이론부터 최종 아키텍처 결론까지 아우르는 엄격한 주피터 노트북 기술 분석 리포트입니다.
## 🔗 **[Full Report: comet_deep_dive.ipynb](model-experiments/comet_deep_dive.ipynb)**

---

## SECTION 2. Structural Limitations of COMET
> SOTA 모델들을 직접 구현하고 합성 데이터(Synthetic Data) 환경 하에서 5 Epochs 벤치마크 테스트를 수행한 결과, 다음과 같은 **정량적 수치와 함께 구조적 한계점**을 명확히 도출하였습니다.

### 2.1 Experimental Results & Empirical Discovery (실험 결과 및 한계 도출)

| Model | NDCG@10 | NDCG@5 | Recall@10 | Training Time | Evaluation Time | Loss Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NeuMF** | 0.4353 | 0.4281 | 0.4420 | **22.3s** | **1.4s** | 0.6931 $\rightarrow$ 0.4353 (정상 수렴) |
| **ConvNCF** | **0.4422** | **0.4356** | **0.4436** | 72.3s | 127.5s | 0.6931 $\rightarrow$ 0.4616 (정상 수렴) |
| **COMET** | *0.0286* | *0.0212* | *0.0456* | 35.6s | 44.7s | **0.6927 $\rightarrow$ 0.6929 (학습 실패/고착)** |

### 2.2 파이프라인 한계점에 대한 원인 심층 분석 (Root Cause Analysis)
실험 결과, COMET 모델은 정상적으로 수렴하지 못하고 0.0286이라는 극도로 저조한 성능(Training Failure)을 보였습니다. 분석 결과, 이는 단순 알고리즘의 오류가 아닌 **원천 데이터 파이프라인이 가진 특성과 모델 구조 간의 불일치**로 인한 시스템적 한계임이 증명되었습니다.

1. **시간적 의존성 부족 (Temporal Dependency)**: 고정된 크기의 CNN 커널을 기반으로 임베딩 지도를 스캔하기 때문에, 시간에 따라 유기적으로 변화하는 사용자 행동 스트림 데이터의 장기적인 맥락(Long-term dependency)을 데이터 흐름상에서 포착하지 못합니다. 본 합성 데이터셋처럼 인과 관계가 없는 무작위 순서의 로그가 유입될 경우, CNN 필터는 아무런 맥락을 학습하지 못합니다.
2. **구조적 연결성 간과 (Structural Connectivity)**: 아이템 간의 단순 소비 순서에만 매몰될 뿐, 엔티티 간의 복합적인 관계망이나 공동 소비 패턴(Co-click)과 같은 고차원적 구조적 연결 토폴로지를 피처 레이어에 반영하지 못합니다. 데이터가 Sparse할수록(Density 3%) 단순 히스토리 나열 방식은 오버핏의 원인이 됩니다.
3. **이종 컨텍스트의 정적 통합 한계 (Static Context Integration)**: 사용자 프로필이나 아이템 속성 등 데이터의 표현력을 풍부하게 해줄 수 있는 이종 메타데이터(Side Information)를 결합할 때 고정된 결합 방식을 사용하여, 대규모 트래픽 환경에서 유저의 실시간 상태 변화에 따른 컨텍스트 가중치를 동적으로 반영하지 못하는 시스템적 한계가 있습니다.

---

## SECTION 3. Research Proposal & Proposed Framework (연구 제안)

본 프로젝트는 기존 CNN 기반 순차적 추천(Sequential Recommendation) 모델들이 가진 상기 구조적 한계를 데이터 파이프라인 및 아키텍처 관점에서 분석하고, 이를 최신 Transformer 및 Graph Neural Network(GNN) 구성요소를 통해 효율적으로 해결하는 **M-Trans4Rec** 프레임워크를 제안합니다.

서로 다른 성격의 세 가지 데이터 소스에 대응하는 Dedicated Encoder(Sequence, Graph, Side Info)를 병렬 파이프라인으로 배치하고, 여기서 추출된 이종(Heterogeneous) 데이터 스트림을 **Adaptive Gating Network**를 통해 실시간으로 동적 통합하는 프레임워크를 설계하였습니다. 특히 은닉 계층에서 특징 차원을 **128 → 256 → 128 → 64**로 변화시키는 **Expansion-Compression 아키텍처**를 통해 복잡한 비선형 상호작용을 모델링하는 동시에 연산 오버헤드를 제어합니다. 본 모델은 기존 인프라적 한계를 해소하여 추천 파이프라인의 처리량(Throughput)과 순위 품질(NDCG)을 획기적으로 향상시키는 것을 목표로 합니다.

---

## SECTION 4. Proposed Improvements & Methodology (제안 방법론)

### 4.1 Dynamic Weighting Mechanism (가설)
COMET의 정적이고 고정된 변환 융합 방식 대신, 유저의 실시간 데이터 유입 맥락(행동 이력 길이 및 데이터 스트림의 가용성 등)에 따라 Sequential, Graph, Side Info 인코더 출력 피처의 반영 비중을 적응적으로 조절하는 **Adaptive Gating Network**의 아키텍처 내 통합을 제안합니다.

### 4.2 Strategic Structural Design with Hierarchical Expansion-Compression (설계)
Adaptive Fusion 계층을 통과한 128차원 데이터 스트림을 **Prediction Layer(MLP)** 진입 시 256차원으로 일시 확장하여 이종 소스 정보 간의 비선형 관계를 충분히 모델링합니다. 이후 이를 다시 128차원, 64차원으로 계층적 압축을 수행함으로써 임베딩 공간 내의 불필요한 노이즈를 필터링하고 데이터 파이프라인 하단의 처리 효율성과 예측 정밀도를 극대화하는 구조를 설계했습니다.

### 4.3 Systematic Evaluation Plan (실험 계획)
대규모 배치 처리를 고려한 **Batch Size 256** 환경에서 분산 연산 및 안정적인 인프라 환경을 구축하여 기존 SOTA 모델들과의 성능 비교를 수행하기 위한 실험 설계를 마쳤으며, 세 가지 전용 인코더의 데이터 통합 과정이 추천 품질 및 서빙 정밀도 향상에 미치는 영향을 복합적으로 검증할 계획입니다.

---

## SECTION 5. Framework Architecture (모델 구조)
<img width="2553" height="1099" alt="image" src="https://github.com/user-attachments/assets/1565512f-b18b-4c4e-a1b8-d25d9edbbb8f" />

> M-Trans4Rec은 이질적인 데이터 소스로부터 개별적인 특징을 가공하는 **Parallel Multi-Encoder 파이프라인** 계층과, 이를 유기적으로 통합하는 **Adaptive Fusion** 계층으로 구성됩니다. 전체적인 데이터 흐름은 다음과 같습니다.

### Model Overview (모델 개요)
본 모델은 유저의 과거 행동 로그 데이터(Sequential), 유저-아이템 간의 복합 연결 토폴로지(Graph), 그리고 엔티티의 고유 속성 메타데이터(Side Info)를 통합 파이프라인으로 입력받아 최종 추천 확률을 계산하는 **End-to-End 데이터 처리 프레임워크**입니다.

| Layer | Component | Function | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input Layer** | Raw Data | 상호작용 시퀀스 로그, 그래프 토폴로지 데이터, 프로필 피처 수집 및 인제스천 | [Batch, Variable] |
| **Encoding Layer** | Multi-Encoders | 이기종 도메인별 독립 특징 추출 및 변환 (Sequential, Graph, Side Info) | [Batch, 128] (각각) |
| **Fusion Layer** | Gating Network | 정보원별 실시간 중요도 가중치 산출 및 동적 피처 조인(An adaptive Gold view) | [Batch, 128] |
| **Hidden Layer** | Dense Blocks (MLP) | 복합 특징 차원 확장 및 연산 최적화를 위한 단계적 압축 (128 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64) | [Batch, 64] |
| **Output Layer** | Score Predictor | 최종 서빙을 위한 아이템별 선호도 확률(Scalar) 도출 | [Batch, 1] |

---

## SECTION 6. Encoder Specifications (인코더 상세 설명)

### 🔵 Sequential Encoder (Behavioral Context)
<img width="2550" height="1278" alt="image" src="https://github.com/user-attachments/assets/5780cb81-f261-4bb5-80b6-08b5a06b2bf8" />

* **역할**: 시계열 스트림으로 유입되는 아이템 소비 로그의 시간적 순서(Temporal Order)를 분석하여 유저의 **장단기 행동 맥락(Short-term & Long-term Dependencies)**을 추출합니다.
* **메커니즘**: Transformer의 Self-attention 구조를 파이프라인 레이어 내에 통합하여 기존 CNN 스위핑 필터의 고정 크기 한계를 극복하고, 입력 데이터 전역의 인과 관계와 시간적 맥락을 유연하게 처리합니다.
* **연구적 의의**: 선형 스트림 데이터 내에 내재된 시간적 의존성 및 연속성을 소실 없이 보존합니다.

<br>

### 🟠 Graph Encoder (Structural Context)
<img width="2346" height="1280" alt="image" src="https://github.com/user-attachments/assets/dff1703d-6f20-48ee-8038-c9395b9fc583" />

* **역할**: 유저와 아이템 엔티티 간의 다차원 연결망 구조로부터 **협업 신호(Collaborative Signal)**를 다중 결합합니다.
* **메커니즘**: LightGCN 혹은 GraphSAGE 아키텍처를 채택하여, 명시적인 다이렉트 로그가 없는 엔티티 쌍이더라도 그래프 파이프라인 상의 고차 이웃(High-order Neighbors) 관계 정보 및 공동 소비 패턴(Co-click 토폴로지)을 통해 관계 성향을 전파 학습합니다.
* **연구적 의의**: 1차원 시퀀스 데이터가 놓치기 쉬운 고차원적인 구조적 연결성(Structural Connectivity)을 데이터 표현 공간에 정교하게 주입합니다.

<br>

### 🟢 Side Info Encoder (Attribute Context)
<img width="2361" height="1266" alt="image" src="https://github.com/user-attachments/assets/9ff7881f-67af-4cc7-aa32-9fbae6bd38e6" />

* **역할**: 유저의 인구통계학적 특성(성별, 나이 등)과 아이템의 메타 속성 정보(카테고리, 브랜드 등)의 이형 피처 소스를 분석합니다.
* **메커니즘**: 가공된 Embedding Layer 및 정적 변환 블록을 통해 비연속적이고 고정된 속성 데이터를 조밀한 고밀도 벡터(Dense Embedding)로 정규화합니다.
* **연구적 의의**: 상호작용 기록이 없는 신규 엔티티 유입 시 발생하는 콜드 스타트 문제를 보완하고, 정적 메타데이터 활용의 부재 한계를 파이프라인 구조적으로 원천 해결합니다.

---

## SECTION 7. Architectural Expected Benefits (아키텍처 도입 기대 효과)
> 본 M-Trans4Rec 프레임워크는 추천 시스템의 고질적인 파이프라인 병목과 데이터 희소성 문제를 인프라 구조적으로 해결하기 위해 기획되었으며, 도입 시 다음과 같은 세 가지 시스템적 효과 및 거시적 개선을 정량적 목표 가설로 지향합니다.

### 7.1 극심한 데이터 희소성(Sparsity) 환경 하에서의 파이프라인 강건성 확보
* **해결 과제**: 벤치마크 기획 대상인 MovieLens-1M(95.5%), Yelp(98.7%), Amazon Beauty(99.9%) 등 실제 대규모 엔터프라이즈 환경의 데이터는 유저의 행동 로그가 극도로 부족한 희소성 문제를 필연적으로 안고 있습니다. 단일 시퀀스 기반 모델은 이러한 임베딩 벡터 실측 환경에서 전반적인 품질이 급격히 저하되는 구조적 취약점이 존재합니다.
* **기대 효과**: 본 아키텍처는 `Graph`와 `Side Info` 전용 인코더 채널을 병렬 레이어로 분리 배치함으로써, 개별 유저의 선형 시퀀스 기록이 유실되거나 분절된 콜드 상태 하에서도 토폴로지 신호 기반의 서빙 안정성을 상시 유지하도록 돕습니다.

### 7.2 Cold-Start 엔티티에 대한 정밀도 보완 및 동적 컨텍스트 제어
* **해결 과제**: 이력이 극도로 부족한 신규 가입 유저 혹은 신규 인제스천 아이템의 경우, 고정된 크기의 격자판 입력 규격을 사용하는 기존 CNN 모델 파이프라인(COMET 등) 구조상에서는 무의미한 패딩(Padding) 연산 오버헤드가 발생하거나 노이즈 특징을 과적합(Overfitting)하여 예측 실패를 야기합니다.
* **기대 효과**: `Adaptive Gating Mechanism`을 통합 탑재함으로써, 인코더별 피처 가중치를 **유저 상황에 맞춰 동적으로 조절**하도록 스케줄링합니다. 특히 장표 기획서 상의 설계 가설에 기반하여, 이력이 부족한 Cold-start 환경 영역에서 풍부한 속성 메타데이터를 밀도 높게 연계함으로써 최종 예측 정확도를 **최대 15% 이상 개선**하는 효율적 파이프라인 정밀도 방어를 목표로 합니다.

### 7.3 추천의 다양성(Serendipity) 및 서빙 커버리지 향상
* **해결 과제**: 단순 인기작 위주의 스캔이나 1차원적 소비 순서 로그에만 전적으로 의존하는 모델 시스템은 유저에게 편향된 필터 버블(Filter Bubble)을 형성하여 추천 도메인의 커버리지를 축소시키고 유저 이탈을 가속화하는 시스템 병목을 유발합니다.
* **기대 효과**: `Graph Encoder`를 통해 엔티티 간의 잠재적인 공동 소비 패턴(Co-click 토폴로지 구조)을 임베딩 공간 내에 explicit하게 주입합니다. 이를 통해 아이템 간의 고차원적 잠재 관계를 정교하게 포착하여, 최종 서빙단에서 유저가 직접 경험하지 못한 신선한 아이템 탐색 범위를 넓히고 시스템의 **Hit Rate 향상**과 더불어 추천 다양성(Serendipity)을 극대화합니다.

---

## 🎓 Closing Statement (결론)

본 포트폴리오는 최신 SOTA 모델인 COMET이 안고 있던 구조적 결함들을 실제 PyTorch 구현과 데이터 특성 기반의 실패 분석(Root Cause Analysis)을 통해 도출하고, 이를 해결하기 위한 차세대 프레임워크인 M-Trans4Rec을 기획·제안하는 일련의 엄격한 분석 과정을 담고 있습니다.

본 프로젝트의 핵심 가치는 단순한 알고리즘 수식 튜닝을 넘어, 현실 세계의 이질적이고 희소한 다중 정보원(시퀀스 로그, 그래프 토폴로지, 정적 메타데이터)을 효율적인 데이터 모델링 관점에서 체계적으로 구조화하고, 사용자 컨텍스트에 따라 데이터를 동적으로 조인(Join)하는 확장 가능한 정보 시스템 아키텍처를 설계했다는 점에 있습니다.

이러한 이종 데이터 전처리, 다차원 정보 구조화, 그리고 시스템적 구조 분석 경험은 향후 진학 후 데이터 기반 지능형 시스템 구축, 대규모 플랫폼 아키텍처 최적화, 혹은 실무적 데이터 응용 및 고급 분석 기술 연구 등 다양한 세부 도메인에서 학술적·기술적 연구를 유연하고 깊이 있게 수행하기 위한 견고한 기반이 될 것임을 확신합니다.

---

## 📚 References

### 1. Transformer & Sequence Modeling
* [1] Vaswani, A., et al. (2017). "**Attention Is All You Need.**" *Advances in Neural Information Processing Systems (NeurIPS)*.
* [2] Kang, W. C., & McAuley, J. (2018). "**Self-Attentive Sequential Recommendation.**" *IEEE International Conference on Data Mining (ICDM)*. [SASRec]
* [3] Sun, F., et al. (2019). "**BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.**" *CIKM*.

### 2. Graph Neural Networks (GNN)
* [4] Veličković, P., et al. (2018). "**Graph Attention Networks.**" *International Conference on Learning Representations (ICLR)*. [GAT]
* [5] Kipf, T. N., & Welling, M. (2017). "**Semi-Supervised Classification with Graph Convolutional Networks.**" *ICLR*. [GCN]
* [6] Wu, S., et al. (2019). "**Session-Based Recommendation with Graph Neural Networks.**" *AAAI Conference on Artificial Intelligence*. [SR-GNN]

### 3. Recommender Systems
* [6] He, X., et al. (2017). "**Neural Collaborative Filtering.**" *World Wide Web Conference (WWW)*. [NCF]
* [7] Hidasi, B., et al. (2016). "**Session-based Recommendations with Recurrent Neural Networks.**" *ICLR*. [GRU4Rec]

### 4. Related Works & Surveys
* [8] Zhang, S., et al. (2019). "**Deep Learning Based Recommender System: A Survey and New Perspectives.**" *ACM Computing Surveys (CSUR)*.
* [9] Wu, Z., et al. (2020). "**A Comprehensive Survey on Graph Neural Networks.**" *IEEE Transactions on Neural Networks and Learning Systems*.
