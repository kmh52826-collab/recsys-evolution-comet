# M-Trans4Rec: Adaptive Multi-Modal Fusion for Robust Sequential Recommendation

## 📝 Research Proposal (연구 제안 초록)

본 프로젝트는 **Sequential Recommendation(순차적 추천)** 의 고질적인 문제인 **Data Sparsity(데이터 희소성)** 를 해결하기 위해 고안된 새로운 모델 아키텍처 및 실험 설계 제안입니다.

기존 **COMET** 모델은 다층 임베딩 변환을 통해 도메인 간 지식을 활용하지만, 정보 통합 과정이 정적(Static)이어서 유저의 동적인 맥락을 반영하는 데 한계가 있다는 가설을 세웠습니다. 이에 본 제안서는 **Adaptive Gating Network**와 **Transformer-based Architecture**를 결합하여, 유저의 다양한 **Multi-modal(다중 모달)** 정보를 동적으로 융합하는 **M-Trans4Rec** 모델 구조를 설계하였습니다.

특히, 모델의 **Hidden Dimensions**를 **128 → 256 → 64**로 변화시키는 **Bottleneck Structure**를 통해 잠재적 특징(Latent Features)을 효과적으로 추출하는 메커니즘을 설계했습니다. 본 제안은 데이터 희소성이 99% 이상인 환경에서 **COMET** 대비 높은 정확도와 강건성을 확보하는 것을 목표로 하며, 이를 검증하기 위한 구체적인 **Experimental Protocol**과 평가지표(Recall, NDCG 등)를 포함하고 있습니다.

---

<br>

## 🚀 Proposed Improvements & Methodology (제안 방법론)

### 1. Dynamic Weighting Mechanism (가설)
COMET의 고정된 변환 방식 대신, 유저의 실시간 데이터에 따라 Sequential/Graph/Profile 정보의 비중을 스스로 조절하는 **Gating Network** 적용을 제안합니다.

### 2. Strategic Feature Expansion (설계)
**Dense Layers**를 활용해 특징 차원을 256으로 확장하여 정보 간의 비선형 관계를 충분히 학습한 후, 다시 64차원으로 압축하여 과적합(Overfitting)을 방지하고 예측력을 높이는 구조를 설계했습니다.

### 3. Systematic Evaluation Plan (실험 계획)
**Batch Size 256** 기반의 안정적인 학습 환경에서 기존 SOTA 모델들과의 성능 비교를 수행하기 위한 실험 설계를 마쳤으며, 특히 극심한 Cold-start 상황에서의 성능 방어력을 중점적으로 검증할 계획입니다.

---

<br>

## 💎 Research Contributions (연구 기여점)

본 제안 모델인 **M-Trans4Rec**은 기존 Sequential Recommendation 연구들이 가진 구조적 한계를 극복하기 위해 다음과 같은 세 가지 핵심적인 연구적 기여(Contributions)를 갖습니다.

### 1. Adaptive Multi-Modal Fusion via Gating Network
기존 연구(예: COMET)들이 다종 정보를 단순 합산(Summation)하거나 고정된 비율로 결합(Concatenation)했던 것과 달리, 본 모델은 **Adaptive Gating Network**를 도입하였습니다.
* **연구적 가치**: 유저마다 정보의 유효성이 다르다는 점(예: 어떤 유저는 과거 행동이 중요하고, 어떤 유저는 프로필 정보가 더 중요함)을 인지하고, 이를 모델이 데이터로부터 스스로 학습하여 **Context-aware**한 특징 융합을 가능하게 합니다.

### 2. Specialized Multi-Encoder Architecture for Heterogeneous Data
서로 다른 성격을 가진 세 가지 데이터 소스(Sequential, Graph, Side Info)를 개별적인 **Dedicated Encoders**를 통해 독립적으로 학습합니다.
* **연구적 가치**: 데이터의 이질성(Heterogeneity)을 인정하고 각 도메인에 최적화된 Representation을 먼저 추출함으로써, 정보 간의 간섭(Interference)을 최소화하고 각 데이터가 가진 고유한 패턴을 최대한 보존하는 **Modular Learning**을 실현합니다.

### 3. Strategic Feature Engineering with Expansion-Compression (128-256-64)
모델 내부의 Hidden Dimension을 의도적으로 확장(Expansion)했다가 다시 압축(Compression)하는 **Bottleneck 구조**를 설계했습니다.
* **연구적 가치**: 초기 Dense Layer에서 차원을 256으로 확장하여 다중 모달 정보 간의 **High-order 비선형 상호작용**을 충분히 모델링하고, 이후 점진적으로 압축하여 예측에 불필요한 노이즈를 제거합니다. 이는 특히 **Data Sparsity**가 높은 환경에서 모델의 Generalization(일반화) 능력을 극대화하는 핵심 기제입니다.

### 4. Robustness against Extreme Data Sparsity
99% 이상의 데이터 희소성을 가진 환경을 가정하여, 부족한 Interaction 데이터를 Graph와 Side Info가 보완하도록 설계되었습니다.
* **연구적 가치**: Cold-start 유저나 Interaction이 적은 아이템에 대해서도 **Robust Embedding**을 생성할 수 있음을 이론적으로 제시하며, 이는 실무적 추천 시스템이 직면한 가장 큰 난제를 해결하기 위한 연구적 대안을 제시합니다.

---

<br>

## 🏗 Framework Architecture (모델 구조)

M-Trans4Rec은 이질적인 데이터 소스로부터 개별적인 특징을 추출하는 **Parallel Multi-Encoder** 계층과, 이를 유기적으로 통합하는 **Adaptive Fusion** 계층으로 구성됩니다. 전체적인 데이터 흐름은 다음과 같습니다.

### 1. Model Overview (모델 개요)
본 모델은 유저의 과거 행동 데이터(Sequential), 유저-아이템 간의 복합 관계(Graph), 그리고 유저의 고유 속성(Side Info)을 동시에 입력받아 최종 추천 확률을 계산하는 **End-to-End Framework**입니다.

| Layer | Component | Function | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input Layer** | Raw Data | 상호작용 로그, 그래프 데이터, 프로필 정보 수집 | [Batch, Variable] |
| **Encoding Layer** | Multi-Encoders | 도메인별 특징 추출 (Sequential, Graph, Side Info) | [Batch, 3, 128] |
| **Fusion Layer** | Gating Network | 정보원별 중요도 가중치 산출 및 벡터 통합 | [Batch, 128] |
| **Hidden Layer** | Dense Blocks | 특징 차원 확장 및 비선형 관계 학습 (128→256→64) | [Batch, 64] |
| **Output Layer** | Score Predictor | 최종 아이템 선호도 확률(Scalar) 도출 | [Batch, 1] |

<br>

### 2. Encoder Specifications (인코더 상세 설명)

#### 🟢 Sequential Encoder (Behavioral Context)
* **역할**: 유저가 소비한 아이템의 시간적 순서(Temporal Order)를 분석하여 유저의 **Short-term Interest**를 파악합니다.
* **메커니즘**: Transformer의 Self-attention 구조를 활용하여, 아이템 간의 인과 관계와 맥락적 의존성을 학습합니다.
* **연구적 의의**: 단순한 구매 빈도가 아닌, '행동의 흐름' 속에 숨겨진 의도를 포착합니다.

#### 🔵 Graph Encoder (Structural Context)
* **역할**: 유저와 아이템, 혹은 유저와 유저 사이의 복잡한 연결망을 통해 **Collaborative Signal**을 추출합니다.
* **메커니즘**: LightGCN 혹은 Graph Sage 구조를 채택하여, 직접적인 상호작용이 없는 아이템이라도 그래프 상의 고차 이웃(High-order Neighbors) 정보를 통해 잠재적 선호도를 계산합니다.
* **연구적 의의**: Interaction 데이터가 부족한 상황(Sparsity)에서 주변 노드의 정보를 전파(Message Passing)받아 데이터 부족 문제를 완화합니다.

#### 🟡 Side Info Encoder (Attribute Context)
* **역할**: 유저의 인구통계학적 특성(성별, 나이, 지역 등)과 아이템의 메타 정보(카테고리, 브랜드 등)를 분석합니다.
* **메커니즘**: Multi-Layer Perceptron(MLP) 혹은 가공된 Embedding Layer를 통해 정적인 속성을 벡터화합니다.
* **연구적 의의**: 상호작용 기록이 전혀 없는 **Cold-start** 유저에 대해서도 최소한의 추천 근거를 제공합니다.

<br>

### 3. Systematic Data Flow (데이터 흐름)

1.  **Parallel Feature Extraction**: 256명의 유저(Batch) 데이터가 입력되면, 세 가지 인코더가 병렬적으로 각 유저를 **128차원의 개별 벡터**로 변환합니다.
2.  **Adaptive Weighting**: Gating Network가 세 전문가의 의견(128x3)을 검토하여, 현재 유저에게 가장 신뢰도 높은 정보에 높은 가중치를 부여하며 하나로 통합($z_{final}$)합니다.
3.  **Dimensional Manifold Learning**: 통합된 128차원 벡터는 **Dense 1(256)**에서 고차원으로 확장되어 복잡한 특징 결합을 학습한 뒤, **Dense 2(128)**와 **Dense 3(64)**를 거치며 핵심 정보로 압축됩니다.
4.  **Final Inference**: 최종적으로 64개의 정제된 특징은 가중치 벡터 $W_{out}$과 연산되어, 해당 아이템을 클릭할 확률을 의미하는 **스칼라값**으로 변환됩니다.
