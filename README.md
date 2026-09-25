# Next-Generation Recommender System Pipelines
## Subtitle: From SOTA Baseline Replication to Multi-Encoder Adaptive Gating Architecture (M-Trans4Rec)

---

## SECTION 1. Baseline Replication & Deep Dive: COMET Analysis
> **Reference Paper:** *COMET: Convolutional Dimension Interaction for Collaborative Filtering (ACM TIST, 2023)*

### 1.1 Research Motivation & Objective
This study was initiated to directly replicate the PyTorch-based architecture of **COMET**, a state-of-the-art (SOTA) recommendation model published in ACM TIST (2023). Through comparative experiments against standard benchmark models, namely **NeuMF (2017)** and **ConvNCF (2018)**, this research validates data transformation efficiency and multi-dimensional entity modeling characteristics in a controlled synthetic data environment.

### 1.2 🎯 Key Architectural Innovations of COMET
* **User/Item History Embedding Maps**: Moving away from the simple single User-Item pair feeding approach, COMET innovates the preprocessing pipeline by horizontally stacking the embeddings of a user's interacted items, and of an item's interacted users, into **two 2D 'embedding maps'**.
* **Internal & Dimensional Interaction via CNN**: To overcome the assumption in existing models that historical interactions and embedding dimensions are independent of each other, COMET applies CNNs with kernels of different sizes to simultaneously model interactions among historical interactions (internal interactions) and among embedding dimensions (dimensional interactions).
* **Representation Enrichment**: The CNN outputs are passed through an MLP to obtain two interaction vectors, which are then used to enrich the original user and item embeddings before the final prediction.

---

### 1.3 🖼️ Visual Guide: Understanding COMET Architecture
Below is the data flow analysis demonstrating how the COMET model transforms raw data into a multi-dimensional embedding map and processes it within the pipeline.

#### ① Full Architecture Overview
> The overall training process is designed to integrate history data from the input stage to extract high-level features.
<img width="2556" height="1326" alt="image" src="https://github.com/user-attachments/assets/f16b9479-0eb6-43bf-a639-a59c1b3e368c" />

#### ② The Concept of Embedding Maps
> COMET maps and transforms a user's historical interaction data into a 2D **'picture'** format rather than processing it as simple numerical values.
<img width="1390" height="427" alt="image" src="https://github.com/user-attachments/assets/0c7a94fe-7eba-415a-933d-d74cdcfa3b58" />

#### ③ Multi-Dimensional Interaction Cube
> Beyond simple two-dimensional relationships, the high-dimensional interactions combining User, Item, and Feature are modeled in a star-schema format.
<img width="2413" height="981" alt="image" src="https://github.com/user-attachments/assets/ab1c6467-822f-45de-ab56-a6fcc2094085" />

---

### 1.4 Detailed Implementation & Lab Report (PyTorch Experiment Summary)
In this experiment, a comprehensive 16-chapter technical investigation was conducted, covering data preprocessing pipeline design, PyTorch model layer implementation, hyperparameter configuration, and ranking evaluation metric analysis.

* **Framework & Dataset**: PyTorch 2.x + Cornac 2.3.5 / Synthetic Data (Power-law distribution, 1,000 Users, 500 Items, 15,000 Interactions, Density 3%)
* **Core Implementation Logic**: Custom implementation of ConvNCF's $32 \times 32$ Interaction Map generation module via `Outer Product`, and the history-embedding-map-based `Multi-layer 2D CNN` forward pass pipeline of the COMET variant.
* **📌 COMET Implementation Scope**: The COMET used in this experiment is a simplified COMET-style variant. It applies a 3-layer 2D CNN with max-pooling to a single user-side history map (the 10 most recent items + the target item), and does not include the paper's item-side embedding map or the step that enriches user and item embeddings. The COMET results below therefore reflect this simplified variant, not the performance of the original model.
* **🔗 Full Lab Report (Source Code)**: A rigorous Jupyter Notebook technical analysis report covering everything from background theories of these models to final architectural conclusions.
## 🔗 **[Full Report: comet_deep_dive.ipynb](model-experiments/comet_deep_dive.ipynb)**

---

## SECTION 2. Training Failure Analysis & Structural Limitations
> After directly implementing SOTA models and conducting a 5-Epoch benchmark test on synthetic data, the following **training failure and structural limitations** were derived.

### 2.1 Experimental Results & Empirical Discovery

| Model | NDCG@10 | NDCG@5 | Recall@10 | Training Time | Evaluation Time | Loss Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **NeuMF** | 0.4353 | 0.4281 | 0.4420 | **22.3s** | **1.4s** | 0.6931 $\rightarrow$ 0.4353 (Normal Convergence) |
| **ConvNCF** | **0.4422** | **0.4356** | **0.4436** | 72.3s | 127.5s | 0.6931 $\rightarrow$ 0.4616 (Normal Convergence) |
| **COMET** | *0.0286* | *0.0212* | *0.0456* | 35.6s | 44.7s | **0.6927 $\rightarrow$ 0.6929 (Training Failure / Stagnation)** |

### 2.2 Deep Root Cause Analysis of Pipeline Limitations
The experimental results revealed that the COMET model failed to converge normally, exhibiting extremely poor performance with an NDCG@10 of 0.0286 (Training Failure). Analysis indicates that this stems from a **mismatch between the characteristics of the source data and the model architecture**. In particular, the synthetic dataset has no temporal order, so the COMET variant, which represents users solely through their interaction histories, had no learnable signal. In addition, unlike the original paper, this implementation has no step that enriches user and item embeddings, leaving the model no other signal to rely on when the history signal is weak. The structural limitations of history-map-based CNN architectures identified through this analysis are as follows:

1. **Lack of Temporal Dependency**: Because embedding maps are scanned using fixed-size CNN kernels, long-term contextual dependencies within user action streams that dynamically change over time cannot be captured along the data flow. Furthermore, when randomly ordered logs without causal relationships (such as this synthetic dataset) are ingested, the CNN filter fails to learn meaningful context.
2. **Overlooking Structural Connectivity**: The model focuses solely on simple item consumption order, failing to reflect complex relational networks between entities or high-order structural connectivity topologies like co-click patterns into the feature layers. Under sparse data conditions (Density 3%), simple history listing leads to severe overfitting.
3. **Limitations of Static Context Integration**: When incorporating heterogeneous side information (e.g., user profiles or item attributes) that enriches representation capacity, a fixed combination method is used. This introduces a systemic limitation where contextual weights corresponding to real-time user status changes under large-scale traffic conditions cannot be dynamically reflected.

---

## SECTION 3. Research Proposal & Proposed Framework

This project analyzes the structural limitations of existing CNN-based sequential recommendation models from the perspective of data pipelines and architecture, and proposes the **M-Trans4Rec** framework, which efficiently resolves these issues using state-of-the-art Transformer and Graph Neural Network (GNN) components.

Dedicated Encoders (Sequence, Graph, Side Info) corresponding to three distinct data sources are placed in a parallel pipeline, and an **Adaptive Gating Network** is designed to dynamically integrate these heterogeneous data streams in real time. In particular, an **Expansion-Compression architecture** that transforms feature dimensions within hidden layers ($128 \rightarrow 256 \rightarrow 128 \rightarrow 64$) models complex non-linear interactions while strictly controlling computational overhead. This model aims to overcome these structural limitations and improve ranking quality (NDCG), particularly under sparse data conditions.

---

## SECTION 4. Proposed Improvements & Methodology

### 4.1 Dynamic Weighting Mechanism (Hypothesis)
Instead of COMET's static and fixed fusion method, we propose integrating an **Adaptive Gating Network** within the architecture to adaptively adjust the reflection weights of Sequential, Graph, and Side Info encoder output features based on real-time user data ingestion context (such as behavioral history length and data stream availability).

### 4.2 Strategic Structural Design with Hierarchical Expansion-Compression (Design)
The 128-dimensional data stream passing through the Adaptive Fusion layer is temporarily expanded to 256 dimensions upon entering the **Prediction Layer (MLP)** to fully model non-linear relationships among heterogeneous information sources. Subsequently, hierarchical compression is performed back to 128 and 64 dimensions, filtering unnecessary noise in the embedding space while maximizing downstream pipeline processing efficiency and prediction precision.

### 4.3 Systematic Evaluation Plan (Experimental Plan)
To verify the effectiveness of this model, performance comparisons against existing SOTA models must be conducted on real-world public benchmarks such as MovieLens-1M, Yelp, and Amazon Beauty, with sufficient training (50+ epochs with early stopping) and repeated runs across multiple seeds. Additionally, an experimental design is required to thoroughly evaluate how the data integration process across three dedicated encoders impacts recommendation quality and serving precision.

---

## SECTION 5. Framework Architecture
<img width="2553" height="1099" alt="image" src="https://github.com/user-attachments/assets/1565512f-b18b-4c4e-a1b8-d25d9edbbb8f" />

> The M-Trans4Rec framework consists of a **Parallel Multi-Encoder pipeline** layer that processes distinct features from heterogeneous data sources, and an **Adaptive Fusion** layer that organically integrates them. The overall data flow is structured as follows.

### Model Overview
This model is an **End-to-End data processing framework** that receives past user interaction logs (Sequential), complex user-item connection topologies (Graph), and intrinsic entity attribute metadata (Side Info) through an integrated pipeline to calculate final recommendation probabilities.

| Layer | Component | Function | Output Shape |
| :--- | :--- | :--- | :--- |
| **Input Layer** | Raw Data | Ingestion and collection of interaction sequence logs, graph topology data, and profile features | [Batch, Variable] |
| **Encoding Layer** | Multi-Encoders | Independent feature extraction and transformation per heterogeneous domain (Sequential, Graph, Side Info) | [Batch, 128] (Each) |
| **Fusion Layer** | Gating Network | Real-time importance weight calculation and dynamic feature fusion per information source | [Batch, 128] |
| **Hidden Layer** | Dense Blocks (MLP) | Feature dimension expansion and step-by-step compression for computational optimization (128 $\rightarrow$ 256 $\rightarrow$ 128 $\rightarrow$ 64) | [Batch, 64] |
| **Output Layer** | Score Predictor | Derivation of item preference probability (Scalar) for final serving | [Batch, 1] |

---

## SECTION 6. Encoder Specifications

### 🔵 Sequential Encoder (Behavioral Context)
<img width="2550" height="1278" alt="image" src="https://github.com/user-attachments/assets/5780cb81-f261-4bb5-80b6-08b5a06b2bf8" />

* **Role**: Analyzes the temporal order of item consumption logs entering as time-series streams to extract short-term and long-term user **behavioral contexts (Short-term & Long-term Dependencies)**.
* **Mechanism**: Integrates Transformer's self-attention structure within the pipeline layer to overcome the fixed-kernel limitations of existing CNN sweeping filters, flexibly handling causal relationships and temporal contexts across input data.
* **Research Significance**: Preserves temporal dependencies and continuity inherent within linear stream data without information loss.

<br>

### 🟠 Graph Encoder (Structural Context)
<img width="2346" height="1280" alt="image" src="https://github.com/user-attachments/assets/dff1703d-6f20-48ee-8038-c9395b9fc583" />

* **Role**: Aggregates **collaborative signals** from multi-dimensional connection network structures between user and item entities.
* **Mechanism**: Adopts LightGCN or GraphSAGE architectures to propagate and learn relational preferences through high-order neighbor relationships and co-consumption patterns (co-click topology) within the graph pipeline, even for entity pairs lacking explicit direct logs.
* **Research Significance**: Precisely injects high-dimensional structural connectivity that 1D sequence data easily misses into the data representation space.

<br>

### 🟢 Side Info Encoder (Attribute Context)
<img width="2361" height="1266" alt="image" src="https://github.com/user-attachments/assets/9ff7881f-67af-4cc7-aa32-9fbae6bd38e6" />

* **Role**: Analyzes heterogeneous feature sources, including demographic characteristics of users (e.g., gender, age) and metadata attributes of items (e.g., category, brand).
* **Mechanism**: Normalizes non-continuous and static attribute data into dense embeddings via processed Embedding Layers and static transformation blocks.
* **Research Significance**: Mitigates the cold-start problem arising when new entities without interaction histories enter the system, structurally resolving the lack of static metadata utilization within the pipeline.

---

## SECTION 7. Architectural Expected Benefits
> The M-Trans4Rec framework is designed to structurally resolve chronic pipeline bottlenecks and data sparsity issues in recommendation systems. Upon adoption, it aims to achieve the following three systemic effects and macro-level improvements as quantitative target hypotheses.

### 7.1 Pipeline Robustness Under Extreme Data Sparsity
* **Challenge**: Real-world public benchmark datasets—such as MovieLens-1M (95.5% sparsity), Yelp (98.7%), and Amazon Beauty (99.9%) targeted in benchmark planning—inevitably suffer from extreme sparsity due to insufficient user behavioral logs. Single sequence-based models suffer from structural vulnerabilities where overall quality degrades rapidly under such sparse embedding environments.
* **Expected Benefit**: By placing dedicated `Graph` and `Side Info` encoder channels as parallel layers, this architecture ensures serving stability based on topology signals, even when an individual user's linear sequence history is lost or fragmented in a cold state.

### 7.2 Precision Compensation for Cold-Start Entities & Dynamic Context Control
* **Challenge**: For newly registered users or newly ingested items with extremely sparse histories, history-map-based CNN architectures using fixed-size grid inputs can incur unnecessary padding computation overhead or overfit to noise features, leading to prediction failures.
* **Expected Benefit**: Incorporating the `Adaptive Gating Mechanism` allows encoder feature weights to be **dynamically adjusted according to user context**. Specifically, densely linking rich attribute metadata in cold-start environments aims to **improve final prediction accuracy**; the magnitude of improvement will be verified through future experiments.

### 7.3 Recommendation Diversity (Serendipity) & Serving Coverage Enhancement
* **Challenge**: Systems relying solely on popular item scanning or 1D consumption order logs create biased filter bubbles for users, causing system bottlenecks that reduce recommendation coverage and accelerate user churn.
* **Expected Benefit**: The `Graph Encoder` explicitly injects latent co-consumption patterns (co-click topology structure) between entities into the embedding space. By capturing high-dimensional latent relationships between items, it expands the exploration range for fresh items that users have not directly experienced, maximizing recommendation diversity (serendipity) alongside **Hit Rate improvements** at the final serving stage.

---

## 🎓 Closing Statement

This portfolio demonstrates a rigorous analytical process of implementing a simplified variant of the SOTA model COMET in PyTorch, observing its training failure, and identifying the limitations of history-map-based CNN architectures through data-characteristic-based root cause analysis, followed by proposing M-Trans4Rec—a next-generation framework designed to overcome these challenges.

The core value of this project extends beyond simple hyperparameter tuning; it lies in systematically structuring heterogeneous and sparse multi-source data (sequence logs, graph topologies, static metadata) from an efficient data modeling perspective, and designing a scalable information system architecture that dynamically joins data according to user contexts.

These experiences in heterogeneous data preprocessing, multi-dimensional information structuring, and systemic architectural analysis will serve as a solid foundation for flexibly and deeply conducting academic and technical research across various sub-domains, including intelligent system development, large-scale platform architecture optimization, and practical data applications.

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
* [7] He, X., et al. (2020). "**LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.**" *SIGIR*. [LightGCN]
* [8] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). "**Inductive Representation Learning on Large Graphs.**" *NeurIPS*. [GraphSAGE]

### 3. Recommender Systems
* [9] Lin, Z., et al. (2023). "**COMET: Convolutional Dimension Interaction for Collaborative Filtering.**" *ACM Transactions on Intelligent Systems and Technology (TIST)*. [COMET]
* [10] He, X., et al. (2018). "**Outer Product-based Neural Collaborative Filtering.**" *IJCAI*. [ConvNCF]
* [11] He, X., et al. (2017). "**Neural Collaborative Filtering.**" *World Wide Web Conference (WWW)*. [NCF]
* [12] Hidasi, B., et al. (2016). "**Session-based Recommendations with Recurrent Neural Networks.**" *ICLR*. [GRU4Rec]

### 4. Related Works & Surveys
* [13] Zhang, S., et al. (2019). "**Deep Learning Based Recommender System: A Survey and New Perspectives.**" *ACM Computing Surveys (CSUR)*.
* [14] Wu, Z., et al. (2020). "**A Comprehensive Survey on Graph Neural Networks.**" *IEEE Transactions on Neural Networks and Learning Systems*.
