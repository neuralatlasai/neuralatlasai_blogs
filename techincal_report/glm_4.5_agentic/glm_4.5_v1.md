

# GLM-4.5: End-to-End Technical Report — Agentic, Reasoning, and Coding Foundation Models

---

## 1. Model Architecture

### 1.1 Formal Definition

GLM-4.5 is a Mixture-of-Experts (MoE) autoregressive language model parameterized by $\theta$, mapping input token sequences $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ to conditional next-token distributions:

$$p_\theta(x_{t+1} \mid x_1, \ldots, x_t) = \text{softmax}(\mathbf{W}_o \cdot h_t^{(L)})$$

where $h_t^{(L)} \in \mathbb{R}^{d}$ is the hidden state at layer $L$, $d = 5120$ for GLM-4.5, and $\mathbf{W}_o \in \mathbb{R}^{|V| \times d}$ is the output projection over vocabulary $V$.

**Design Principle:** Reduce model width (hidden dimension, number of routed experts per layer) and increase model height (number of layers) relative to comparable MoE architectures, based on the empirical finding that deeper models exhibit superior reasoning capacity.

### 1.2 Architecture Specification

| Parameter | GLM-4.5 | GLM-4.5-Air |
|---|---|---|
| Total Parameters | $355 \times 10^9$ | $106 \times 10^9$ |
| Activated Parameters | $32 \times 10^9$ | $12 \times 10^9$ |
| Dense Layers | 3 | 1 |
| MoE Layers | 89 | 45 |
| MTP Layers | 1 | 1 |
| Hidden Dim $d$ | 5120 | 4096 |
| Dense Intermediate Dim | 12288 | 10944 |
| MoE Intermediate Dim $d_{\text{ff}}^{\text{MoE}}$ | 1536 | 1408 |
| Attention Head Dim $d_h$ | 128 | 128 |
| Attention Heads $n_h$ | 96 | 96 |
| Key-Value Heads $n_{kv}$ | 8 | 8 |
| Total Experts $E$ | 160 | 128 |
| Active Experts $k$ | 8 | 8 |
| Shared Experts | 1 | 1 |
| QK-Norm | Yes | No |

**Key Invariant:** Activated parameter count at each forward pass equals parameters from dense layers + shared expert + $k=8$ selected routed experts + attention + embeddings, yielding $\approx 32$B for GLM-4.5.

### 1.3 Attention Mechanism: Grouped-Query Attention with Partial RoPE

#### 1.3.1 GQA Formulation

Given $n_h = 96$ query heads and $n_{kv} = 8$ key-value heads, the GQA group ratio is:

$$g = \frac{n_h}{n_{kv}} = \frac{96}{8} = 12$$

Each KV head is shared across 12 query heads. For token position $t$, layer $\ell$:

$$\mathbf{Q}_t^{(\ell)} = \mathbf{W}_Q^{(\ell)} h_t^{(\ell-1)} \in \mathbb{R}^{n_h \times d_h}$$

$$\mathbf{K}_t^{(\ell)} = \mathbf{W}_K^{(\ell)} h_t^{(\ell-1)} \in \mathbb{R}^{n_{kv} \times d_h}$$

$$\mathbf{V}_t^{(\ell)} = \mathbf{W}_V^{(\ell)} h_t^{(\ell-1)} \in \mathbb{R}^{n_{kv} \times d_h}$$

For query head $i$ belonging to group $\lfloor i/g \rfloor$:

$$\text{Attn}_i(t) = \text{softmax}\left(\frac{\mathbf{q}_t^{(i)} (\mathbf{K}^{(\lfloor i/g \rfloor)})^\top}{\sqrt{d_h}}\right) \mathbf{V}^{(\lfloor i/g \rfloor)}$$

**KV Cache Size per Layer per Token:**

$$\text{KV}_{\text{cache}} = 2 \times n_{kv} \times d_h \times \text{precision\_bytes} = 2 \times 8 \times 128 \times 2 = 4096 \text{ bytes (BF16)}$$

This is $12\times$ smaller than full MHA ($n_h = 96$), directly enabling long-context serving at 128K.

**Counterintuitive Finding on Head Count:** Using $2.5\times$ more attention heads (96 heads for $d=5120$, versus the standard $\sim$40 heads) does **not** improve training loss but **consistently improves downstream reasoning benchmarks** (MMLU, BBH). This suggests that higher head multiplicity provides a richer representational basis for multi-step inference patterns that are not captured by next-token loss alone.

#### 1.3.2 Partial RoPE

Rotary Position Embedding is applied to a subset of the head dimension. For head dimension $d_h = 128$, RoPE is applied to the first $d_r$ dimensions while the remaining $d_h - d_r$ dimensions receive no positional encoding:

$$\mathbf{q}_t^{(\text{rot})} = \text{RoPE}(\mathbf{q}_t[0:d_r], t), \quad \mathbf{q}_t^{(\text{pass})} = \mathbf{q}_t[d_r:d_h]$$

$$\mathbf{k}_t^{(\text{rot})} = \text{RoPE}(\mathbf{k}_t[0:d_r], t), \quad \mathbf{k}_t^{(\text{pass})} = \mathbf{k}_t[d_r:d_h]$$

The RoPE rotation matrix for position $t$ and dimension pair $(2i, 2i+1)$:

$$\text{RoPE}(t, i) = \begin{pmatrix} \cos(t \cdot \omega_i) & -\sin(t \cdot \omega_i) \\ \sin(t \cdot \omega_i) & \cos(t \cdot \omega_i) \end{pmatrix}$$

where $\omega_i = \beta^{-2i/d_r}$ and base frequency $\beta$:

- **Pre-training:** $\beta = 10{,}000$
- **Mid-training (32K+ context):** $\beta = 1{,}000{,}000$

This adjustment from $\beta = 10^4$ to $\beta = 10^6$ extends the effective context length by compressing the rotational frequency spectrum, allowing the model to distinguish positions at larger separations.

#### 1.3.3 QK-Norm

GLM-4.5 applies RMSNorm to query and key projections before computing attention logits:

$$\hat{\mathbf{q}} = \text{RMSNorm}(\mathbf{q}), \quad \hat{\mathbf{k}} = \text{RMSNorm}(\mathbf{k})$$

$$\text{RMSNorm}(\mathbf{z}) = \frac{\mathbf{z}}{\sqrt{\frac{1}{d}\sum_{i=1}^d z_i^2 + \epsilon}} \odot \gamma$$

**Purpose:** Prevents attention logit explosion during deep training (92 total layers), bounding $\hat{\mathbf{q}}^\top \hat{\mathbf{k}} \in [-d_h, d_h]$ regardless of hidden state magnitude growth. This is critical for numerical stability in 89-layer MoE stacks.

### 1.4 MoE Layer Design

#### 1.4.1 Loss-Free Balance Routing

Each MoE layer contains $E = 160$ routed experts and 1 shared expert. For input token representation $\mathbf{h} \in \mathbb{R}^d$, the routing mechanism computes:

$$\mathbf{s} = \sigma(\mathbf{W}_g \mathbf{h} + \mathbf{b}_{\text{route}}) \in \mathbb{R}^E$$

where $\sigma(\cdot)$ is the sigmoid function (not softmax), $\mathbf{W}_g \in \mathbb{R}^{E \times d}$, and $\mathbf{b}_{\text{route}} \in \mathbb{R}^E$ is the routing bias vector.

**Top-$k$ Selection:**

$$\mathcal{S}_k = \text{TopK}(\mathbf{s}, k=8)$$

**Gated Expert Output:**

$$\mathbf{y}_{\text{MoE}} = \underbrace{\text{FFN}_{\text{shared}}(\mathbf{h})}_{\text{shared expert}} + \sum_{i \in \mathcal{S}_k} s_i \cdot \text{FFN}_i(\mathbf{h})$$

**Loss-Free Balancing Mechanism:** The bias $\mathbf{b}_{\text{route}}$ is updated via an auxiliary rule (not gradient-based) to equalize expert load:

$$b_{\text{route}, i} \leftarrow b_{\text{route}, i} + \alpha \cdot \text{sign}(\bar{f} - f_i)$$

where $f_i$ is the empirical selection frequency of expert $i$, $\bar{f} = k/E$ is the target frequency, and $\alpha$ is the bias update rate:

- $\alpha = 0.001$ for the first 15T tokens
- $\alpha = 0.0$ for remaining tokens (bias frozen)

This avoids auxiliary load-balancing losses that distort the primary language modeling objective.

**Auxiliary Sequence-Level Balance Loss:** Despite loss-free routing, an additional regularizer prevents extreme within-sequence imbalance:

$$\mathcal{L}_{\text{bal}} = \lambda_{\text{bal}} \cdot \sum_{i=1}^{E} f_i^{(\text{seq})} \cdot p_i^{(\text{seq})}$$

where $f_i^{(\text{seq})}$ is the fraction of tokens in a sequence routed to expert $i$, $p_i^{(\text{seq})}$ is the mean routing probability for expert $i$ across the sequence, and $\lambda_{\text{bal}} = 0.0001$.

#### 1.4.2 Expert FFN Architecture

Each expert is an FFN with intermediate dimension $d_{\text{ff}}^{\text{MoE}} = 1536$:

$$\text{FFN}_i(\mathbf{h}) = \mathbf{W}_{\text{down}}^{(i)} \left[ \text{SiLU}(\mathbf{W}_{\text{gate}}^{(i)} \mathbf{h}) \odot (\mathbf{W}_{\text{up}}^{(i)} \mathbf{h}) \right]$$

where $\mathbf{W}_{\text{gate}}^{(i)}, \mathbf{W}_{\text{up}}^{(i)} \in \mathbb{R}^{d_{\text{ff}}^{\text{MoE}} \times d}$ and $\mathbf{W}_{\text{down}}^{(i)} \in \mathbb{R}^{d \times d_{\text{ff}}^{\text{MoE}}}$.

**Parameters per Routed Expert:**

$$P_{\text{expert}} = 3 \times d \times d_{\text{ff}}^{\text{MoE}} = 3 \times 5120 \times 1536 = 23{,}592{,}960 \approx 23.6\text{M}$$

**Total Routed Expert Parameters per MoE Layer:**

$$P_{\text{MoE-layer}} = E \times P_{\text{expert}} = 160 \times 23.6\text{M} \approx 3.77\text{B}$$

**Activated Expert Parameters per Token per MoE Layer:**

$$P_{\text{active}} = (k + 1) \times P_{\text{expert}} = 9 \times 23.6\text{M} \approx 212\text{M}$$

(including 1 shared expert)

### 1.5 Multi-Token Prediction (MTP) Layer

An additional MoE layer serves as the MTP head, predicting the next $M$ tokens simultaneously to support speculative decoding:

$$p_\theta(x_{t+1}, x_{t+2}, \ldots, x_{t+M} \mid x_{\leq t})$$

**MTP Loss:**

$$\mathcal{L}_{\text{MTP}} = -\frac{1}{M} \sum_{m=1}^{M} \log p_\theta(x_{t+m} \mid x_{\leq t})$$

**Combined Pre-Training Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda_{\text{MTP}} \cdot \mathcal{L}_{\text{MTP}} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{bal}}$$

where:
- $\lambda_{\text{MTP}} = 0.3$ for the first 15T tokens
- $\lambda_{\text{MTP}} = 0.1$ for remaining tokens

**Purpose at Inference:** The MTP layer generates $M$ draft tokens in parallel; the main model verifies them in a single forward pass (speculative decoding), yielding up to $M\times$ latency reduction when acceptance rate is high.

### 1.6 Layer Composition and Tensor Flow

**Full Forward Pass for Token $t$:**

```
PSEUDO-ALGORITHM: GLM-4.5 Forward Pass
────────────────────────────────────────
Input: token index x_t, position t, KV cache
Output: logits ∈ ℝ^|V|, updated KV cache

1. h_0 ← Embedding(x_t) ∈ ℝ^d                          // d = 5120
2. FOR ℓ = 1 TO L:                                       // L = 3 + 89 + 1 = 93
   a. h̃ ← RMSNorm(h_{ℓ-1})
   b. // ── Attention Block ──
      Q ← W_Q^(ℓ) h̃ ∈ ℝ^{n_h × d_h}                   // 96 × 128
      K ← W_K^(ℓ) h̃ ∈ ℝ^{n_kv × d_h}                   // 8 × 128
      V ← W_V^(ℓ) h̃ ∈ ℝ^{n_kv × d_h}
      IF QK-Norm enabled:
          Q ← RMSNorm(Q), K ← RMSNorm(K)
      Q_rot, K_rot ← Apply Partial RoPE(Q, K, t, β)
      Append K, V to KV cache for layer ℓ
      attn_out ← GQA(Q_rot, KV_cache^(ℓ))               // FlashAttention kernel
      h_attn ← W_O^(ℓ) attn_out
      h_mid ← h_{ℓ-1} + h_attn                          // residual connection
   c. h̃_mid ← RMSNorm(h_mid)
   d. // ── FFN/MoE Block ──
      IF layer ℓ is Dense:
          ffn_out ← DenseFFN(h̃_mid)                      // intermediate = 12288
      ELSE IF layer ℓ is MoE:
          s ← σ(W_g h̃_mid + b_route) ∈ ℝ^E              // sigmoid gating
          S_k ← TopK(s, k=8)
          ffn_out ← FFN_shared(h̃_mid) + Σ_{i∈S_k} s_i · FFN_i(h̃_mid)
      h_ℓ ← h_mid + ffn_out                              // residual connection
3. h_final ← RMSNorm(h_L)
4. logits ← W_o h_final ∈ ℝ^|V|
5. // MTP head (inference only, for speculative decoding)
   draft_logits ← MTP_MoE_Layer(h_final) ∈ ℝ^{M × |V|}
6. RETURN logits, draft_logits, updated KV cache
```

**Memory Flow Per Token (BF16):**

- Hidden state: $d \times 2 = 10{,}240$ bytes per layer
- KV cache per layer: $2 \times n_{kv} \times d_h \times 2 = 4{,}096$ bytes
- Total KV cache at 128K context: $4{,}096 \times 92 \times 131{,}072 \approx 49.4$ GB

### 1.7 Complexity Analysis

**FLOPs per Token (Forward Pass):**

For attention at sequence length $T$:

$$\text{FLOPs}_{\text{attn}} = 2 \times T \times d \times (n_h + 2n_{kv}) \times d_h + 2 \times n_h \times T^2 \times d_h$$

For activated MoE FFN per token:

$$\text{FLOPs}_{\text{MoE}} = 2 \times (k+1) \times 3 \times d \times d_{\text{ff}}^{\text{MoE}} = 2 \times 9 \times 3 \times 5120 \times 1536 \approx 0.42\text{G}$$

**Total FLOPs per Token across 92 layers (excluding embedding):**

$$\text{FLOPs}_{\text{total}} \approx 92 \times (\text{FLOPs}_{\text{attn}} + \text{FLOPs}_{\text{FFN}}) \approx 2 \times P_{\text{active}} \approx 64\text{G FLOPs}$$

This is comparable to a $\sim$32B dense model despite having 355B total parameters, yielding an **activation efficiency ratio** of $355/32 \approx 11.1\times$.

---

## 2. Data Pipeline

### 2.1 Pre-Training Corpus: 23T Tokens Total

#### 2.1.1 Web Data Processing

**Quality Bucketing (Nemotron-CC inspired):**

```
PSEUDO-ALGORITHM: Web Quality Bucketing
────────────────────────────────────────
Input: Crawled webpage corpus W
Output: Quality-stratified corpus W_filtered

1. FOR each document w ∈ W:
   a. score(w) ← QualityClassifier(w)          // trained classifier
   b. Assign w to bucket B_j where j = ⌊score(w) × N_buckets⌋
2. Discard B_0 (lowest quality bucket)
3. FOR each remaining bucket B_j:
   a. upsample_ratio(B_j) ← f(j)              // monotonically increasing
   b. Highest bucket: ~3.2 epochs exposure during pre-training
4. // Template-generated page removal
   Apply MinHash deduplication (standard LSH)
   Apply SemDedup pipeline:
     a. Compute document embeddings e(w) via encoder model
     b. Cluster embeddings via approximate nearest neighbor
     c. Within each cluster, retain most informative document
     d. Remove near-duplicate template pages
5. RETURN W_filtered
```

**Invariant:** High-quality buckets contribute $\geq 3.2$ epochs to emphasize high-frequency knowledge for reasoning; lower-quality buckets contribute $< 1$ epoch but maintain long-tail world knowledge coverage.

#### 2.1.2 Multilingual Data

- Sources: Crawled webpages + FineWeb-2
- Quality filter: Educational utility classifier trained on human annotations
- Strategy: Up-sample high-quality multilingual documents

#### 2.1.3 Code Data

```
PSEUDO-ALGORITHM: Code Data Curation
────────────────────────────────────────
Input: Raw code repositories from GitHub + code hosting platforms
Output: Quality-stratified code corpus C

1. Rule-based filtering:
   - Remove files < 50 bytes or > 1MB
   - Remove auto-generated files (detected via header patterns)
   - Remove files with entropy < threshold (likely binary)
2. Language-specific quality classification:
   - Train per-language quality models (e.g., Python, Java, C++)
   - Classify each file into {high, medium, low} quality tiers
3. Sampling strategy:
   - Up-sample high-quality tier
   - Include medium-quality tier at 1× rate
   - Exclude low-quality tier entirely
4. Apply Fill-In-the-Middle (FIM) objective:
   - For each code file, randomly select split point
   - Rearrange as: [PREFIX] [SUFFIX] [MIDDLE]
   - This enables infilling capabilities
5. Code-related web documents:
   a. Stage 1 retrieval:
      - Select documents with HTML <code> tags
      - OR documents classified as code-related by FastText classifier
   b. Stage 2 quality assessment:
      - Dedicated quality model → {high, medium, low}
      - Same sampling strategy as source code
   c. Fine-grained re-parsing to preserve code format
6. RETURN C
```

#### 2.1.4 Math & Science Data

- Source: Webpages, books, papers
- LLM scoring: Ratio of educational content about mathematics/science
- Small-scale classifier trained on LLM scores as pseudo-labels
- Documents above threshold are up-sampled in pre-training corpus

### 2.2 Data Composition Across Stages

| Stage | Tokens | Sequence Length | Primary Data |
|---|---|---|---|
| Pre-training Stage 1 | ~15T | 4,096 | General web documents |
| Pre-training Stage 2 | ~7T | 4,096 | Up-sampled code, math, science |
| Mid-training: Repo-level | ~500B | 32,768 | Concatenated repo files, issues, PRs |
| Mid-training: Synthetic Reasoning | ~500B | 32,768 | Synthetic CoT for math/science/code |
| Mid-training: Long-context + Agent | ~100B | 131,072 | Long documents, agent trajectories |

**Total: ~23T tokens**

### 2.3 Packing Strategy

- **Pre-training:** Random truncation (no best-fit packing) — acts as data augmentation for general documents
- **Mid-training:** Best-fit packing applied to avoid truncating reasoning processes or repo-level code, preserving logical completeness of training instances

---

## 3. Optimization Strategy

### 3.1 Optimizer: Muon

The Muon optimizer is applied to all parameters **except** word embeddings, bias terms, and RMSNorm weights (which use AdamW).

**Muon Update Rule:**

Given parameter matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$, gradient $\mathbf{G} = \nabla_{\mathbf{W}} \mathcal{L}$:

1. **Momentum update:**

$$\mathbf{M}_t = \mu \mathbf{M}_{t-1} + (1 - \mu) \mathbf{G}_t, \quad \mu = 0.95$$

2. **Newton-Schulz orthogonalization** ($N = 5$ iterations):

Starting from $\mathbf{X}_0 = \mathbf{M}_t / \|\mathbf{M}_t\|_F$:

$$\mathbf{X}_{i+1} = \frac{3}{2}\mathbf{X}_i - \frac{1}{2}\mathbf{X}_i \mathbf{X}_i^\top \mathbf{X}_i, \quad i = 0, \ldots, N-1$$

This converges to the orthogonal component $\mathbf{U} = \mathbf{X}_N$ such that $\mathbf{U}$ approximates the polar factor of $\mathbf{M}_t$.

3. **RMS scaling:**

$$\Delta \mathbf{W}_t = \eta \cdot \frac{\mathbf{U}}{\text{RMS}(\mathbf{U})} \times 0.2$$

where the update RMS is scaled to 0.2.

**Advantages over Adam:**
- Accelerates convergence empirically
- Tolerates larger batch sizes without degradation
- The Newton-Schulz orthogonalization normalizes the effective step direction, providing implicit preconditioning

### 3.2 Learning Rate Schedule

**Cosine Decay (not WSD):**

$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warmup}}}{T_{\text{total}} - t_{\text{warmup}}} \pi\right)\right)$$

where:
- $\eta_{\max} = 2.5 \times 10^{-4}$
- $\eta_{\min} = 2.5 \times 10^{-5}$
- Warmup: linear from 0 to $\eta_{\max}$
- Decay continues until end of mid-training

**Justification against WSD:** Early experiments showed WSD schedule causes underfitting during the stable phase, degrading general benchmarks (SimpleQA, MMLU). Cosine decay provides continuous learning rate reduction that better matches the diminishing marginal information in training data.

### 3.3 Batch Size Schedule

$$B(t) = \begin{cases} B_{\min} + \frac{t}{t_{\text{ramp}}}(B_{\max} - B_{\min}) & t < t_{\text{ramp}} \\ B_{\max} & t \geq t_{\text{ramp}} \end{cases}$$

- $B_{\min} = 16\text{M tokens}$
- $B_{\max} = 64\text{M tokens}$
- $t_{\text{ramp}}$: first 500B tokens

**Motivation:** Small initial batch size provides higher gradient noise (implicit regularization) during early training. Large final batch size maximizes throughput and provides stable gradients for fine-grained optimization.

### 3.4 Regularization

- Weight decay: $\lambda_{\text{wd}} = 0.1$
- Dropout: **None** (0.0 throughout)
- QK-Norm: Acts as implicit regularizer on attention logit magnitude

### 3.5 Hyperparameter Summary Table

| Hyperparameter | Value |
|---|---|
| Muon momentum $\mu$ | 0.95 |
| Newton-Schulz iterations $N$ | 5 |
| Update RMS scale | 0.2 |
| Peak LR $\eta_{\max}$ | $2.5 \times 10^{-4}$ |
| Min LR $\eta_{\min}$ | $2.5 \times 10^{-5}$ |
| Weight decay | 0.1 |
| Dropout | 0.0 |
| Batch size range | 16M → 64M tokens |
| RoPE base $\beta$ (pre-train / mid-train) | 10,000 / 1,000,000 |
| Routing bias update rate $\alpha$ | 0.001 → 0.0 (at 15T) |
| Balance loss weight $\lambda_{\text{bal}}$ | 0.0001 |
| MTP loss weight $\lambda_{\text{MTP}}$ | 0.3 → 0.1 (at 15T) |

---

## 4. Training Stages

### 4.1 Stage 1: General Pre-Training (15T tokens, ctx=4K)

**Objective:**

$$\mathcal{L}_{\text{Stage1}} = \underbrace{-\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})}_{\mathcal{L}_{\text{NTP}}} + \lambda_{\text{MTP}} \cdot \mathcal{L}_{\text{MTP}} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{bal}}$$

- **Input:** General web documents, multilingual data, code (initial mix), math/science
- **Context length:** 4,096 tokens
- **Packing:** Random truncation (data augmentation)
- **Data emphasis:** Quality-bucketed web data with highest bucket at ~3.2 epochs
- **Output:** Base model with broad knowledge and linguistic competence

**Failure Modes:**
- Template-generated web pages surviving MinHash → addressed by SemDedup
- Expert load imbalance → addressed by loss-free bias routing + sequence-level balance loss
- Attention logit explosion in deep network → addressed by QK-Norm

### 4.2 Stage 2: Code & Reasoning Continual Pre-Training (7T tokens, ctx=4K)

**Objective:** Same loss as Stage 1, with modified data distribution.

- **Data shift:** Up-sample GitHub source code, code-related web documents, math/science documents
- **FIM applied:** To all source code data for infilling capability
- **Context length:** Remains at 4,096
- **Output:** Model with strengthened code and reasoning foundations

### 4.3 Stage 3: Mid-Training — Repo-Level Code (500B tokens, ctx=32K)

**Context extension:** 4,096 → 32,768

**RoPE adjustment:** $\beta: 10{,}000 \rightarrow 1{,}000{,}000$

**Data composition:**
- Concatenated files from same repository (cross-file dependency learning)
- Model-filtered issues, pull requests (PRs), commits from GitHub
- PRs organized in diff-like format
- Related issues/PRs/commits concatenated into single context

**Packing:** Best-fit packing (no truncation of repo context)

### 4.4 Stage 4: Mid-Training — Synthetic Reasoning (500B tokens, ctx=32K)

**Data:**
- Questions from webpages and books for math, science, coding competitions
- Reasoning processes synthesized by a separate reasoning model (teacher)
- This constitutes synthetic Chain-of-Thought data at scale

**Purpose:** Bridge from pattern matching to structured multi-step reasoning before RL

### 4.5 Stage 5: Mid-Training — Long-Context & Agent (100B tokens, ctx=128K)

**Context extension:** 32,768 → 131,072

**Data:**
- Up-sampled long documents from pre-training corpus
- Large-scale synthetic agent trajectories (tool calls, multi-turn interactions)

**Packing:** Best-fit packing to preserve trajectory completeness

**Output of Mid-Training:** A base model ready for post-training with:
- 128K effective context
- Strong code/reasoning/agent foundations
- Cross-file software engineering understanding

---

## 5. Post-Training: Expert Model Iteration

### 5.1 Overview

Post-training is divided into two stages:

1. **Expert Training:** Construct domain-specialized expert models (Reasoning, Agent, General Chat) via cold-start SFT + domain-specific RL
2. **Unified Training:** Self-distillation to integrate experts into a single hybrid reasoning model, followed by general RL

### 5.2 Supervised Fine-Tuning (SFT)

#### 5.2.1 Cold-Start SFT

**Objective:**

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{D}|}\sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \sum_{t=1}^{|\mathbf{y}|} \log p_\theta(y_t \mid \mathbf{x}, y_{<t})$$

- Small set of SFT data with extended Chain-of-Thought responses
- Provides foundational chat, reasoning, and tool-use capabilities
- Serves as initialization point for RL training

#### 5.2.2 Overall SFT (Unified Training Stage)

**Data Sources:** Millions of samples covering:
- Reasoning tasks (math, code, science)
- General chat (writing, translation, summarization, chit-chat)
- Agentic tasks (tool using, project development)
- Long-context understanding tasks

**Context length:** Up to 128K tokens

**Hybrid Reasoning Design:**

The model learns to operate in two modes:
1. **Thinking mode:** Full CoT reasoning within `<think>...</think>` tags
2. **Non-thinking (direct response) mode:** Immediate response without explicit reasoning

This is achieved by balancing training data:
- Samples **with** full reasoning traces (complex tasks)
- Samples **without** explicit thought processes (simple queries, chit-chat)

The model learns to route internally based on task complexity.

#### 5.2.3 Function Call Template Design

**Problem:** Standard JSON function call templates require extensive character escaping for code parameters, increasing learning burden.

**Solution:** XML-like special token template:

```
<tool_call>{function-name}
  <arg_key>{key}</arg_key>
  <arg_value>{value}</arg_value>
  ...
</tool_call>
```

**Invariant:** Code content within `<arg_value>` tags requires zero escaping in the vast majority of cases, reducing token count and learning difficulty for agentic foundation models.

#### 5.2.4 Rejection Sampling Pipeline

```
PSEUDO-ALGORITHM: Multi-Stage Rejection Sampling
────────────────────────────────────────────────
Input: Expert model π_expert, prompt set P
Output: Filtered high-quality SFT dataset D_filtered

1. FOR each prompt p ∈ P:
   a. Sample N responses {y_1, ..., y_N} ~ π_expert(·|p)
   b. // Stage 1: Format Filtering
      Remove responses that are:
        - Repetitive (n-gram repetition > threshold)
        - Excessively short (|y| < min_length)
        - Truncated (missing end token)
        - Invalid reasoning format (malformed <think> tags)
   c. // Stage 2: Correctness Verification
      IF prompt has objective answer:
        Verify answer correctness programmatically
        Retain only correct responses
   d. // Stage 3: Reward Model Filtering
      IF prompt is subjective:
        Score responses via reward model
        Retain top-scoring responses
   e. // Stage 4: Tool Call Validation
      IF prompt involves tool calling:
        Verify tool invocation protocol compliance
        Verify trajectory reaches expected terminal state
        Retain only valid trajectories
2. RETURN D_filtered
```

#### 5.2.5 Prompt Selection and Response-Level Scaling

**Finding:** Removing the easiest 50% of prompts (bottom 50% by response length) yields 2-4% improvement on math/science tasks despite using only half the data.

**Response Scaling:** For hard prompts, generating $N = 4$ responses per prompt provides an additional 1-2% improvement via increased diversity of correct reasoning paths.

#### 5.2.6 Automatic Agentic SFT Data Construction

```
PSEUDO-ALGORITHM: Agentic SFT Data Pipeline
────────────────────────────────────────────
Input: Tool APIs, MCP servers, agentic frameworks
Output: High-quality agentic SFT trajectories

1. TOOL COLLECTION:
   - Gather real-world tool APIs and MCP servers
   - Use LLMs to automatically construct simulated tools
2. TASK SYNTHESIS:
   FOR each framework/tool set:
     - LLM comprehends tool functionality
     - Generates relevant queries/tasks
     - Covers single-step AND multi-step scenarios
3. TRAJECTORY GENERATION:
   FOR each synthesized task:
     - LLM generates tool-call trajectories
     - For multi-step tasks: LLM acts as user simulator
       → Converts to multi-round dialogue trajectories
4. QUALITY FILTERING:
   FOR each trajectory:
     - Multiple judge agents evaluate task completion
     - RETAIN only successful trajectories
5. RETURN validated trajectories
```

### 5.3 Reasoning RL

#### 5.3.1 Base Algorithm: GRPO (without KL term)

For each problem $x$, sample $K$ responses $\{y_1, \ldots, y_K\}$ from policy $\pi_{\text{old}}$. The GRPO objective (KL-free variant):

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{K} \sum_{i=1}^{K} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \min\left( r_t(\theta) \hat{A}_i, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

where:

$$r_t(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})}$$

**Group-relative advantage:**

$$\hat{A}_i = \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r(x) + \epsilon}$$

$$\bar{r}(x) = \frac{1}{K}\sum_{i=1}^{K} r(x, y_i), \quad \sigma_r(x) = \sqrt{\frac{1}{K}\sum_{i=1}^{K}(r(x, y_i) - \bar{r}(x))^2}$$

**Critical design choice:** No KL divergence penalty term — the clipping mechanism alone constrains policy drift.

#### 5.3.2 Difficulty-Based Curriculum Learning

```
PSEUDO-ALGORITHM: Two-Stage Difficulty Curriculum for RL
────────────────────────────────────────────────────────
Input: Problem pool P, initial policy π_0
Output: Final policy π_final

1. // Stage 1: Moderate Difficulty
   P_moderate ← {p ∈ P : 0 < pass@16(π_0, p) < 1}
   // Problems where model succeeds sometimes but not always
   samples_per_prompt ← 16
   Train π_1 via GRPO on P_moderate until convergence plateau

2. // Stage 2: Extreme Difficulty
   P_extreme ← {p ∈ P : pass@8(π_1, p) = 0 AND pass@512(π_1, p) > 0}
   // Problems unsolvable in 8 tries but solvable in 512
   // CONSTRAINT: All problems must have verified correct answers
   samples_per_prompt ← 512
   Train π_final via GRPO on P_extreme

3. RETURN π_final
```

**Rationale:** When all rewards are 1 (too easy) or all 0 (too hard), the advantage $\hat{A}_i$ has zero variance → zero gradient signal. The curriculum maintains reward variance throughout training.

**Empirical Result:** Stage 2 pushes AIME 24 Avg@32 from 81.8% to 83.4% on the experimental model.

#### 5.3.3 Single-Stage RL at 64K Output Length

**Finding:** Multi-stage RL with progressively increasing output length (16K → 32K → 48K → 64K) is **inferior** to single-stage RL at 64K directly.

**Mechanism of Failure:** Since cold-start SFT already conditions the model on 64K-length responses, introducing shorter maximum lengths during RL stages causes the model to "unlearn" long-context generation capability. The average output length decreases irreversibly, and recovery in the final 64K stage is limited.

**Empirical Evidence:** Single-stage at 64K: 83.4% AIME 24 Avg@32 vs. multi-stage: 80.6%.

#### 5.3.4 Dynamic Sampling Temperature

```
PSEUDO-ALGORITHM: Dynamic Temperature Adjustment
────────────────────────────────────────────────
Input: Current policy π, validation set V, temperature τ_current
Output: Updated temperature τ_next

1. Monitor rolling average reward R̄ over last W training steps
2. IF R̄ has stabilized (|ΔR̄| < δ for consecutive steps):
   // Convergence detected → increase exploration
   3. FOR τ_candidate ∈ {τ_current + Δτ, τ_current + 2Δτ, ...}:
      a. Evaluate π on V at temperature τ_candidate
      b. Compute performance P(τ_candidate)
   4. τ_next ← max{τ : P(τ) ≥ P(τ_current) × (1 - 0.01)}
   // Select maximum temperature with ≤1% performance drop
5. ELSE:
   τ_next ← τ_current
6. RETURN τ_next
```

**Design Principle:** As policy entropy decreases during training, a fixed temperature produces increasingly deterministic outputs. Dynamic adjustment maintains exploration while bounding quality degradation to $\leq 1\%$.

#### 5.3.5 Code RL: Token-Weighted Mean Loss

**Standard Sequence-Mean Loss:**

$$\mathcal{L}_{\text{seq}} = \frac{1}{K}\sum_{i=1}^{K} \hat{A}_i \cdot \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})$$

**Token-Weighted Mean Loss (preferred):**

$$\mathcal{L}_{\text{tok}} = \frac{1}{\sum_{i=1}^{K} |y_i|}\sum_{i=1}^{K} \hat{A}_i \cdot \sum_{t=1}^{|y_i|} \log \pi_\theta(y_{i,t} \mid x, y_{i,<t})$$

**Advantage:** Token-weighted loss provides finer-grained gradient signal, alleviates length bias inherent in sequence-level rewards, and suppresses generation of trivially short "base case" samples. Empirically achieves faster convergence (Figure 7, left).

#### 5.3.6 Science RL: Data Quality Paramount

**Finding on GPQA-Diamond:** Training exclusively on a small set of expert-verified multiple-choice questions significantly outperforms training on mixed-quality science data.

- Expert-verified MCQ only: 65.8% GPQA accuracy
- Mixed-quality data: 62.9% GPQA accuracy

**Implication:** For science RL, rigorous data curation dominates volume.

### 5.4 Agentic RL

#### 5.4.1 Data Collection and Synthesis

**Web-Search Tasks:**

Two approaches for generating demanding QA pairs:
1. **Automated pipeline:** Multi-hop reasoning over knowledge graphs → questions requiring synthesis across multiple web sources
2. **Human-in-the-loop:** Extraction and selective obfuscation of content from web pages → RL training signals

**Software-Engineering Tasks:**

- GitHub pull requests and issues curated into realistic SWE benchmark
- Each instance has user prompt + executable unit tests
- Execution in hardened sandbox with distributed system (horizontal scalability + isolation)

#### 5.4.2 Agentic RL Objective

Same GRPO framework as reasoning RL. For problem $x$, sample $K$ agent traces $\{y_1, \ldots, y_K\}$ from $\pi_{\text{old}}$:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{K}\sum_{i=1}^{K} \min\left( r_t(\theta)\hat{A}_i, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i \right) \right]$$

$$\hat{A}_i = \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r(x) + \epsilon}, \quad \bar{r}(x) = \frac{1}{K}\sum_{i=1}^{K} r(x, y_i)$$

**Critical constraint:** Only model-generated tokens are used for optimization; environment feedback tokens are excluded from loss computation.

#### 5.4.3 Reward Design

**Web Search — Outcome Supervision with Process Format Penalty:**

$$r(x, y) = \begin{cases} r_{\text{accuracy}}(\text{final\_answer}(y), \text{ground\_truth}(x)) & \text{if format valid throughout} \\ 0 & \text{if any tool call has invalid format (early halt)} \end{cases}$$

**Coding Agents — SWE Verification:**

$$r(x, y) = \begin{cases} 1 & \text{if all test cases pass} \\ 0 & \text{otherwise} \end{cases}$$

with same format penalty for invalid tool calls.

**Process Format Penalty Mechanism:** If the model generates an incorrectly formatted tool call at any step, the entire trace is halted and receives reward 0. This enforces structural correctness as a hard constraint.

#### 5.4.4 Iterative Self-Distillation

```
PSEUDO-ALGORITHM: Iterative Distillation for Agentic RL
───────────────────────────────────────────────────────
Input: Cold-start SFT model π_sft, task pool P
Output: Final agent policy π_final

1. π_rl^(1) ← GRPO_Train(π_sft, P, difficulty=easy)
   // RL training until plateau
2. FOR iteration k = 1, 2, ..., K_max:
   a. // Self-distillation step
      Generate new SFT data using π_rl^(k):
        D_distill ← {(x, y) : x ∈ P, y ~ π_rl^(k)(·|x), r(x,y) = 1}
      Replace original cold-start data with D_distill
      π_sft^(k+1) ← SFT_Train(π_base, D_distill)
   b. // RL on improved base, with increased difficulty
      P^(k+1) ← IncreaseDifficulty(P)
      π_rl^(k+1) ← GRPO_Train(π_sft^(k+1), P^(k+1))
3. RETURN π_rl^(K_max)
```

**Motivation:** RL on agent tasks is extremely time-consuming due to long-horizon environment interactions. Distillation provides a "ratchet" — each iteration produces a better cold-start, enabling RL to reach higher performance ceilings with progressively harder tasks.

#### 5.4.5 Test-Time Compute Scaling via Interaction Turns

For agent tasks, test-time compute scales through **interaction turns** with the environment rather than output token count:

$$\text{Performance}(n) \propto \log(n)$$

where $n$ is the number of interaction turns (e.g., search queries for web browsing, test-write-fix cycles for coding). BrowseComp accuracy scales smoothly from ~5% at 8 turns to ~30% at 128 turns.

### 5.5 General RL

#### 5.5.1 Holistic RL

**Data:** ~5,000 prompts spanning 7 primary, 33 secondary, 139 tertiary categories.

**Multi-Source Reward:**

$$r_{\text{holistic}}(x, y) = \alpha \cdot r_{\text{RM}}(x, y) + (1 - \alpha) \cdot r_{\text{AI}}(x, y)$$

where:
- $r_{\text{RM}}(x, y)$: Reward model trained on human preference annotations (pairwise comparisons across instruction following, safety, factual correctness)
- $r_{\text{AI}}(x, y)$: AI judge score using separate rubrics for:
  - Prompts with objective ground-truth answers: correctness-focused scoring
  - Open-ended prompts: multi-dimensional quality scoring

#### 5.5.2 Instruction Following RL

**Constraint Taxonomy:** 7 major categories, 151 minor constraint types (content requirements, formatting rules, length constraints, structural constraints, etc.)

**Hybrid Feedback System:**

$$r_{\text{IF}}(x, y) = w_1 \cdot r_{\text{rule}}(x, y) + w_2 \cdot r_{\text{RM}}(x, y) + w_3 \cdot r_{\text{critique}}(x, y)$$

- $r_{\text{rule}}$: Deterministic verification rules (exact format checking)
- $r_{\text{RM}}$: Trained reward model for soft quality assessment
- $r_{\text{critique}}$: Critique model providing detailed feedback signals

**Stability Result:** Up to ~1,000 GRPO training steps, no clear evidence of reward hacking was observed (SysBench-ISR improves monotonically from 64.8 to 77.2, Figure 9).

#### 5.5.3 Function Calling RL

**Step-wise Rule-based RL:**

For a trajectory of $T$ tool calls, the reward for each step $t$:

$$r(a_t) = \begin{cases} 1 & \text{if } \text{format}(a_t) \text{ is valid} \wedge a_t = a_t^* \text{ (exact match on name, params, fields)} \\ 0 & \text{otherwise} \end{cases}$$

where $a_t$ is the model-generated function call and $a_t^*$ is the ground truth.

**Strictness Invariant:** Exact match on function name, all parameter names, and all parameter values. This simultaneously enforces correctness and output formatting.

**End-to-End Multi-turn RL:**

$$r_{\text{e2e}}(I, \tau) = \text{TaskCompleted}(I, o_0, a_1, o_1, \ldots, a_T, o_T)$$

where:
- $I$: Original complex task specification
- $a_t$: $t$-th function call
- $o_t$: Tool feedback or user information at step $t$
- $\text{TaskCompleted}(\cdot)$: Binary signal from environment rules or LLM Judge Agent

**Task Types:**
1. **Single-turn multi-step:** Multiple sequential function calls within one turn
2. **Multi-turn multi-step:** Function calls + interaction with LLM-simulated user agent

#### 5.5.4 Pathology RL

**Target:** Rare failure modes (< 1% occurrence rate):
- Language mixing
- Excessive repetition
- Formatting mistakes

**Strategy:** Curate targeted dataset of prompts highly likely to trigger pathological behaviors. Apply penalty rewards during GRPO training on this dataset.

**Efficiency Rationale:** General RL over random prompts encounters these pathologies too rarely for efficient gradient signal. Targeted prompt curation concentrates the training signal.

---

## 6. RL Infrastructure: Slime

### 6.1 Architecture Overview

Three core modules:

1. **Training Engine (Megatron):** Reads from Data Buffer, executes GRPO parameter updates, synchronizes weights to Rollout Engine
2. **Rollout Engine (SGLang + Router):** Generates trajectories, computes rewards/verifier outputs, writes to Data Buffer
3. **Data Buffer:** Manages prompt initialization, custom data, rollout generation strategies; acts as bridge between Training and Rollout

### 6.2 Flexible Scheduling Modes

**Synchronous Colocated Mode (Reasoning/General RL):**
- Training and inference engines on same GPU workers
- Dynamic sampling reduces GPU idle time
- Optimal for tasks with relatively fast rollout generation

**Asynchronous Disaggregated Mode (Agentic RL):**
- Training GPUs and inference GPUs scheduled independently
- Rollout component exposed directly to agent environment
- Agent environments operate continuously without stalling on training cycles
- Implemented via Ray framework for resource scheduling

### 6.3 Mixed-Precision Inference for Accelerated Rollout

```
PSEUDO-ALGORITHM: Mixed-Precision Rollout
────────────────────────────────────────
1. After each policy update iteration:
   a. Quantize model parameters: W_fp8 ← BlockwiseFP8Quantize(W_bf16)
   b. Dispatch W_fp8 to rollout engines
2. Rollout engines perform FP8 inference for trajectory generation
3. Training engine maintains BF16 parameters for gradient computation
```

**Impact:** FP8 inference provides ~2× throughput improvement for rollout generation, which is the dominant bottleneck in RL training.

### 6.4 Agent-Oriented Infrastructure

**High-Concurrency Execution:**
- Docker-based runtime provisions isolated environments per task
- Reduces rollout overhead for environment setup/teardown

**Fully Asynchronous Training Loop:**
- Rollout engines continuously generate trajectories
- Training engines update weights and periodically sync to rollout engines
- Prevents long/diverse trajectories from blocking training pipeline

**Unified Interface for Heterogeneous Agent Frameworks:**
- HTTP endpoint interface + centralized data pool
- All agent frameworks produce rollouts in message-list format
- Stored in shared data pool with task-specific filtering
- Dynamic sampling strategies ensure data quality across diverse tasks

```
PSEUDO-ALGORITHM: Asynchronous Agentic RL Loop
──────────────────────────────────────────────
ROLLOUT WORKERS (continuous):
  WHILE training not done:
    1. Fetch prompt p from Data Buffer
    2. Provision isolated Docker environment for p
    3. Execute agent framework with π_current
    4. Generate trajectory τ = (o_0, a_1, o_1, ..., a_T, o_T)
    5. Compute reward r(τ)
    6. Write (p, τ, r) to Data Buffer

TRAINING WORKERS (periodic):
  WHILE training not done:
    1. Read batch B from Data Buffer (filtered, sampled)
    2. Compute GRPO loss on B
    3. Update π_θ parameters
    4. IF sync_interval reached:
       Broadcast W_bf16 to rollout workers
       Rollout workers: W_fp8 ← BlockwiseFP8Quantize(W_bf16)
```

---

## 7. Inference Path

### 7.1 Speculative Decoding via MTP Layer

The MTP (Multi-Token Prediction) MoE layer enables speculative decoding:

```
PSEUDO-ALGORITHM: Speculative Decoding with MTP
──────────────────────────────────────────────
Input: Prompt tokens x_{1:n}, main model M, MTP head D
Output: Generated sequence

1. Compute h_n ← M.forward(x_{1:n})              // Full forward pass
2. WHILE not done:
   a. // Draft phase (MTP head)
      Generate M draft tokens: ŷ_{1:M} ← D(h_n)   // Single forward, M outputs
   b. // Verify phase (main model)
      Compute logits for positions n+1 to n+M+1 in single forward pass
      FOR m = 1 TO M:
        IF ŷ_m matches verified token (acceptance criterion):
          Accept ŷ_m, advance position
        ELSE:
          Reject ŷ_m and all subsequent drafts
          Sample correct token from main model logits
          BREAK
   c. Update h for next iteration
3. RETURN generated sequence
```

**Throughput Gain:** When draft acceptance rate is $\alpha$, expected tokens per forward pass = $\frac{1 - \alpha^{M+1}}{1 - \alpha}$, providing up to $M\times$ latency reduction for high $\alpha$.

### 7.2 Hybrid Reasoning Mode at Inference

The model supports two inference modes:

1. **Thinking mode:** Model generates `<think>...</think>` block before the final response. Used for complex reasoning and agentic tasks.
2. **Non-thinking mode:** Direct response without explicit reasoning. Used for simple queries, chit-chat.

Mode selection is implicit (learned from training data distribution) or can be controlled via system prompt configuration.

### 7.3 Serving Configuration

For SWE-bench evaluation:
- Framework: OpenHands v0.34.0
- Max iterations: 100
- History truncation: 128K context limit
- Temperature: 0.6
- Top-$p$: 1.0

### 7.4 Test-Time Compute Scaling

**Reasoning tasks:** Scale via increased output token count (longer CoT)

**Agent tasks:** Scale via increased interaction turns with environment:
- BrowseComp: Accuracy scales from ~5% (8 turns) to ~30% (128 turns)
- Relationship: Approximately logarithmic in compute

$$\text{Accuracy} \approx a + b \cdot \log_2(\text{turns})$$

---

## 8. Evaluation Protocol

### 8.1 Agentic Benchmarks

| Benchmark | Metric | GLM-4.5 Score | Evaluation Details |
|---|---|---|---|
| TAU-Bench (Retail) | Task completion rate | 79.7% | Optimized user simulator (Figure 11) |
| TAU-Bench (Airline) | Task completion rate | 60.4% | Same simulator |
| BFCL V3 | Function call accuracy | 77.8% | User-defined function calling |
| BrowseComp | Answer accuracy | 26.4% | Web browsing agent |

**TAU-Bench User Simulator Design:** Carefully engineered prompt constraining:
- Single-line generation per turn
- Information rationing (only reveal needed info per step)
- Constraint handling (no assumption, extension, or generalization)
- Conversation termination protocol (`###STOP###`)

### 8.2 Reasoning Benchmarks

| Benchmark | Metric | GLM-4.5 | Evaluation Protocol |
|---|---|---|---|
| MMLU-Pro | EM | 84.6% | Standard |
| AIME 24 | Avg@32 | 91.0% | 32 samples, LLM answer validation |
| MATH-500 | EM | 98.2% | Standard |
| SciCode | Accuracy | 41.7% | Standard |
| GPQA | Avg@8 | 79.1% | 8 samples, LLM answer validation |
| HLE | Accuracy | 14.4% | Text-only, GPT-4o judge |
| LCB (2407-2501) | Pass@1 | 72.9% | Dynamic benchmark, date-ranged |

**Variance Mitigation:** AIME uses Avg@32 (32 independent samples), GPQA uses Avg@8 to reduce stochastic evaluation noise.

### 8.3 Coding Benchmarks

| Benchmark | Metric | GLM-4.5 | Framework |
|---|---|---|---|
| SWE-bench Verified | Resolve rate | 64.2% | OpenHands v0.34.0, 100 iterations |
| Terminal-Bench | Task completion | 37.5% | Terminus framework, standard function calling |
| CC-Bench | Head-to-head win rate | — | Claude Code framework, 52 tasks |

**CC-Bench Protocol:**
- 52 programming tasks across diverse domains
- Isolated containerized environments
- Human expert interactive evaluation
- Standardized initial prompts + iterative interactions
- Same expert, consistent strategy across all models
- Primary metric: Task completion
- Secondary metrics: Tool calling success rate, token efficiency

### 8.4 Safety Evaluation

SafetyBench: 11,435 multiple-choice questions across 7 categories.

| Category | GLM-4.5 Score |
|---|---|
| Ethics & Morality | 94.3% |
| Illegal Activities | 91.0% |
| Mental Health | 94.7% |
| Offensiveness | 83.0% |
| Physical Health | 96.7% |
| Privacy & Property | 92.0% |
| Unfairness & Bias | 77.4% |
| **Average** | **89.9%** |

### 8.5 Human Evaluation Protocol

**Dataset:** 660 prompts (392 English, 108 Chinese, 160 other languages)

**Categories:** Mathematics, Text Processing, Text Generation, Subjective QA, Objective QA, Logical Reasoning, Code Instructions

**Protocol:**
- Responses presented in randomized order (eliminate sequential bias)
- Single consistent evaluator per batch (minimize inter-annotator variance)
- Scoring: 0-10 scale
- Reasoning content (CoT) **not** shown to evaluators
- Factual prompts annotated with ground truth

### 8.6 Logical Reasoning Evaluation

**Contamination Mitigation:** Novel problems structurally different from internet-available logical questions. Each requires multiple deduction steps. Unified scoring standard per question. Human expert scoring.

**Result:** GLM-4.5 scores 62.0, competitive with Gemini 2.5 Pro (65.8) and DeepSeek-R1-0528 (62.1).

---

## 9. Compression and Quantization

### 9.1 FP8 Quantization for Inference

Applied during RL rollout and production serving:

**Block-wise FP8 Quantization:**

For parameter block $\mathbf{W}_{\text{block}} \in \mathbb{R}^{B_r \times B_c}$:

$$s = \frac{\max(|\mathbf{W}_{\text{block}}|)}{E4M3\_MAX}$$

$$\mathbf{W}_{\text{FP8}} = \text{round}\left(\frac{\mathbf{W}_{\text{block}}}{s}\right)$$

where $E4M3\_MAX = 448.0$ for FP8 E4M3 format.

**Information Preservation Invariant:**

$$\|\mathbf{W}_{\text{BF16}} - s \cdot \mathbf{W}_{\text{FP8}}\|_\infty \leq \frac{s}{2}$$

**Memory Reduction:** BF16 → FP8 yields $2\times$ memory compression per parameter, reducing total model weight from ~710 GB (BF16) to ~355 GB (FP8) for GLM-4.5.

### 9.2 KV Cache Compression via GQA

The GQA design with $n_{kv} = 8$ versus $n_h = 96$ provides implicit KV cache compression:

$$\text{Compression Ratio} = \frac{n_h}{n_{kv}} = 12\times$$

At 128K context, total KV cache:

$$\text{KV}_{\text{total}} = 2 \times n_{kv} \times d_h \times L_{\text{attn}} \times T \times \text{bytes} = 2 \times 8 \times 128 \times 92 \times 131072 \times 2 \approx 49.4 \text{ GB (BF16)}$$

versus full MHA ($n_{kv} = 96$): $\approx 593$ GB — making 128K context serving infeasible.

---

## 10. Convergence Dynamics and Training Stability

### 10.1 Pre-Training Convergence

**Muon + Cosine Decay:** The Muon optimizer's implicit orthogonal preconditioning combined with cosine decay provides monotonically decreasing loss throughout 23T tokens without the underfitting plateau observed with WSD schedules.

**Batch Size Warmup:** Small initial batch size ($16$M) provides high gradient noise → implicit regularization → better generalization on knowledge benchmarks. Transition to $64$M provides throughput scaling.

### 10.2 RL Convergence

**Reward Variance as Gradient Signal Proxy:**

$$\text{Var}[\hat{A}] = \text{Var}\left[\frac{r(x,y) - \bar{r}(x)}{\sigma_r(x)}\right]$$

When reward variance $\to 0$ (all rewards identical), $\hat{A}_i \to 0$ for all $i$, yielding zero effective gradient regardless of clipping. The difficulty-based curriculum maintains $\text{Var}[r] > 0$ throughout training.

**Instruction Following RL Stability:** Hybrid feedback system (rules + RM + critique model) provides reward robustness. SysBench-ISR improves from 64.8 to 77.2 over 1,000 steps without reward hacking evidence (Figure 9).

### 10.3 Failure Modes

| Failure Mode | Manifestation | Mitigation |
|---|---|---|
| Expert load collapse | All tokens routed to subset of experts | Loss-free bias routing + sequence-level balance loss |
| Attention logit explosion | NaN in deep layers | QK-Norm |
| Reward hacking (IF RL) | Reward increases but quality degrades | Hybrid feedback (rules + RM + critique) |
| Format degradation (agent RL) | Invalid tool calls | Process format penalty (zero reward on invalid format) |
| Context length unlearning | Output length decreases during staged RL | Single-stage RL at target length |
| Pathological outputs | Language mixing, repetition | Targeted pathology RL with curated trigger prompts |
| Template page contamination | Low-diversity training data | SemDedup on document embeddings |
| Underfitting in stable LR phase | Poor SimpleQA/MMLU | Cosine decay instead of WSD |

---

## 11. Deployment Constraints

### 11.1 Memory Budget

| Component | BF16 | FP8 |
|---|---|---|
| Model weights (355B) | ~710 GB | ~355 GB |
| KV cache (128K, 92 layers) | ~49.4 GB | ~24.7 GB |
| Activation memory (per token) | Negligible (streaming) | — |
| **Total minimum** | **~760 GB** | **~380 GB** |

Minimum serving configuration: 5× H100 80GB (FP8) or 10× H100 80GB (BF16), requiring tensor parallelism and/or pipeline parallelism.

### 11.2 Throughput Characteristics

- Activated FLOPs per token: ~64 GFLOPs
- Equivalent to ~32B dense model in compute
- MoE routing overhead: negligible (sigmoid + TopK)
- Speculative decoding via MTP: additional latency reduction proportional to acceptance rate

### 11.3 Pareto Efficiency

GLM-4.5 and GLM-4.5-Air lie on the Pareto frontier of SWE-bench Verified score vs. total model parameters among open-source models:

- GLM-4.5: 355B params, 64.2% SWE-bench → best performance/size ratio at this scale
- DeepSeek-R1: 671B params (~2× larger), 41.4% SWE-bench
- Kimi K2: 1043B params (~3× larger), 65.4% SWE-bench

### 11.4 Overall Performance Rankings

| Domain | GLM-4.5 Rank | Key Competitors |
|---|---|---|
| **Overall (12 ARC benchmarks)** | 3rd | Behind o3, Grok 4 |
| **Agentic** | 2nd | Behind o3 |
| **Reasoning** | Competitive 3rd-4th | Close to DeepSeek-R1-0528 |
| **Coding** | 3rd | Close to Claude Sonnet 4 |
| **Average across all 12** | 3rd | With ~$\frac{1}{2}$× params of DeepSeek-R1, ~$\frac{1}{3}$× of Kimi K2 |

---

## 12. Key Technical Contributions and Design Decisions Summary

### 12.1 Architecture

- **Deeper-narrower MoE:** 92 layers with $d=5120$ vs. wider-shallower alternatives → improved reasoning capacity
- **2.5× attention heads:** 96 heads for $d=5120$ → no training loss improvement but consistent reasoning benchmark gains
- **Sigmoid gating + loss-free balance routing:** Avoids auxiliary loss distortion of primary objective
- **QK-Norm:** Stabilizes 92-layer deep attention

### 12.2 Training

- **Muon optimizer:** Accelerated convergence via Newton-Schulz orthogonalization, tolerates large batch sizes
- **Cosine decay over WSD:** Prevents underfitting on knowledge benchmarks
- **5-stage progressive training:** General → Code/Reasoning → Repo-level → Synthetic reasoning → Long-context/Agent
- **23T total tokens** with careful quality stratification

### 12.3 Post-Training

- **Expert model iteration:** Train domain experts → self-distill into unified hybrid reasoning model
- **Difficulty-based curriculum for RL:** Two-stage with verified answer constraint
- **Single-stage 64K RL:** Avoids irreversible context length regression
- **Dynamic sampling temperature:** Quality-controlled exploration maintenance
- **Token-weighted loss for code RL:** Finer-grained gradient, faster convergence
- **Iterative self-distillation for agentic RL:** Progressive difficulty ratchet
- **Multi-source general RL:** Rules + human RM + AI judge + critique model

### 12.4 Infrastructure

- **Slime framework:** Synchronous (reasoning) + asynchronous (agentic) dual-mode RL
- **BF16 training + FP8 rollout:** Online block-wise quantization for 2× rollout throughput
- **Docker-isolated agent environments:** High-concurrency, horizontally scalable
- **Unified HTTP interface:** Heterogeneous agent framework integration via shared data pool