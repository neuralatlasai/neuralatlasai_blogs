

# GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models — End-to-End Technical Report

---

## 1. Data Pipeline

### 1.1 Objectives, Inputs, Outputs, Invariants

**Objective:** Construct a 23T-token pre-training corpus with controlled quality distribution, domain coverage (web, code, math, science, multilingual), deduplication, and domain-specific upsampling to maximize downstream reasoning, coding, and agentic performance.

**Inputs:**
- Raw internet crawls (English, Chinese, multilingual)
- GitHub repositories (source code, issues, PRs, commits)
- Books, academic papers
- Social media documents
- Fineweb-2 multilingual corpus
- Open-source agentic datasets (AgentGym, MCP servers)

**Outputs:**
- Quality-bucketed, deduplicated, domain-tagged token streams with per-document quality scores, language labels, and domain annotations
- Packed sequences at lengths 4K, 32K, 128K depending on training stage

**Invariants:**
- High-quality bucket documents appear $\geq 3.2$ epochs; lowest-quality bucket documents are discarded entirely
- MinHash deduplication followed by SemDedup (embedding-based) to remove template-generated near-duplicates
- Fill-In-the-Middle (FIM) objective applied to all source code
- No best-fit packing during pre-training (random truncation as augmentation); best-fit packing enforced during mid-training

**Failure Modes:**
- Template-generated pages passing MinHash but detected by SemDedup
- Quality classifier assigning high scores to low-educational-value content
- Domain imbalance causing catastrophic forgetting of long-tail knowledge

---

### 1.2 Web Data Processing Pipeline

**Stage 1 — Quality Scoring and Bucketing:**

Documents $d \in \mathcal{D}_{\text{web}}$ are scored by a quality classifier $q(d) \in [0, 1]$. The score space is partitioned into $B$ buckets:

$$\mathcal{B}_k = \{d \mid \tau_{k-1} \leq q(d) < \tau_k\}, \quad k = 1, \ldots, B$$

Upsampling weight for bucket $k$:

$$w_k = \begin{cases} \alpha_k > 1 & \text{if } k \geq k_{\text{high}} \\ 1 & \text{if } k_{\text{low}} < k < k_{\text{high}} \\ 0 & \text{if } k \leq k_{\text{low}} \end{cases}$$

The highest-quality bucket contributes $>3.2$ epochs. This emphasizes high-frequency knowledge for reasoning while maintaining coverage for long-tail world knowledge.

**Stage 2 — Deduplication:**

- **MinHash deduplication:** Standard Jaccard-similarity-based shingling with locality-sensitive hashing (LSH). Pairs with estimated Jaccard $\hat{J}(d_i, d_j) > \theta_{\text{minhash}}$ are deduplicated.
- **SemDedup:** Document embeddings $\mathbf{e}_d = f_{\text{embed}}(d) \in \mathbb{R}^{D}$ are computed. Cosine similarity clusters are formed:

$$\text{sim}(d_i, d_j) = \frac{\mathbf{e}_{d_i}^\top \mathbf{e}_{d_j}}{\|\mathbf{e}_{d_i}\| \|\mathbf{e}_{d_j}\|}$$

Documents with $\text{sim}(d_i, d_j) > \theta_{\text{sem}}$ within a cluster are reduced to a single representative. This catches template-generated pages that MinHash misses due to minor lexical variation.

**Stage 3 — Multilingual Quality Filtering:**

A quality classifier judging educational utility is applied. High-quality multilingual documents are upsampled. Sources: crawled webpages + Fineweb-2.

---

### 1.3 Code Data Processing Pipeline

**Step 1 — Rule-based Filtering:** Language detection, license filtering, file-size thresholding, boilerplate removal.

**Step 2 — Language-specific Quality Classification:** Each code sample $c$ is classified into three tiers: $\{$high, medium, low$\}$ by language-specific quality models $q_{\text{code}}^{\ell}(c)$ where $\ell$ denotes programming language.

- High-quality: upsampled
- Medium-quality: included at base rate
- Low-quality: excluded

**Step 3 — Fill-In-the-Middle (FIM) Objective:** All source code is reformatted for FIM training. Given a code document partitioned at a random split point into prefix $p$, middle $m$, suffix $s$:

$$\text{FIM}(c) = [\texttt{<PRE>}] \, p \, [\texttt{<SUF>}] \, s \, [\texttt{<MID>}] \, m$$

The model learns to predict $m$ given $(p, s)$, enabling infilling capabilities.

**Step 4 — Code-related Web Documents:**

Two-stage retrieval from text pre-training corpus:
1. Initial selection: documents with HTML code tags OR identified by a FastText classifier for code-related content
2. Quality assessment by a dedicated classifier into $\{$high, medium, low$\}$, same upsampling strategy
3. Fine-grained parser re-parses selected web pages to preserve code format and content

---

### 1.4 Math & Science Data Processing

Documents from webpages, books, papers are scored by an LLM for the ratio of educational content about mathematics and science. A small-scale classifier $q_{\text{math}}(d)$ is trained to predict these scores. Documents with $q_{\text{math}}(d) > \tau_{\text{math}}$ are upsampled.

---

### 1.5 Pseudo-Algorithm: Data Processing Pipeline

```
ALGORITHM: DataPipeline
Input: Raw crawl D_raw, GitHub repos R, books B, papers P
Output: Quality-scored, deduplicated, domain-tagged corpus C

1. FOR each source S in {D_raw, R, B, P}:
     a. Apply source-specific rule-based filtering
     b. Language detection → assign lang(d)
     c. Domain tagging → assign domain(d)

2. FOR d in D_web:
     a. q(d) ← QualityClassifier(d)
     b. Assign to bucket B_k based on q(d)
     c. Discard if k ≤ k_low

3. MinHash deduplication on D_web
4. Compute embeddings e_d = f_embed(d) for all d in D_web
5. SemDedup: cluster by cosine similarity, retain representatives

6. FOR c in D_code:
     a. Rule-based filter
     b. q_code(c) ← LanguageSpecificQualityModel(c)
     c. Discard if low quality
     d. Apply FIM transformation

7. FOR d in D_web:
     a. IF has_code_tags(d) OR FastTextCodeClassifier(d) > τ:
          i. q_web_code(d) ← CodeWebQualityModel(d)
          ii. Re-parse with fine-grained parser
          iii. Include if not low quality

8. FOR d in D_math_science:
     a. q_math(d) ← MathScienceClassifier(d)
     b. Upsample if q_math(d) > τ_math

9. FOR d in D_multilingual:
     a. q_multi(d) ← EducationalUtilityClassifier(d)
     b. Upsample if high quality

10. Merge all sources → C with per-document weights w(d)
11. RETURN C
```

---

## 2. Model Architecture

### 2.1 Overview

GLM-4.5 is a Mixture-of-Experts (MoE) Transformer with the following specifications:

| Parameter | GLM-4.5 | GLM-4.5-Air |
|---|---|---|
| Total Parameters | 355B | 106B |
| Activated Parameters | 32B | 12B |
| Dense Layers | 3 | 1 |
| MoE Layers | 89 | 45 |
| MTP Layers | 1 | 1 |
| Hidden Dim $d_{\text{model}}$ | 5120 | 4096 |
| Dense FFN Dim $d_{\text{ffn}}^{\text{dense}}$ | 12288 | 10944 |
| MoE FFN Dim $d_{\text{ffn}}^{\text{moe}}$ | 1536 | 1408 |
| Attention Head Dim $d_h$ | 128 | 128 |
| Attention Heads $n_h$ | 96 | 96 |
| KV Heads $n_{\text{kv}}$ | 8 | 8 |
| Total Experts $E$ | 160 | 128 |
| Active Experts $k$ | 8 | 8 |
| Shared Experts | 1 | 1 |
| QK-Norm | Yes | No |

---

### 2.2 Architectural Design Principles

#### 2.2.1 Depth vs. Width Trade-off

GLM-4.5 departs from DeepSeek-V3 (58 MoE layers, 7168 hidden dim) and Kimi K2 (60 MoE layers, 7168 hidden dim) by **reducing width** (hidden dimension 5120, 160 routed experts with FFN dim 1536) and **increasing depth** (89 MoE layers + 3 dense layers + 1 MTP layer = 93 total layers).

**Rationale:** Empirically, deeper models exhibit better reasoning capacity. The total depth of 93 layers significantly exceeds DeepSeek-V3 (62 layers) and Kimi K2 (62 layers).

#### 2.2.2 Increased Attention Head Count

GLM-4.5 uses $n_h = 96$ attention heads for $d_{\text{model}} = 5120$, yielding $d_h = d_{\text{model}} / n_h \approx 53.3$. However, the specification states $d_h = 128$, implying the attention projection dimension differs from $d_{\text{model}}$:

$$d_{\text{attn}} = n_h \times d_h = 96 \times 128 = 12288$$

This means the attention module projects to a higher-dimensional space than $d_{\text{model}}$, with $d_{\text{attn}} / d_{\text{model}} = 2.4$, approximately 2.5× more heads than a standard configuration where $d_{\text{attn}} = d_{\text{model}}$.

**Key finding:** This increased head count does **not** improve training loss but **consistently improves** performance on reasoning benchmarks (MMLU, BBH). This suggests the multi-head structure provides richer representational diversity for downstream evaluation despite similar loss landscape properties.

#### 2.2.3 Grouped-Query Attention (GQA) with Partial RoPE

**GQA configuration:**
- Query heads: $n_h = 96$
- KV heads: $n_{\text{kv}} = 8$
- GQA group size: $g = n_h / n_{\text{kv}} = 12$

Each KV head serves 12 query heads. The KV cache per layer per token:

$$\text{KV cache per layer} = 2 \times n_{\text{kv}} \times d_h = 2 \times 8 \times 128 = 2048 \text{ elements}$$

For 93 layers at sequence length $L$:

$$\text{Total KV cache} = 93 \times L \times 2048 \times \text{sizeof(dtype)}$$

At $L = 131072$ in BF16:

$$93 \times 131072 \times 2048 \times 2 \approx 49.9 \text{ GB}$$

**Partial RoPE:** Rotary Position Embeddings are applied to a subset of the head dimensions. Given head dimension $d_h = 128$, RoPE is applied to the first $d_{\text{rope}}$ dimensions, while the remaining $d_h - d_{\text{rope}}$ dimensions receive no positional encoding.

For query $\mathbf{q}_m$ and key $\mathbf{k}_n$ at positions $m, n$:

$$\text{Attn}(m, n) = \text{Re}\left[\sum_{j=0}^{d_{\text{rope}}/2-1} (\mathbf{q}_m^{(j)} + i\mathbf{q}_m^{(j+d_{\text{rope}}/2)}) \cdot (\mathbf{k}_n^{(j)} + i\mathbf{k}_n^{(j+d_{\text{rope}}/2)})^* \cdot e^{i(m-n)\theta_j}\right] + \mathbf{q}_{m,\text{nope}}^\top \mathbf{k}_{n,\text{nope}}$$

where $\theta_j = \beta^{-2j/d_{\text{rope}}}$ and base frequency $\beta = 10000$ during pre-training, $\beta = 1000000$ during mid-training for long-context extension.

#### 2.2.4 QK-Norm

Applied in GLM-4.5 (not GLM-4.5-Air). Before computing attention logits:

$$\hat{\mathbf{q}} = \frac{\mathbf{q}}{\text{RMSNorm}(\mathbf{q})}, \quad \hat{\mathbf{k}} = \frac{\mathbf{k}}{\text{RMSNorm}(\mathbf{k})}$$

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

This stabilizes attention logit magnitudes, preventing entropy collapse in deep networks (93 layers) where unnormalized QK products can grow unbounded.

---

### 2.3 MoE Layer Architecture

#### 2.3.1 Loss-Free Balance Routing

Standard MoE routing introduces auxiliary balance losses that can distort the primary training objective. Loss-free balance routing maintains load balance without an auxiliary loss by dynamically adjusting routing biases.

Given router logits for token $t$ and expert $e$:

$$z_{t,e} = \mathbf{h}_t^\top \mathbf{w}_e$$

A bias term $b_e$ is added:

$$\hat{z}_{t,e} = z_{t,e} + b_e$$

Expert selection via top-$k$:

$$\mathcal{S}_t = \text{Top-}k(\{\hat{z}_{t,e}\}_{e=1}^{E})$$

The bias $b_e$ is updated based on expert load imbalance:

$$b_e \leftarrow b_e + \eta_b \cdot (\bar{f} - f_e)$$

where $f_e$ is the fraction of tokens routed to expert $e$, $\bar{f} = k/E$ is the target fraction, and $\eta_b$ is the bias update rate.

**Schedule:**
- $\eta_b = 0.001$ for first 15T tokens
- $\eta_b = 0.0$ for remaining tokens (biases frozen)

#### 2.3.2 Sigmoid Gating

Instead of softmax gating over selected experts, sigmoid gates are used:

$$g_{t,e} = \sigma(\hat{z}_{t,e}) = \frac{1}{1 + e^{-\hat{z}_{t,e}}}, \quad e \in \mathcal{S}_t$$

Expert output:

$$\mathbf{y}_t = \sum_{e \in \mathcal{S}_t} g_{t,e} \cdot \text{FFN}_e(\mathbf{h}_t)$$

Sigmoid gating allows each expert's contribution to be independently scaled $\in (0, 1)$ rather than forced to sum to 1 (softmax), providing more flexible expert combination.

#### 2.3.3 Shared Expert

One shared expert is always active for every token, providing a baseline capacity:

$$\mathbf{y}_t = \text{FFN}_{\text{shared}}(\mathbf{h}_t) + \sum_{e \in \mathcal{S}_t} g_{t,e} \cdot \text{FFN}_e(\mathbf{h}_t)$$

Total activated parameters per token: 1 shared expert + 8 routed experts = 9 expert FFN blocks, plus attention.

#### 2.3.4 Auxiliary Sequence-Level Balance Loss

Despite loss-free routing, an auxiliary sequence-level balance loss is applied with weight $\alpha = 0.0001$ to avoid extreme imbalance within a single sequence:

$$\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{e=1}^{E} f_e^{\text{seq}} \cdot P_e^{\text{seq}}$$

where $f_e^{\text{seq}} = \frac{1}{T}\sum_{t=1}^{T} \mathbf{1}[e \in \mathcal{S}_t]$ is the fraction of tokens in the sequence routed to expert $e$, and $P_e^{\text{seq}} = \frac{1}{T}\sum_{t=1}^{T} \frac{g_{t,e}}{\sum_{e'} g_{t,e'}}$ is the average gating probability.

---

### 2.4 Multi-Token Prediction (MTP) Layer

An additional MoE layer serves as the MTP layer, predicting $M$ future tokens simultaneously.

For position $t$, the MTP head predicts tokens at positions $t+1, t+2, \ldots, t+M$:

$$P(x_{t+m} | \mathbf{h}_t) = \text{softmax}(\mathbf{W}_{\text{out}}^{(m)} \cdot \text{MoE}_{\text{MTP}}(\mathbf{h}_t^{(m)}))$$

MTP loss:

$$\mathcal{L}_{\text{MTP}} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{m=1}^{M} \log P(x_{t+m} | \mathbf{h}_t)$$

Total pre-training loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \lambda \mathcal{L}_{\text{MTP}} + \mathcal{L}_{\text{balance}}$$

**Schedule for $\lambda$:**
- $\lambda = 0.3$ for first 15T tokens
- $\lambda = 0.1$ for remaining tokens

**Inference use:** The MTP layer enables speculative decoding, where the MTP head generates $M$ draft tokens that are verified in parallel by the main model, yielding throughput improvement proportional to the draft acceptance rate $\alpha_{\text{accept}}$:

$$\text{Speedup} \approx \frac{M \cdot \alpha_{\text{accept}}}{1 + M \cdot c_{\text{verify}}/c_{\text{decode}}}$$

---

### 2.5 Full Forward Pass Tensor Flow

For a single token at position $t$ with input embedding $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$:

```
ALGORITHM: ForwardPass
Input: Token embedding x_t ∈ ℝ^{d_model}, position t
Output: Logits over vocabulary V

1. h_t^{(0)} ← x_t

2. FOR layer l = 1 to L (L = 93 for GLM-4.5):
   a. // Self-Attention with GQA + Partial RoPE + QK-Norm
      h̃_t^{(l)} ← RMSNorm(h_t^{(l-1)})
      Q ← h̃_t^{(l)} W_Q^{(l)}  ∈ ℝ^{n_h × d_h}   [96 × 128]
      K ← h̃_t^{(l)} W_K^{(l)}  ∈ ℝ^{n_kv × d_h}  [8 × 128]
      V ← h̃_t^{(l)} W_V^{(l)}  ∈ ℝ^{n_kv × d_h}  [8 × 128]

      // Apply partial RoPE to first d_rope dims of Q, K
      Q_rope, Q_nope ← split(Q, d_rope)
      K_rope, K_nope ← split(K, d_rope)
      Q_rope ← apply_RoPE(Q_rope, t, β)
      K_rope ← apply_RoPE(K_rope, t, β)
      Q ← concat(Q_rope, Q_nope)
      K ← concat(K_rope, K_nope)

      // QK-Norm (GLM-4.5 only)
      IF qk_norm:
        Q ← RMSNorm(Q)
        K ← RMSNorm(K)

      // GQA: expand K, V to match Q heads
      K_expanded ← repeat_interleave(K, g=12)  [96 × 128]
      V_expanded ← repeat_interleave(V, g=12)  [96 × 128]

      // Scaled dot-product attention (FlashAttention kernel)
      A ← softmax(Q K_expanded^T / √d_h) V_expanded
      O ← A W_O^{(l)}  ∈ ℝ^{d_model}

      h_t^{(l,attn)} ← h_t^{(l-1)} + O

   b. // FFN: Dense or MoE
      h̃_t^{(l,ffn)} ← RMSNorm(h_t^{(l,attn)})

      IF layer l is dense:
        FFN_out ← SwiGLU(h̃_t^{(l,ffn)} W_1^{(l)}, h̃_t^{(l,ffn)} W_3^{(l)}) W_2^{(l)}

      ELSE IF layer l is MoE:
        // Router
        z_e ← h̃_t^{(l,ffn)}^T w_e + b_e   for e = 1..E
        S_t ← Top-k({z_e}, k=8)
        g_e ← σ(z_e)  for e ∈ S_t

        // Shared expert
        shared_out ← FFN_shared(h̃_t^{(l,ffn)})

        // Routed experts
        routed_out ← Σ_{e ∈ S_t} g_e · FFN_e(h̃_t^{(l,ffn)})

        FFN_out ← shared_out + routed_out

      h_t^{(l)} ← h_t^{(l,attn)} + FFN_out

3. h_final ← RMSNorm(h_t^{(L)})
4. logits ← h_final W_vocab  ∈ ℝ^{|V|}
5. RETURN logits
```

---

### 2.6 Complexity Analysis

**Attention complexity per layer:**

$$\mathcal{O}_{\text{attn}} = \mathcal{O}(L \cdot n_h \cdot d_h \cdot T) = \mathcal{O}(T^2 \cdot d_{\text{attn}})$$

With FlashAttention, memory complexity reduces from $\mathcal{O}(T^2)$ to $\mathcal{O}(T)$ while maintaining $\mathcal{O}(T^2 \cdot d_{\text{attn}})$ compute.

**MoE FFN complexity per layer:**

For each token, 9 expert FFNs (1 shared + 8 routed) are computed:

$$\mathcal{O}_{\text{MoE}} = 9 \times \mathcal{O}(d_{\text{model}} \times d_{\text{ffn}}^{\text{moe}}) = 9 \times \mathcal{O}(5120 \times 1536) \approx 9 \times 15.7\text{M FLOPs}$$

**Dense FFN complexity per layer:**

$$\mathcal{O}_{\text{dense}} = \mathcal{O}(d_{\text{model}} \times d_{\text{ffn}}^{\text{dense}}) = \mathcal{O}(5120 \times 12288) \approx 125.8\text{M FLOPs}$$

**Total FLOPs per token (approximate):**

$$\text{FLOPs}_{\text{total}} \approx L_{\text{dense}} \cdot (\mathcal{O}_{\text{attn}} + \mathcal{O}_{\text{dense}}) + L_{\text{MoE}} \cdot (\mathcal{O}_{\text{attn}} + \mathcal{O}_{\text{MoE}}) + \mathcal{O}_{\text{MTP}}$$

The activated parameter count of 32B determines the per-token compute budget, substantially lower than the 355B total.

---

## 3. Training Pipeline

### 3.1 Pre-Training Stage (Stage 1): General Corpus

**Data:** 15T tokens from general web, multilingual, code, math/science
**Sequence length:** 4096
**RoPE base frequency:** $\beta = 10000$

**Optimizer:** Muon optimizer for all parameters except:
- Word embeddings
- Bias terms
- RMSNorm weights

(These use standard AdamW or similar.)

#### 3.1.1 Muon Optimizer

The Muon optimizer applies Newton-Schulz iterations for preconditioning:

Given gradient $\mathbf{G} \in \mathbb{R}^{m \times n}$, the preconditioned update $\hat{\mathbf{G}}$ is computed via $N$ iterations of Newton-Schulz to approximate $\mathbf{G} (\mathbf{G}^\top \mathbf{G})^{-1/2}$:

$$\mathbf{X}_0 = \frac{\mathbf{G}}{\|\mathbf{G}\|_F}$$

$$\mathbf{X}_{i+1} = \mathbf{X}_i \left(\frac{3\mathbf{I} - \mathbf{X}_i^\top \mathbf{X}_i}{2}\right), \quad i = 0, \ldots, N-1$$

After $N = 5$ iterations, $\mathbf{X}_N$ approximates the orthogonalized gradient.

With momentum $\mu = 0.95$:

$$\mathbf{M}_t = \mu \mathbf{M}_{t-1} + (1 - \mu) \mathbf{G}_t$$

$$\hat{\mathbf{M}}_t = \text{NewtonSchulz}_N(\mathbf{M}_t)$$

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \cdot \text{scale} \cdot \hat{\mathbf{M}}_t$$

where $\text{scale}$ is set so that the update RMS equals 0.2.

**Properties:**
- Accelerates convergence compared to AdamW
- Tolerates larger batch sizes (critical for 64M token batches)
- Newton-Schulz with $N=5$ is sufficient for practical preconditioning

#### 3.1.2 Learning Rate Schedule

Cosine decay (not WSD):

$$\eta_t = \begin{cases} \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} & t \leq T_{\text{warmup}} \\[6pt] \eta_{\min} + \frac{\eta_{\max} - \eta_{\min}}{2}\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \cdot \pi\right)\right) & t > T_{\text{warmup}} \end{cases}$$

where $\eta_{\max} = 2.5 \times 10^{-4}$, $\eta_{\min} = 2.5 \times 10^{-5}$.

**Rationale for cosine over WSD:** WSD schedule led to worse performance on general benchmarks (SimpleQA, MMLU), indicating underfitting during the stable phase. Cosine decay maintains a gradual reduction throughout, avoiding premature convergence.

#### 3.1.3 Batch Size Warmup

$$B_t = \begin{cases} B_{\min} + (B_{\max} - B_{\min}) \cdot \frac{t}{T_{\text{batch\_warmup}}} & t \leq T_{\text{batch\_warmup}} \\[6pt] B_{\max} & t > T_{\text{batch\_warmup}} \end{cases}$$

where $B_{\min} = 16\text{M tokens}$, $B_{\max} = 64\text{M tokens}$, $T_{\text{batch\_warmup}}$ corresponds to the first 500B tokens.

#### 3.1.4 Regularization

- Weight decay: $\lambda_{\text{wd}} = 0.1$
- Dropout: none (0.0)
- No auxiliary balance loss beyond the sequence-level term ($\alpha = 0.0001$)

---

### 3.2 Pre-Training Stage (Stage 2): Code & Reasoning Upsampling

**Data:** 7T tokens with upsampled source code from GitHub, coding-related webpages, math and science content
**Sequence length:** 4096
**All other hyperparameters continue from Stage 1**

The domain mixture shifts to emphasize:
- Source code (GitHub repositories)
- Code-related web documents
- Mathematics and science documents

This constitutes a form of domain-focused continual pre-training within the pre-training phase itself.

---

### 3.3 Mid-Training Stage 1: Repo-Level Code Training

**Data:** ~500B tokens of concatenated code files from same repository + model-filtered issues, PRs, commits
**Sequence length:** 32K (extended from 4K)
**RoPE base frequency:** $\beta = 1{,}000{,}000$ (extended from 10,000)

**Objectives:**
- Learn cross-file dependency in repositories
- Software engineering capability (understanding issues → PRs → commits)
- Commits organized in diff-like format

**Data format:**
- Related issues, PRs, commits concatenated into single context
- Repository files concatenated preserving directory structure

**Packing:** Best-fit packing (no random truncation) to avoid truncating reasoning processes or repo-level code.

---

### 3.4 Mid-Training Stage 2: Synthetic Reasoning Data Training

**Data:** ~500B tokens of synthetic reasoning content for math, science, coding competitions
**Sequence length:** 32K

**Pipeline:**
1. Collect questions and answers from webpages and books
2. Synthesize reasoning processes (chain-of-thought) using a reasoning model
3. Include competition-level problems with verified solutions

---

### 3.5 Mid-Training Stage 3: Long-Context & Agent Training

**Data:** ~100B tokens of long documents (upsampled from pre-training corpus) + large-scale synthetic agent trajectories
**Sequence length:** 128K (extended from 32K)

**Objectives:**
- Long-context performance improvement
- Agent trajectory learning (tool use, multi-step interaction patterns)

---

### 3.6 Summary of Training Stages

| Stage | Data Size | Seq Length | RoPE Base | Packing | Key Focus |
|---|---|---|---|---|---|
| Pre-train 1 | 15T | 4K | 10,000 | Random truncation | General knowledge |
| Pre-train 2 | 7T | 4K | 10,000 | Random truncation | Code + reasoning upsampling |
| Mid-train 1 | 500B | 32K | 1,000,000 | Best-fit | Repo-level code, SWE |
| Mid-train 2 | 500B | 32K | 1,000,000 | Best-fit | Synthetic reasoning |
| Mid-train 3 | 100B | 128K | 1,000,000 | Best-fit | Long context + agents |

Total: ~23T tokens.

---

## 4. Post-Training: Expert Model Iteration

### 4.1 Stage 1: Expert Training

#### 4.1.1 Cold Start SFT

**Objective:** Provide each expert model (Reasoning, Agent, General Chat) with basic capabilities before RL.

**Data:** Small set of SFT data with extended Chain-of-Thought (CoT) responses.

**Loss:**

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{T}_{\text{output}}|} \sum_{t \in \mathcal{T}_{\text{output}}} \log P_\theta(x_t | x_{<t})$$

where $\mathcal{T}_{\text{output}}$ denotes only the output tokens (not prompt tokens).

#### 4.1.2 Expert RL Training

Three separate expert models are trained via domain-specific RL:
1. **Reasoning Expert** (math, code, science)
2. **Agent Expert** (tool use, web search, SWE)
3. **General Chat Expert** (writing, translation, instruction following)

Details of each RL training are given in Sections 4.3–4.5.

---

### 4.2 Stage 2: Unified Training

#### 4.2.1 Overall SFT (Self-Distillation)

**Objective:** Distill capabilities from all expert models into a single hybrid reasoning generalist.

**Data:** Millions of samples covering:
- Reasoning tasks (math, code, science)
- General chat (writing, translation, summarization, chit-chat)
- Agentic tasks (tool use, project development)
- Long-context understanding (up to 128K tokens)

**Key design: Hybrid Reasoning Model**

Data is carefully balanced between:
- Samples with full reasoning (CoT in `<think>...</think>` blocks)
- Samples without explicit thought processes (direct response)

This enables two modes:
- **Thinking mode:** Extended deliberative reasoning for complex tasks
- **Non-thinking mode:** Instant responses for simple queries

**Loss:** Standard autoregressive cross-entropy on distilled expert outputs.

#### 4.2.2 Function Call Template Design

Standard JSON-based function calls force heavy character escaping when parameters contain code. GLM-4.5 introduces an XML-like template that wraps function call keys and values in special token tags:

```
<tool_call>{function-name}
  <arg_key>{key}</arg_key>
  <arg_value>{value}</arg_value>
  ...
</tool_call>
```

**Advantage:** Code segments within parameter values are represented in native form without escaping, reducing the learning burden. Empirically, this does not compromise function call execution accuracy.

#### 4.2.3 Rejection Sampling Pipeline

```
ALGORITHM: RejectionSampling
Input: Expert model π_expert, prompt set P
Output: Filtered SFT dataset D_sft

1. FOR each prompt p ∈ P:
   a. Sample K responses {y_1, ..., y_K} ~ π_expert(·|p)

   b. Stage 1 — Format Filter:
      Remove responses with:
      - Excessive repetition
      - Truncation
      - Invalid reasoning format
      - Excessively short length

   c. Stage 2 — Correctness Verification:
      IF p has objective answer a*:
        Retain only y_i where answer(y_i) = a*

   d. Stage 3 — Reward Model Filter:
      IF p is subjective:
        Score r(p, y_i) via reward model
        Retain y_i with r(p, y_i) > τ_reward

   e. Stage 4 — Tool Call Verification:
      IF p involves tool calling:
        Verify proper invocation protocol
        Verify trajectory reaches expected terminal state
        Discard non-compliant trajectories

2. RETURN D_sft = {(p, y_best)}
```

#### 4.2.4 Prompt Selection and Response-Level Scaling

- **Prompt filtering:** Remove bottom 50% of prompts by response length (easy prompts). Result: 2–4% improvement on math/science with half the data.
- **Response scaling on hard prompts:** Generate 4 responses per hard prompt → additional 1–2% improvement. This is a form of best-of-$N$ selection applied at training data construction time.

#### 4.2.5 Automatic Agentic SFT Data Construction

```
ALGORITHM: AgenticSFTDataConstruction
Input: Frameworks F, tool APIs T, MCP servers M
Output: Agentic SFT dataset D_agent

1. COLLECT agentic frameworks, real-world tool APIs, MCP servers
2. LLM-GENERATE simulated tools for coverage

3. FOR each (framework, tool_set):
   a. IF framework is mature:
      LLM comprehends functionality → generates queries/tasks
   b. ELSE:
      Select representative tool subset → LLM constructs tasks
   c. Tasks include single-step and multi-step tool calling

4. FOR each synthesized task:
   a. Generate tool-call trajectory using existing LLM
   b. FOR multi-step tasks:
      Use LLM as user simulator → multi-round dialogue trajectories

5. FOR each trajectory:
   a. Multiple judge agents evaluate task completion
   b. RETAIN only successful trajectories

6. RETURN D_agent
```

---

### 4.3 Reasoning RL

#### 4.3.1 Base Algorithm: GRPO (without KL)

The reasoning RL objective builds on Group Relative Policy Optimization (GRPO) without the KL divergence term. For problem $x$, sample $K$ responses $\{y_1, \ldots, y_K\}$ from $\pi_{\text{old}}$:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{K} \sum_{i=1}^{K} \min\left( \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) \right]$$

Group-relative advantage:

$$\hat{A}_i = \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r(x) + \delta}$$

where:

$$\bar{r}(x) = \frac{1}{K} \sum_{i=1}^{K} r(x, y_i), \quad \sigma_r(x) = \sqrt{\frac{1}{K} \sum_{i=1}^{K} (r(x, y_i) - \bar{r}(x))^2}$$

**No KL regularization term** is included (departure from standard GRPO/PPO).

#### 4.3.2 Difficulty-Based Curriculum Learning

**Two-stage curriculum:**

**Stage 1 — Moderate difficulty:**
- Problems where the current policy has intermediate pass rates
- $\text{samples\_per\_prompt} = 16$
- Provides gradient signal where reward variance is non-zero

**Stage 2 — Extreme difficulty:**
- Problems with $\text{pass@8} = 0$ but $\text{pass@512} > 0$
- $\text{samples\_per\_prompt} = 512$
- All problems strictly have verified correct answers
- Continues improvement beyond Stage 1 plateau

**Failure mode of static data:** Without curriculum, early-stage batches with all-0 rewards (too hard) or all-1 rewards (too easy) provide zero reward variance → zero gradient signal → training stalls.

Formally, the effective gradient signal is proportional to:

$$\text{Var}[\hat{A}_i] \propto \text{Var}[r(x, y_i)]$$

When $\text{Var}[r] = 0$ (all same rewards), $\hat{A}_i = 0 \; \forall i$, yielding zero policy gradient.

#### 4.3.3 Single-Stage RL at 64K Output Length

**Finding:** Multi-stage RL with progressively increasing output lengths (16K → 32K → 48K → 64K) is **inferior** to single-stage RL directly at 64K.

**Mechanism of failure:** SFT has already conditioned the model on 64K-length responses. Introducing shorter-length RL stages causes the model to "unlearn" long-context generation capabilities. Average output length decreases, and the degradation is **irreversible** — the final 64K stage cannot fully recover.

**Result:** Single-stage at 64K achieves 83.4% on AIME 24 (Avg@32) vs. 80.6% for multi-stage (Figure 6).

#### 4.3.4 Dynamic Sampling Temperature

**Problem:** Fixed temperature $\tau$ fails to adapt as policy entropy decreases during training.

**Algorithm:**

```
ALGORITHM: DynamicTemperature
Input: Current policy π_θ, validation set V, current temperature τ
Output: Updated temperature τ'

1. Monitor average rollout reward r̄ over window W
2. IF r̄ has stabilized (convergence detected):
   a. Evaluate π_θ on V at temperatures {τ, τ+Δ, τ+2Δ, ...}
   b. FOR each candidate temperature τ_c:
      perf(τ_c) ← evaluate(π_θ, V, τ_c)
   c. τ' ← max{τ_c : perf(τ_c) ≥ perf(τ*) - 0.01 · perf(τ*)}
      where τ* = argmax perf(τ_c)
3. ELSE:
   τ' ← τ  (no change)
4. RETURN τ'
```

The temperature is set to the maximum value that does not cause $>1\%$ performance drop from the current optimum, maximizing exploration while bounding quality degradation.

#### 4.3.5 Code RL: Token-Weighted Mean Loss

**Standard sequence-mean loss:**

$$\mathcal{L}_{\text{seq}} = \frac{1}{K} \sum_{i=1}^{K} \hat{A}_i \cdot \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} | y_{i,<t}, x)}{\pi_{\text{old}}(y_{i,t} | y_{i,<t}, x)}$$

**Token-weighted mean loss:**

$$\mathcal{L}_{\text{token}} = \frac{1}{\sum_{i=1}^{K} |y_i|} \sum_{i=1}^{K} \sum_{t=1}^{|y_i|} \hat{A}_i \cdot \log \frac{\pi_\theta(y_{i,t} | y_{i,<t}, x)}{\pi_{\text{old}}(y_{i,t} | y_{i,<t}, x)}$$

**Advantage:** Token-weighted provides finer-grained, more stable gradient signal. Alleviates length bias (long responses aren't artificially down-weighted per-token). Suppresses generation of overly simplistic "base case" responses. Faster convergence (Figure 7, left).

#### 4.3.6 Science RL: Data Quality over Quantity

On GPQA-Diamond, training exclusively on a small set of expert-verified multiple-choice questions outperforms training on mixed-quality science data:

- Expert-verified MCQ only: 65.8% (Figure 7, right)
- Mixed-quality data: 62.9%

**Principle:** For RL, data quality (verified correctness, appropriate difficulty) dominates data quantity.

---

### 4.4 Agentic RL

#### 4.4.1 Data Collection for Agent RL

**Web-search tasks:**
- Multi-hop reasoning over knowledge graphs → demanding QA pairs requiring multi-step reasoning across multiple web sources
- Human-in-the-loop extraction and selective obfuscation of web content for RL signal preparation

**Software engineering tasks:**
- GitHub PRs and issues curated into realistic SWE benchmark
- User prompts + executable unit tests
- Hardened sandbox with distributed system for evaluation (horizontal scalability + isolation)

#### 4.4.2 GRPO for Agent Tasks

Same GRPO objective as reasoning RL. For problem $x$, sample $K$ agent traces $\{y_1, \ldots, y_K\}$ from $\pi_{\text{old}}$:

$$\mathcal{J}(\theta) = \mathbb{E}_{x} \left[ \frac{1}{K} \sum_{i=1}^{K} \min\left( \rho_i \hat{A}_i, \; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) \right]$$

$$\hat{A}_i = \frac{r(x, y_i) - \bar{r}(x)}{\sigma_r(x) + \delta}, \quad \bar{r}(x) = \frac{1}{K} \sum_{i=1}^{K} r(x, y_i)$$

**Critical constraint:** Only model-generated tokens are used for optimization. Environment feedback tokens are excluded from loss computation.

#### 4.4.3 Reward Design

**Web search tasks — Outcome supervision with process format penalty:**

$$r(x, y) = \begin{cases} r_{\text{accuracy}}(x, y) & \text{if all tool calls in } y \text{ have correct format} \\ 0 & \text{otherwise (process halted)} \end{cases}$$

where $r_{\text{accuracy}} \in \{0, 1\}$ based on final answer correctness.

**Coding agent tasks (SWE):**

$$r(x, y) = \begin{cases} 1 & \text{if all unit tests pass} \\ 0 & \text{otherwise} \end{cases}$$

with the same format penalty: incorrect tool format → trace halted → reward 0.

**Key finding:** RL training on web search and SWE tasks produces **generalized** improvements on other tasks (general tool usage, Terminal-Bench), demonstrating transfer of agentic capabilities.

#### 4.4.4 Iterative Self-Distillation

```
ALGORITHM: IterativeSelfDistillation
Input: Cold-start model π_0, agent RL data D
Output: Improved model π_final

1. π_rl^{(1)} ← RL_train(π_0, D)  // Initial RL training

2. FOR iteration i = 1, 2, ...:
   a. // Self-distillation: generate improved SFT data
      D_sft^{(i)} ← GenerateTrajectories(π_rl^{(i)}, D)
      Replace cold-start SFT data with D_sft^{(i)}

   b. // Create improved cold-start model
      π_sft^{(i+1)} ← SFT(π_base, D_sft^{(i)})

   c. // Further RL with increased difficulty
      π_rl^{(i+1)} ← RL_train(π_sft^{(i+1)}, D_harder^{(i+1)})

   d. IF converged: BREAK

3. RETURN π_rl^{(final)}
```

**Rationale:** Agent RL is time-consuming. Self-distillation creates a better starting point for each RL iteration, progressively increasing difficulty. This pushes performance limits more efficiently than extended single-run RL.

#### 4.4.5 Test-Time Compute Scaling via Interaction Turns

For agent tasks, performance scales with interaction turns rather than output tokens:

$$\text{Accuracy}(T_{\text{turns}}) \propto \log(T_{\text{turns}})$$

At BrowseComp (Figure 8), accuracy scales smoothly from ~5% at 8 turns to ~30% at 128 turns (log-scale). The mechanism: more turns enable searching for hard-to-find information and self-verification through writing test cases.

---

### 4.5 General RL

#### 4.5.1 Holistic RL

**Data:** ~5,000 prompts spanning 7 primary, 33 secondary, 139 tertiary categories.

**Multi-source reward:**

$$r_{\text{holistic}}(x, y) = \alpha_{\text{human}} \cdot r_{\text{RM}}(x, y) + \alpha_{\text{AI}} \cdot r_{\text{RLAIF}}(x, y)$$

- **Human feedback:** Reward model $r_{\text{RM}}$ trained on preference annotations across multiple dimensions (instruction following, safety, factual correctness)
- **AI feedback:** Separate scoring rubrics depending on whether prompt has objective ground truth

#### 4.5.2 Instruction Following RL

**Taxonomy:** 7 major, 151 minor constraint types (content requirements, formatting rules, etc.)

**Hybrid feedback system:**
1. Deterministic verification rules (format checking, constraint satisfaction)
2. Trained reward model (semantic quality)
3. Critique model (detailed assessment)

$$r_{\text{IF}}(x, y) = w_{\text{rule}} \cdot r_{\text{rule}}(x, y) + w_{\text{RM}} \cdot r_{\text{RM}}(x, y) + w_{\text{critique}} \cdot r_{\text{critique}}(x, y)$$

**Observation:** Mitigated reward hacking (Figure 9). Up to ~1,000 training steps, instruction following performance (SysBench-ISR) improves monotonically with reward, from 64.8 to 77.2 with no evidence of reward over-optimization.

#### 4.5.3 Function Calling RL

**Step-wise Rule-based RL:**

For tasks with clear tool invocation procedures, ground truth function call $a_t^*$ is annotated for each step $t$.

Reward function:

$$r(a_t, a_t^*) = \begin{cases} 1 & \text{if } \text{format}(a_t) \text{ is valid} \wedge \text{name}(a_t) = \text{name}(a_t^*) \\ & \wedge \text{params}(a_t) = \text{params}(a_t^*) \wedge \forall \text{fields match} \\ 0 & \text{otherwise} \end{cases}$$

This is a **strict exact-match reward** — all fields (name, parameters, every value) must match exactly.

**End-to-End Multi-turn RL:**

For complex tasks requiring dynamic interaction:

**Single-turn multi-step tasks:**
- Model makes multiple function calls interacting with environment
- Uses complex tasks from MCP servers and AgentGym

**Multi-turn multi-step tasks:**
- Model interacts with both tool execution environment and LLM-simulated user agent

Reward:

$$r(I, \text{trajectory}) = \begin{cases} 1 & \text{if } \text{TaskCompleted}(I, o_0, a_1, o_1, \ldots, a_T, o_T) = \text{True} \\ 0 & \text{otherwise} \end{cases}$$

where $I$ is the original task specification, $a_t$ is the $t$-th function call, $o_t$ is the tool feedback or user information, and task completion is determined by environment rules or LLM Judge Agent.

#### 4.5.4 Pathology RL

**Objective:** Correct rare but problematic behaviors:
- Language mixing
- Excessive repetition
- Formatting mistakes

**Challenge:** Incidence rate < 1% of outputs → sample-inefficient to penalize during normal RL.

**Solution:** Curate targeted dataset of prompts highly likely to trigger pathological behaviors. Targeted penalty training efficiently reduces residual error rates.

---

## 5. RL Infrastructure: Slime

### 5.1 Architecture Overview

Three core modules (Figure 10):

1. **Training (Megatron):** Reads data from Data Buffer, performs gradient updates, synchronizes parameters with Rollout module
2. **Rollout (SGLang + Router):** Generates new data including rewards and verifier outputs, writes to Data Buffer
3. **Data Buffer:** Bridge module managing prompt initialization, custom data, rollout generation strategies

### 5.2 Flexible Hybrid Training Modes

**Synchronous Colocated Mode (Reasoning RL):**
- Training and inference engines on same worker
- Dynamic sampling minimizes GPU idle time
- Suitable for math, code, general RL where rollouts are fast

**Disaggregated Asynchronous Mode (Agent RL):**
- Rollout component exposed directly to agent environment
- Training and inference GPUs scheduled independently
- Agent environments operate continuously without stalling for training cycles
- Ray framework handles resource scheduling

### 5.3 Accelerated Rollout with Mixed-Precision

- Training: BF16
- Inference/Rollout: FP8

**Online block-wise FP8 quantization:**

During each policy update iteration:
1. Parameters $\boldsymbol{\theta}$ in BF16
2. Before rollout dispatch: $\boldsymbol{\theta}_{\text{FP8}} = \text{BlockwiseQuantize}(\boldsymbol{\theta}, \text{block\_size})$
3. Rollout uses $\boldsymbol{\theta}_{\text{FP8}}$ for generation

This significantly improves rollout throughput (the primary bottleneck in RL training).

### 5.4 Agent-Oriented RL Infrastructure

**High-concurrency Docker-based runtime:**
- Isolated container per task
- Reduces rollout overhead for complex environments

**Fully asynchronous RL loop:**
- GPUs partitioned into dedicated rollout engines and training engines
- Rollout engines continuously generate trajectories
- Training engines update weights and periodically synchronize to rollout engines
- Prevents long/diverse trajectories from blocking training

**Unified HTTP endpoint + centralized data pool:**
- All agent frameworks produce rollouts in message-list format
- Stored in shared data pool
- Supports task-specific filtering and dynamic sampling
- Decouples rollout logic from RL training

### 5.5 Pseudo-Algorithm: RL Training Loop

```
ALGORITHM: SlimeRLTrainingLoop
Input: Policy π_θ, prompt dataset D, environment E, mode ∈ {sync, async}
Output: Trained policy π_θ*

1. Initialize Data Buffer DB with prompts from D
2. Initialize Rollout Engine R with π_θ (FP8 quantized)
3. Initialize Training Engine T with π_θ (BF16)

4. WHILE not converged:

   IF mode = sync:
     a. Sample batch B of prompts from DB
     b. FOR each prompt x ∈ B:
        Generate K responses {y_1,...,y_K} ~ R.generate(x, E)
        Compute rewards r(x, y_i) for each i
        Compute advantages Â_i = (r(x,y_i) - r̄(x)) / (σ_r(x) + δ)
     c. Write (x, {y_i, Â_i}) to DB
     d. T.update(θ) using GRPO objective on DB batch
     e. Synchronize θ: R.load_params(BlockwiseQuantize(θ))

   IF mode = async:
     a. // Rollout engines run continuously
        PARALLEL FOR each rollout worker:
          Sample prompt x from DB
          Generate trajectory y ~ R.generate(x, E)
          Compute reward r(x, y)
          Write (x, y, r) to DB

     b. // Training engine consumes from DB
        WHILE DB has sufficient new data:
          Sample batch from DB
          Compute advantages
          T.update(θ) using GRPO objective
          IF sync_interval reached:
            R.load_params(BlockwiseQuantize(θ))

5. RETURN π_θ*
```

---

## 6. Inference Path

### 6.1 Hybrid Reasoning Modes

**Thinking Mode:**
- For complex reasoning and agentic tasks
- Generates extended CoT within `<think>...</think>` blocks
- Maximum output length: 64K tokens

**Non-thinking Mode:**
- For instant responses (chit-chat, simple queries)
- No `<think>` block, direct answer generation

Mode selection is learned during Overall SFT through data balancing between CoT and non-CoT samples.

### 6.2 Speculative Decoding via MTP Layer

The MTP layer generates $M$ draft tokens that are verified by the main model:

```
ALGORITHM: SpeculativeDecoding
Input: Model M (main + MTP), prompt x, max_tokens N
Output: Generated sequence y

1. y ← []
2. WHILE len(y) < N:
   a. // Draft phase: MTP layer generates M candidates
      h ← M.encode(x + y)  // forward through main model
      drafts ← MTP_layer.predict(h, M)  // M draft tokens

   b. // Verify phase: main model verifies all M drafts in parallel
      logits ← M.forward(x + y + drafts)
      FOR m = 1 to M:
        IF sample_matches(logits[m], drafts[m]):
          Accept draft token m
        ELSE:
          Resample from logits[m]
          BREAK  // reject remaining drafts

   c. Append accepted/resampled tokens to y

3. RETURN y
```

### 6.3 KV Cache Configuration

Per layer per token KV cache:

$$\text{KV entries} = 2 \times n_{\text{kv}} \times d_h = 2 \times 8 \times 128 = 2048$$

Total KV cache for sequence length $L$ across all 93 layers in BF16:

$$\text{KV cache (bytes)} = 93 \times L \times 2048 \times 2$$

| Sequence Length | KV Cache Size |
|---|---|
| 4K | ~1.5 GB |
| 32K | ~11.7 GB |
| 128K | ~46.9 GB |

### 6.4 Serving Configuration (SWE-bench)

- Framework: OpenHands v0.34.0
- Max iterations: 100
- History truncation at 128K context limit
- Temperature: 0.6
- top_p: 1.0

### 6.5 Test-Time Compute Strategies

**For reasoning:**
- Average over multiple samples (Avg@32 for AIME, Avg@8 for GPQA)
- Best-of-N selection

**For agents:**
- Scale interaction turns (8 → 128 for BrowseComp)
- Self-verification through test case writing (coding agents)

---

## 7. Evaluation Protocol

### 7.1 Benchmark Suite and Metrics

#### 7.1.1 Agentic Benchmarks

| Benchmark | Metric | GLM-4.5 Score |
|---|---|---|
| TAU-Bench Retail | Success Rate (%) | 79.7 |
| TAU-Bench Airline | Success Rate (%) | 60.4 |
| BFCL V3 | Overall Score (%) | 77.8 |
| BrowseComp | Accuracy (%) | 26.4 |

**TAU-Bench:** Uses optimized user simulator (Figure 11) with detailed prompt engineering for natural multi-turn interaction.

**BFCL V3:** Measures ability to call user-defined functions.

**BrowseComp:** Web browsing agent for complicated information retrieval.

#### 7.1.2 Reasoning Benchmarks

| Benchmark | Metric | GLM-4.5 Score |
|---|---|---|
| MMLU-Pro | EM (%) | 84.6 |
| AIME 24 | Avg@32 (%) | 91.0 |
| MATH-500 | EM (%) | 98.2 |
| SciCode | Pass@1 (%) | 41.7 |
| GPQA | Avg@8 (%) | 79.1 |
| HLE | Accuracy (%) | 14.4 |
| LCB (2407-2501) | Accuracy (%) | 72.9 |

**AA-Index (Artificial Analysis Intelligence Index):** Composite metric across 7 reasoning benchmarks. GLM-4.5: 67.7 (estimated).

#### 7.1.3 Coding Benchmarks

| Benchmark | Metric | GLM-4.5 Score |
|---|---|---|
| SWE-bench Verified | Resolve Rate (%) | 64.2 |
| Terminal-Bench | Success Rate (%) | 37.5 |

**SWE-bench Verified:** 500 human-filtered GitHub issue resolution instances, evaluated via OpenHands v0.34.0.

**Terminal-Bench:** Complex terminal tasks using Terminus framework with standard function calling.

#### 7.1.4 General Benchmarks

| Benchmark | Metric | GLM-4.5 Score |
|---|---|---|
| MMLU | EM (%) | 90.0 |
| SimpleQA | Correct (%) | 26.4 |
| IFEval | Prompt Strict (%) | 86.1 |
| SysBench | ISR (%) | 81.0 |
| MultiChallenge | Score (%) | 52.8 |

#### 7.1.5 Safety Benchmark

| Category | GLM-4.5 Score |
|---|---|
| Overall | 89.87 |
| Ethics & Morality | 94.33 |
| Illegal Activities | 90.97 |
| Mental Health | 94.67 |
| Offensiveness | 83.00 |
| Physical Health | 96.67 |
| Privacy & Property | 92.00 |
| Unfairness & Bias | 77.40 |

SafetyBench: 11,435 multiple-choice questions across 7 safety categories.

### 7.2 Evaluation Methodology

**Automated evaluation:**
- LLM-based answer validation for AIME and GPQA
- GPT-4o judging for HLE text-based questions
- Deterministic unit test execution for SWE-bench and Terminal-Bench
- Open-sourced evaluation toolkit at `glm-simple-evals`

**Variance mitigation:**
- Multiple sampling with averaging: Avg@32 (AIME), Avg@8 (GPQA)
- Controlled evaluation parameters (temperature, top_p specified)

**Human evaluation:**
- 660 curated prompts (392 English, 108 Chinese, 160 other languages)
- 7 categories: Math, Text Processing, Text Generation, Subjective QA, Objective QA, Logical Reasoning, Code
- Single evaluator per batch for consistency
- 0–10 scoring scale
- Reasoning content hidden from evaluators (for GLM-4.5 and DeepSeek-R1-0528)
- Randomized response ordering

**CC-Bench (Coding Agent):**
- 52 tasks on Claude Code framework
- Isolated containerized environments
- Human expert interactive evaluation
- Primary metric: task completion
- Secondary metrics: tool calling success rate, token consumption efficiency

### 7.3 Pseudo-Algorithm: Evaluation Pipeline

```
ALGORITHM: EvaluationPipeline
Input: Model M, benchmark set B
Output: Scores per benchmark

1. FOR each benchmark b ∈ B:
   a. Load evaluation dataset D_b
   b. Configure evaluation parameters:
      - temperature, top_p, max_tokens per benchmark spec
      - num_samples (32 for AIME, 8 for GPQA, 1 otherwise)

   c. FOR each problem p ∈ D_b:
      i. Generate N responses: {y_1,...,y_N} ~ M(·|p)
      ii. IF b has objective ground truth:
          score_p ← (1/N) Σ_i 1[answer(y_i) = gt(p)]
      iii. ELSE IF b has test cases:
          score_p ← (1/N) Σ_i execute_tests(y_i, tests(p))
      iv. ELSE IF b is subjective:
          score_p ← human_eval(y_i) or LLM_judge(y_i)

   d. Score(b) ← mean({score_p}_{p ∈ D_b})

2. Compute aggregate metrics:
   - ARC average across 12 benchmarks
   - AA-Index for reasoning
   - Per-domain averages

3. RETURN all scores
```

---

## 8. Compression and Information Preservation

### 8.1 MoE as Structural Compression

The MoE architecture achieves effective compression by activating only a fraction of total parameters per token:

**Compression ratio:**

$$\rho = \frac{\text{Activated parameters}}{\text{Total parameters}} = \frac{32\text{B}}{355\text{B}} \approx 0.090$$

This means ~9% of parameters are activated per token, yielding compute equivalent to a 32B dense model while storing 355B parameters' worth of knowledge.

**Information capacity vs. compute trade-off:**

$$\text{FLOPs per token} \propto P_{\text{active}} = 32\text{B}$$

$$\text{Knowledge capacity} \propto P_{\text{total}} = 355\text{B}$$

GLM-4.5 achieves comparable or superior performance to DeepSeek-R1 (671B total, 37B active) and Kimi K2 (1043B total, 32B active) with significantly fewer total parameters, lying on the Pareto frontier of SWE-bench performance vs. model parameters (Figure 2).

### 8.2 FP8 Quantization for Inference

**Block-wise dynamic FP8 quantization:**

For parameter tensor $\mathbf{W} \in \mathbb{R}^{m \times n}$ in BF16, partitioned into blocks of size $B$:

$$s_b = \frac{\max_{i \in \text{block}_b} |W_i|}{127}, \quad \hat{W}_i = \text{round}\left(\frac{W_i}{s_b}\right)$$

$$\tilde{W}_i = \hat{W}_i \times s_b$$

**Quantization error bound:**

$$\|W - \tilde{W}\|_\infty \leq \frac{s_b}{2} = \frac{\max_{i \in \text{block}_b} |W_i|}{254}$$

**Compression ratio (BF16 → FP8):**

$$\text{Memory ratio} = \frac{1 \text{ byte (FP8)} + \text{scale overhead}}{2 \text{ bytes (BF16)}} \approx 0.5$$

For 355B parameters: ~710 GB BF16 → ~355 GB FP8 (plus scale factors).

### 8.3 Information Preservation Guarantees

**MoE routing preserves information** because:
1. All experts store distinct knowledge, accessed on-demand via learned routing
2. Shared expert provides baseline capacity for all tokens
3. Sigmoid gating (not softmax) allows flexible expert weighting without forced normalization

**FP8 quantization impact:** Used only for rollout inference, not training. Training remains in BF16, ensuring no information loss in the learning process. FP8 rollout introduces bounded noise but empirically does not degrade RL training quality.

---

## 9. Convergence Dynamics

### 9.1 Pre-Training Convergence

**Loss trajectory:** Cosine LR decay ensures continuous loss reduction throughout training. No stable phase (unlike WSD) prevents underfitting.

**Muon optimizer convergence properties:**
- Newton-Schulz preconditioning provides adaptive step sizes across parameter dimensions
- $N=5$ iterations sufficient for practical convergence of the preconditioner
- Momentum $\mu = 0.95$ smooths gradient noise at large batch sizes
- Update RMS scaling to 0.2 provides implicit learning rate normalization

### 9.2 RL Convergence Dynamics

**GRPO without KL:**
- Removing KL regularization allows larger policy updates
- Risk: policy can deviate significantly from initial distribution
- Mitigation: clipping ratio $\epsilon$ bounds per-step policy change

**Curriculum learning convergence:**
- Stage 1 (moderate difficulty): rapid initial improvement, plateau around 81.8% (AIME 24)
- Stage 2 (extreme difficulty): breaks through plateau to 83.4%
- Without stage transition: performance saturates

**Dynamic temperature convergence:**
- Temperature increases during convergence phases maintain exploration
- 1% quality bound prevents catastrophic temperature increases
- Enables continued improvement past fixed-temperature plateaus

### 9.3 Instruction Following RL Convergence

From Figure 9:
- Reward increases monotonically from ~0.5 to ~2.0 over 1,000 steps
- SysBench-ISR improves from 64.8 to 77.2
- No evidence of reward hacking (reward and true performance remain correlated)
- Hybrid feedback system (rules + RM + critique) provides robust gradient signal

---

## 10. Failure Modes

### 10.1 Pre-Training Failure Modes

| Failure Mode | Detection | Mitigation |
|---|---|---|
| Template page contamination | SemDedup cluster analysis | Embedding-based deduplication |
| Quality classifier false positives | Manual audit of high-score bucket | SemDedup secondary filter |
| Domain imbalance | Benchmark tracking during training | Quality-based upsampling schedule |
| Underfitting (WSD schedule) | Worse SimpleQA/MMLU scores | Switch to cosine decay |

### 10.2 RL Failure Modes

| Failure Mode | Detection | Mitigation |
|---|---|---|
| Zero reward variance | All-0 or all-1 rewards in batch | Difficulty-based curriculum |
| Long-context capability loss | Output length decrease in multi-stage RL | Single-stage 64K RL |
| Insufficient exploration | Reward plateau | Dynamic sampling temperature |
| Length bias (code RL) | Simple/repetitive outputs | Token-weighted mean loss |
| Low-quality science data | Poor GPQA performance | Expert-verified data only |
| Reward hacking | Reward ↑ but performance ↓ | Hybrid feedback (rules + RM + critique) |
| Agent format errors | Halted trajectories, zero rewards | Process format penalty |
| Pathological behaviors | Language mixing, repetition (<1%) | Targeted pathology RL |

### 10.3 Inference Failure Modes

| Failure Mode | Detection | Mitigation |
|---|---|---|
| Unnecessary thinking for simple queries | Excessive latency on easy tasks | Hybrid reasoning mode selection |
| Character escaping overhead | Increased token count in tool calls | XML-like function call template |
| KV cache overflow at 128K | OOM errors | History truncation, GQA compression |
| Speculative decode rejection | Low acceptance rate | MTP layer quality monitoring |

---

## 11. Deployment Constraints

### 11.1 Memory Requirements

**Model weights (BF16):**

$$355\text{B} \times 2 \text{ bytes} = 710 \text{ GB}$$

**Model weights (FP8):**

$$355\text{B} \times 1 \text{ byte} \approx 355 \text{ GB}$$

**KV cache at 128K (BF16):**

$$\approx 47 \text{ GB}$$

**Minimum serving configuration (FP8):** ~402 GB model + KV cache, requiring multi-GPU deployment.

### 11.2 Activated Compute per Token

With 32B activated parameters, per-token FLOPs are comparable to a 32B dense model:

$$\text{FLOPs per token} \approx 6 \times 32\text{B} = 192 \text{ GFLOPs (forward pass estimate)}$$

### 11.3 Expert Parallelism

With $E = 160$ total experts and $k = 8$ active per token, expert parallelism distributes experts across devices:

$$\text{Experts per device} = \frac{E}{N_{\text{devices}}}$$

All-to-all communication required for routing tokens to their selected experts across devices.

### 11.4 Context Window

Maximum supported: 128K tokens (after mid-training context extension).

### 11.5 Batch Serving

For SWE-bench evaluation: sequential (100 iterations per task). For throughput optimization: continuous batching with paged KV cache management.

### 11.6 Evaluation Reproducibility

Open-sourced evaluation toolkit: `glm-simple-evals` with fixed random seeds, specified temperature/top_p, and standardized prompts.

---

## 12. Key Quantitative Results Summary

### 12.1 Overall Rankings (12 ARC Benchmarks, July 28, 2025)

| Model | Overall Rank | Agentic Rank | Coding Rank |
|---|---|---|---|
| GLM-4.5 (355B/32B) | **3rd** | **2nd** | **3rd** |
| GLM-4.5-Air (106B/12B) | **6th** | — | — |

### 12.2 Parameter Efficiency

| Model | Total Params | Activated Params | SWE-bench Verified |
|---|---|---|---|
| GLM-4.5 | 355B | 32B | 64.2% |
| DeepSeek-R1 | 671B | 37B | 41.4% |
| Kimi K2 | 1043B | 32B | 65.4% |
| Claude Sonnet 4 | Unknown | Unknown | 70.4% |

GLM-4.5 has half the parameters of DeepSeek-R1 and one-third those of Kimi K2, while achieving superior or competitive performance on SWE-bench. GLM-4.5 and GLM-4.5-Air lie on the **Pareto frontier** of SWE-bench performance vs. model parameters (Figure 2).

### 12.3 Ablation Impact Summary

| Technique | Metric | Improvement |
|---|---|---|
| Difficulty curriculum Stage 2 | AIME 24 Avg@32 | 81.8% → 83.4% (+1.6%) |
| Single-stage 64K vs. multi-stage | AIME 24 Avg@32 | 80.6% → 83.4% (+2.8%) |
| Token-weighted vs. sequence-mean loss | LCB accuracy convergence | Faster convergence, same final (~46.5%) |
| Expert-verified vs. mixed science data | GPQA-Diamond | 62.9% → 65.8% (+2.9%) |
| Prompt filtering (remove easy 50%) | Math/Science | +2–4% with half the data |
| Response scaling (4× on hard prompts) | Math/Science | Additional +1–2% |
| BrowseComp interaction scaling | BrowseComp | ~5% (8 turns) → ~30% (128 turns) |