

# GPT-OSS-120B & GPT-OSS-20B: End-to-End Technical Report

---

## 1. Data Pipeline

### 1.1 Objective

Curate a text-only pretraining corpus with trillions of tokens, biased toward STEM, coding, and general knowledge, with proactive removal of hazardous biosecurity content. Knowledge cutoff: June 2024.

### 1.2 Data Curation and Filtering

**Inputs:** Raw web crawl data, curated text corpora, code repositories, scientific literature, general knowledge sources.

**Outputs:** Deduplicated, filtered, tokenized training shards with CBRN-harmful content removed.

#### 1.2.1 CBRN Pre-Training Filters

- Reuses the Chemical, Biological, Radiological, Nuclear (CBRN) pre-training filters from GPT-4o.
- Filters target hazardous biosecurity knowledge specifically, removing content that could provide actionable information across five biothreat creation stages: Ideation, Acquisition, Magnification, Formulation, Release.
- Filter operates as a classifier-based pipeline: documents flagged by a trained CBRN classifier with $P(\text{harmful}) > \tau_{\text{CBRN}}$ are excluded from the training set.

#### 1.2.2 Domain Weighting

- Explicit upweighting of STEM, coding, and general knowledge domains.
- Let $w_d$ denote the sampling weight for domain $d$. The effective probability of sampling a document from domain $d$ is:

$$
P(d) = \frac{w_d \cdot |D_d|}{\sum_{d'} w_{d'} \cdot |D_{d'}|}
$$

where $|D_d|$ is the number of documents in domain $d$.

#### 1.2.3 Deduplication

- Standard fuzzy and exact deduplication (MinHash-based near-duplicate detection, exact substring deduplication).
- Removes training-evaluation contamination by filtering documents with high $n$-gram overlap against evaluation benchmarks.

#### 1.2.4 Invariants

- No document in the training set should contain actionable CBRN-harmful content above the filter threshold.
- Domain distribution must match the target mixing ratio within statistical tolerance.
- Evaluation benchmark contamination rate must be below a predetermined threshold.

#### 1.2.5 Failure Modes

- **False negatives in CBRN filtering:** Harmful content passes the classifier. Mitigated by high-recall classifier design and manual auditing.
- **Domain imbalance drift:** Sampling weights misaligned with actual shard composition. Mitigated by periodic validation of domain statistics.
- **Contamination leakage:** Evaluation data present in training. Mitigated by $n$-gram overlap filtering and held-out set management.

### 1.3 Tokenizer: o200k_harmony (BPE)

**Definition:** Byte Pair Encoding tokenizer extending the o200k vocabulary with special tokens for the harmony chat format. Total vocabulary size: $|V| = 201{,}088$ tokens.

**Key properties:**
- BPE algorithm iteratively merges the most frequent byte pairs until the target vocabulary size is reached.
- Special tokens include message boundary delimiters, role indicators (System, Developer, User, Assistant, Tool), and channel markers (analysis, commentary, final).
- Shared across all training stages (pretraining, post-training, inference).

**Tokenization algorithm (pseudo):**

```
ALGORITHM: BPE_Tokenize(text, merges, special_tokens)
  1. Identify and extract special_tokens from text as atomic units
  2. Split remaining text into UTF-8 byte sequences
  3. FOR each merge_rule in merges (ordered by priority):
       Replace all adjacent pairs matching merge_rule with merged token
  4. RETURN sequence of token IDs
```

**Invariant:** Tokenization is deterministic and invertible (lossless round-trip from text to token IDs and back).

---

## 2. Model Architecture

### 2.1 Overview

Autoregressive Mixture-of-Experts (MoE) Transformer, building on GPT-2/GPT-3 architectural lineage.

| Specification | gpt-oss-120b | gpt-oss-20b |
|---|---|---|
| Layers | 36 | 24 |
| Total Parameters | 116.83B | 20.91B |
| Active Parameters/Token | 5.13B | 3.61B |
| Residual Stream Dimension $d_{\text{model}}$ | 2880 | 2880 |
| Number of Experts $E$ | 128 | 32 |
| Top-$k$ Experts Selected | 4 | 4 |
| Query Heads $n_q$ | 64 | 64 |
| Head Dimension $d_h$ | 64 | 64 |
| KV Heads (GQA) $n_{kv}$ | 8 | 8 |
| Attention Window (banded layers) | 128 tokens | 128 tokens |
| Max Context Length (dense layers) | 131,072 tokens | 131,072 tokens |
| Checkpoint Size (MXFP4) | 60.8 GiB | 12.8 GiB |

### 2.2 Residual Stream and Normalization

**Pre-LN Transformer (GPT-2 style placement):** Normalization is applied before each sub-layer (attention, MoE), not after.

For layer $\ell$, the residual stream update:

$$
\mathbf{h}^{(\ell)}_{\text{attn}} = \mathbf{h}^{(\ell-1)} + \text{Attn}\!\left(\text{RMSNorm}\!\left(\mathbf{h}^{(\ell-1)}\right)\right)
$$

$$
\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell)}_{\text{attn}} + \text{MoE}\!\left(\text{RMSNorm}\!\left(\mathbf{h}^{(\ell)}_{\text{attn}}\right)\right)
$$

**RMSNorm definition:**

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}
$$

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learned scale parameter, $\epsilon$ is a small constant for numerical stability, and $d = d_{\text{model}} = 2880$.

**Rationale for Pre-LN:** Stabilizes gradient flow in deep networks; avoids the gradient explosion/vanishing issues of Post-LN at the cost of slightly reduced representational capacity per layer (empirically negligible at this scale).

### 2.3 Attention Mechanism

#### 2.3.1 Grouped Query Attention (GQA)

- $n_q = 64$ query heads, $n_{kv} = 8$ key-value heads.
- GQA group ratio: $g = n_q / n_{kv} = 8$ query heads share each KV head.
- Reduces KV cache memory by factor $g$ relative to MHA while preserving near-MHA quality.

For input $\mathbf{X} \in \mathbb{R}^{T \times d_{\text{model}}}$:

$$
\mathbf{Q} = \mathbf{X} \mathbf{W}^Q, \quad \mathbf{Q} \in \mathbb{R}^{T \times n_q \times d_h}
$$

$$
\mathbf{K} = \mathbf{X} \mathbf{W}^K, \quad \mathbf{K} \in \mathbb{R}^{T \times n_{kv} \times d_h}
$$

$$
\mathbf{V} = \mathbf{X} \mathbf{W}^V, \quad \mathbf{V} \in \mathbb{R}^{T \times n_{kv} \times d_h}
$$

Each query head $q_i$ in group $j = \lfloor i / g \rfloor$ attends using $\mathbf{K}_j, \mathbf{V}_j$:

$$
\text{Attn}(\mathbf{Q}_i, \mathbf{K}_j, \mathbf{V}_j) = \text{softmax}\!\left(\frac{\mathbf{Q}_i \mathbf{K}_j^T}{\sqrt{d_h}} + \mathbf{M} + b_i\right) \mathbf{V}_j
$$

where $\mathbf{M}$ is the causal (and optionally banded-window) mask, and $b_i$ is a **learned scalar bias in the softmax denominator** (see §2.3.3).

#### 2.3.2 Alternating Banded-Window and Dense Attention

Layers alternate between two attention patterns:

- **Banded-window layers:** Each token attends only to the nearest $w = 128$ tokens in the past. The mask $\mathbf{M}_{s,t} = 0$ if $0 \leq s - t \leq w$, else $-\infty$.
- **Dense (full) layers:** Standard causal mask with context length extended to $T_{\max} = 131{,}072$ tokens using YaRN.

**Complexity per layer:**

| Pattern | Attention FLOPs | KV Cache Memory |
|---|---|---|
| Banded-window | $O(T \cdot w \cdot d_h \cdot n_q)$ | $O(w \cdot n_{kv} \cdot d_h)$ per token |
| Dense | $O(T^2 \cdot d_h \cdot n_q)$ | $O(T \cdot n_{kv} \cdot d_h)$ per token |

Alternating pattern significantly reduces average-case compute and memory relative to all-dense attention while preserving global information flow through the dense layers.

#### 2.3.3 Learned Softmax Denominator Bias (Attention Sink Mechanism)

Each attention head $i$ has a **learned scalar bias** $b_i$ added to the logits before softmax:

$$
\alpha_{s,t}^{(i)} = \frac{\exp\!\left(\frac{\mathbf{q}_s^{(i)T} \mathbf{k}_t^{(i)}}{\sqrt{d_h}} + M_{s,t}\right)}{\sum_{t'} \exp\!\left(\frac{\mathbf{q}_s^{(i)T} \mathbf{k}_{t'}^{(i)}}{\sqrt{d_h}} + M_{s,t'}\right) + \exp(b_i)}
$$

This is equivalent to adding a virtual "sink" token with logit $b_i$. When $b_i$ is large, the attention distribution can effectively attend to *no* real token, redistributing mass to the implicit sink. This addresses the "attention sink" phenomenon and the off-by-one softmax issue, allowing the model to represent "pay no attention" without distorting actual attention weights.

**Invariant:** $\sum_{t} \alpha_{s,t}^{(i)} + \frac{\exp(b_i)}{Z} = 1$ where $Z$ is the full partition function including $\exp(b_i)$.

#### 2.3.4 Rotary Position Embeddings (RoPE) with YaRN Extension

**RoPE** encodes absolute position through rotation of query-key pairs:

$$
\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m \boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(m \boldsymbol{\theta})
$$

where $m$ is the token position and $\boldsymbol{\theta}_i = \theta_{\text{base}}^{-2i/d_h}$ with $\theta_{\text{base}} = 10{,}000$ (standard).

**YaRN** (Yet another RoPE extensioN) extends context beyond training length by:

1. **NTK-aware frequency scaling:** Modifies $\theta_{\text{base}}$ to $\theta_{\text{base}}' = \theta_{\text{base}} \cdot s^{d_h/(d_h - 2)}$ where $s$ is the scaling factor.
2. **Attention temperature correction:** Scales attention logits by $\sqrt{1/t}$ where $t$ accounts for the entropy increase from longer sequences.
3. **Per-dimension interpolation:** Different frequency bands are scaled differently—low frequencies are interpolated (NTK-by-parts), high frequencies are left unchanged.

This enables extending dense-layer context to $T_{\max} = 131{,}072$ tokens without retraining from scratch.

### 2.4 Mixture-of-Experts (MoE) Block

#### 2.4.1 Router

The router is a single linear projection from residual activations to expert scores:

$$
\mathbf{s} = \mathbf{W}_r \cdot \text{RMSNorm}(\mathbf{h}) \in \mathbb{R}^{E}
$$

where $\mathbf{W}_r \in \mathbb{R}^{E \times d_{\text{model}}}$ and $E$ is the number of experts (128 for 120b, 32 for 20b).

**Top-$k$ selection ($k = 4$):**

$$
\mathcal{S} = \text{Top-}k(\mathbf{s}), \quad |\mathcal{S}| = 4
$$

**Expert weighting via restricted softmax over selected experts:**

$$
g_e = \frac{\exp(s_e)}{\sum_{e' \in \mathcal{S}} \exp(s_{e'})}, \quad e \in \mathcal{S}
$$

$$
\text{MoE}(\mathbf{h}) = \sum_{e \in \mathcal{S}} g_e \cdot \text{Expert}_e(\mathbf{h})
$$

#### 2.4.2 Expert Architecture: Gated SwiGLU (Modified)

Each expert is a gated feed-forward network with SwiGLU activation, with two unconventional modifications: **clamping** and a **residual connection**.

**Standard SwiGLU:**

$$
\text{SwiGLU}(\mathbf{x}) = \left(\text{SiLU}(\mathbf{x} \mathbf{W}_1) \odot (\mathbf{x} \mathbf{W}_3)\right) \mathbf{W}_2
$$

where $\text{SiLU}(z) = z \cdot \sigma(z)$, $\sigma$ is the sigmoid function, $\mathbf{W}_1, \mathbf{W}_3 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$.

**gpt-oss modification (unconventional):**

$$
\text{Expert}(\mathbf{x}) = \text{clamp}\!\left(\text{SwiGLU}(\mathbf{x}), -C, C\right) + \alpha \cdot \mathbf{x}
$$

where clamping bounds the intermediate activations to $[-C, C]$ for numerical stability under quantized weights, and the residual connection $\alpha \cdot \mathbf{x}$ stabilizes expert outputs (exact $\alpha$ and $C$ values are architecture hyperparameters).

#### 2.4.3 Parameter Accounting

For gpt-oss-120b:
- MLP (MoE) parameters: 114.71B (98.2% of total)
- Attention parameters: 0.96B
- Embed + Unembed: 1.16B (unembedding counted toward active)
- Active per token: 4 experts × (expert params) + attention + unembed ≈ 5.13B

For gpt-oss-20b:
- MLP parameters: 19.12B
- Active per token: 3.61B

#### 2.4.4 Computational Complexity

Per-token forward pass FLOPs (dominant terms):

$$
\text{FLOPs}_{\text{active}} \approx 2 \times P_{\text{active}} = 2 \times 5.13 \times 10^9 \approx 10.26 \text{ GFLOPs (120b)}
$$

This is the key efficiency advantage of MoE: total parameter count is 116.8B but per-token compute scales with active parameters (5.13B), yielding ~23× parameter-to-compute ratio.

#### 2.4.5 Failure Modes

- **Expert load imbalance:** Some experts receive disproportionate traffic, causing underutilization. Standard mitigation: auxiliary load-balancing loss $\mathcal{L}_{\text{balance}}$ (not explicitly mentioned in the model card but standard practice).
- **Router collapse:** Router converges to always selecting the same subset of experts. Same mitigation as above.
- **Expert interference under quantization:** MXFP4 quantization of expert weights could degrade individual expert quality. Mitigated by quantization-aware training (see §3).

### 2.5 Embedding and Unembedding

- **Embedding:** $\mathbf{W}_{\text{emb}} \in \mathbb{R}^{|V| \times d_{\text{model}}}$, maps token IDs to $d_{\text{model}}$-dimensional vectors.
- **Unembedding:** $\mathbf{W}_{\text{unemb}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$, projects final hidden states to logits over vocabulary.
- Embedding parameters are **not** counted toward active parameters; unembedding parameters **are** counted toward active parameters.
- Combined: 1.16B parameters = $2 \times |V| \times d_{\text{model}} = 2 \times 201{,}088 \times 2{,}880 \approx 1.16 \times 10^9$.

### 2.6 Complete Forward Pass (Pseudo-Algorithm)

```
ALGORITHM: GPT-OSS Forward Pass
INPUT: token_ids ∈ ℤ^T, T ≤ 131,072
OUTPUT: logits ∈ ℝ^{T × |V|}

1. h ← W_emb[token_ids]                          // h ∈ ℝ^{T × d_model}
2. FOR ℓ = 1 TO L:                                // L = 36 (120b) or 24 (20b)
   a. h_norm ← RMSNorm(h)
   b. IF ℓ is odd:                                // Banded-window attention
        mask ← BandedCausalMask(T, w=128)
      ELSE:                                       // Dense attention
        mask ← CausalMask(T)
   c. Q ← h_norm · W_Q^(ℓ)                       // ℝ^{T × n_q × d_h}
      K ← h_norm · W_K^(ℓ)                       // ℝ^{T × n_kv × d_h}
      V ← h_norm · W_V^(ℓ)                       // ℝ^{T × n_kv × d_h}
   d. Apply RoPE to Q, K (with YaRN scaling for dense layers)
   e. Compute GQA attention with learned softmax bias b^(ℓ):
        attn_out ← FlashAttention(Q, K, V, mask, bias=b^(ℓ))
   f. h ← h + W_O^(ℓ) · attn_out                // Residual connection
   g. h_norm ← RMSNorm(h)
   h. s ← W_r^(ℓ) · h_norm                      // Router scores ∈ ℝ^{T × E}
   i. FOR each token t:
        S_t ← TopK(s_t, k=4)
        g_t ← RestrictedSoftmax(s_t, S_t)
        moe_out_t ← Σ_{e ∈ S_t} g_{t,e} · Expert_e(h_norm_t)
   j. h ← h + moe_out                            // Residual connection
3. h_final ← RMSNorm(h)
4. logits ← h_final · W_unemb                    // ℝ^{T × |V|}
5. RETURN logits
```

---

## 3. Compression Pipeline: Quantization

### 3.1 Objective

Reduce model checkpoint size and inference memory footprint by quantizing MoE weights to MXFP4 format, enabling:
- gpt-oss-120b on a single 80GB GPU
- gpt-oss-20b on systems with as little as 16GB memory

### 3.2 MXFP4 Format

**Definition:** Microscaling FP4 (MXFP4) is a block-scaled 4-bit floating-point format where weights are quantized to 4.25 bits per parameter on average.

**Format structure:**
- Each block of $B$ weight elements shares a single scaling factor (exponent).
- Within each block, individual weights are represented in 4-bit floating-point (1 sign bit, 2 exponent bits, 1 mantissa bit, or similar micro-format).
- The shared exponent adds $\sim 8$ bits per block, amortized over $B$ elements, yielding $4 + 8/B \approx 4.25$ bits/param for typical block sizes ($B = 32$).

**Quantization equation:**

$$
\hat{w}_i = s_b \cdot Q_{\text{FP4}}(w_i / s_b)
$$

where $s_b = \max_{i \in \text{block}_b} |w_i|$ is the block scaling factor and $Q_{\text{FP4}}(\cdot)$ maps to the nearest representable FP4 value.

### 3.3 Quantization-Aware Training (QAT)

The model card states: *"We post-trained the models with quantization of the MoE weights to MXFP4 format."* This indicates quantization-aware training during post-training (not just post-hoc quantization).

**QAT procedure:**
- During post-training (RL stage), MoE weights use simulated MXFP4 quantization in the forward pass:

$$
\hat{\mathbf{W}} = Q_{\text{MXFP4}}(\mathbf{W})
$$

- Gradients are computed with respect to the full-precision weights $\mathbf{W}$ using straight-through estimation (STE):

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{W}}}
$$

- Only MoE weights are quantized (90+% of total parameters). Attention weights, embeddings, and normalization parameters remain in higher precision.

### 3.4 Compression Analysis

| Component | Original (FP16) | MXFP4 | Compression Ratio |
|---|---|---|---|
| MoE weights (120b) | 114.71B × 2 bytes = 229.4 GB | 114.71B × 0.53 bytes ≈ 60.8 GB | ~3.77× |
| Non-MoE weights | ~2.12B × 2 bytes ≈ 4.2 GB | Kept at higher precision | 1× |
| Total (120b) | ~233.6 GB | ~60.8 GiB | ~3.84× |

### 3.5 Information Preservation

**Invariant:** QAT ensures that the model's loss landscape is adapted to quantized weights, minimizing the quantization error:

$$
\mathbb{E}\left[\|\mathbf{W} - \hat{\mathbf{W}}\|_F^2\right] \leq \epsilon_{\text{QAT}}
$$

where $\epsilon_{\text{QAT}}$ is bounded by the FP4 representation granularity and block size.

**Empirical validation:** The reported benchmark scores (AIME 96.6%, GPQA 80.1%, etc.) are achieved with the quantized model, confirming negligible quality degradation.

### 3.6 Failure Modes

- **Outlier sensitivity:** Weight outliers within a quantization block can cause the shared scaling factor to be dominated by the outlier, reducing precision for other elements. Mitigated by outlier-aware block partitioning.
- **Gradient approximation error (STE):** Biased gradient estimates can cause suboptimal convergence. Mitigated by careful learning rate scheduling during QAT.
- **Activation precision:** Only weights are quantized; activations remain in higher precision (likely BF16/FP16), which is critical for maintaining numerical stability in the attention softmax and router softmax.

---

## 4. Training Pipeline

### 4.1 Pretraining

#### 4.1.1 Hardware and Framework

- **Hardware:** NVIDIA H100 GPUs.
- **Framework:** PyTorch with expert-optimized Triton kernels.
- **Attention:** FlashAttention algorithms for memory reduction and training acceleration.

#### 4.1.2 Compute Budget

| Model | H100-Hours |
|---|---|
| gpt-oss-120b | 2.1 million |
| gpt-oss-20b | ~210,000 (≈10× fewer) |

**Approximate FLOPs estimate (Chinchilla scaling):**

For $C$ H100-hours at peak $\sim 990$ TFLOP/s (BF16 MFU-adjusted, assume ~40% MFU):

$$
\text{Total FLOPs}_{120b} \approx 2.1 \times 10^6 \times 3600 \times 990 \times 10^{12} \times 0.4 \approx 3.0 \times 10^{24} \text{ FLOPs}
$$

Using the approximation $C \approx 6 N_{\text{active}} T$ where $N_{\text{active}} = 5.13 \times 10^9$:

$$
T \approx \frac{C}{6 N_{\text{active}}} = \frac{3.0 \times 10^{24}}{6 \times 5.13 \times 10^9} \approx 9.7 \times 10^{13} \text{ tokens}
$$

This is consistent with "trillions of tokens" as stated.

#### 4.1.3 Distributed Training Strategy

Given the model architecture (MoE with 128 experts, 116.8B total params):

- **Expert Parallelism (EP):** Distributes experts across devices. With 128 experts, natural partition across multiple nodes.
- **Tensor Parallelism (TP):** For attention and shared components across GPUs within a node.
- **Data Parallelism (DP):** Across nodes for throughput scaling.
- **Sequence Parallelism:** For activation memory reduction in long-context training.
- **Activation Checkpointing:** Trades compute for memory, essential at this scale.

#### 4.1.4 Pretraining Objective

Standard autoregressive language modeling:

$$
\mathcal{L}_{\text{pretrain}} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

where $P_\theta(x_t \mid x_{<t}) = \text{softmax}(\text{logits}_t)[x_t]$.

#### 4.1.5 Training Pseudo-Algorithm

```
ALGORITHM: Pretraining
INPUT: Tokenized corpus D, model θ, optimizer (AdamW), scheduler, QAT config
OUTPUT: Pretrained model θ*

1. Initialize θ with appropriate initialization scheme
2. FOR each training step s = 1, 2, ..., S:
   a. Sample mini-batch B from D according to domain weights {w_d}
   b. Distribute B across DP × TP × EP dimensions
   c. Forward pass (with FlashAttention, alternating attention patterns):
      logits = GPT_OSS_Forward(B, θ)
   d. Compute loss:
      L = CrossEntropy(logits, targets) + λ_balance · L_balance
      where L_balance is the expert load-balancing auxiliary loss
   e. Backward pass with activation checkpointing
   f. All-reduce gradients across DP dimension
   g. Gradient clipping: clip_grad_norm_(θ, max_norm)
   h. Optimizer step with learning rate from scheduler
   i. IF s mod checkpoint_interval == 0:
      Save checkpoint
3. RETURN θ*
```

### 4.2 Post-Training: Reasoning and Tool Use via RL

#### 4.2.1 Objective

Transform the pretrained base model into a reasoning model capable of:
- Chain-of-thought (CoT) reasoning
- Variable effort reasoning (low, medium, high)
- Agentic tool use (browsing, Python execution, arbitrary function calling)
- Instruction hierarchy adherence
- Safety refusal (deliberative alignment)

#### 4.2.2 RL Training (o-series Techniques)

The model card states: *"We post-train the models using similar CoT RL techniques as OpenAI o3."*

**Core RL formulation:**

Given a prompt $x$, the model generates a response $y = (\text{CoT}, \text{answer})$. The reward signal $r(x, y)$ combines:

$$
r(x, y) = r_{\text{correctness}}(x, y) + \lambda_{\text{format}} \cdot r_{\text{format}}(y) + \lambda_{\text{safety}} \cdot r_{\text{safety}}(x, y) + \lambda_{\text{tool}} \cdot r_{\text{tool}}(x, y)
$$

where:
- $r_{\text{correctness}}$: Verifiable reward signal (outcome-based, e.g., math answer matches ground truth, code passes tests).
- $r_{\text{format}}$: Rewards proper use of harmony chat format, channel markers, role adherence.
- $r_{\text{safety}}$: Deliberative alignment reward — model should refuse disallowed content, adhere to instruction hierarchy.
- $r_{\text{tool}}$: Rewards correct tool invocation syntax and effective tool use.

**RL objective (RLVR-style with KL regularization):**

$$
\mathcal{L}_{\text{RL}} = -\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}\left[r(x, y)\right] + \beta \cdot D_{\text{KL}}\!\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]
$$

where $\pi_{\text{ref}}$ is the reference policy (pretrained model or previous RL checkpoint) and $\beta$ controls the KL penalty to prevent reward hacking and maintain language quality.

#### 4.2.3 Variable Effort Reasoning Training

Three reasoning levels: low, medium, high. Configured via system prompt keyword "Reasoning: {level}".

**Mechanism:** The model is trained with reward shaping that modulates the optimal CoT length:

- **Low:** Short CoT, penalize excessive reasoning length. Average CoT: ~2k tokens.
- **Medium:** Moderate CoT. Average CoT: ~4-8k tokens.
- **High:** Long CoT, reward thorough reasoning. Average CoT: ~16-32k tokens.

**Test-time scaling property:** Figure 3 in the model card shows log-linear accuracy scaling with CoT length:

$$
\text{Accuracy} \approx a + b \cdot \log(\text{CoT length})
$$

This demonstrates smooth test-time compute scaling, a hallmark of well-trained reasoning models.

#### 4.2.4 Deliberative Alignment

The model is trained to explicitly reason about safety policies in its CoT before deciding whether to comply with a request:

1. Model receives a potentially unsafe prompt.
2. In the CoT (analysis channel), the model deliberates: identifies relevant safety policy, assesses whether the request violates it.
3. Model either refuses (in final channel) or complies if the request is safe.

**Key design choice:** *No direct optimization pressure on the CoT content.* The CoT is not penalized for containing "bad thoughts" — only the final output is evaluated against safety policies. This preserves CoT monitorability and avoids the risk of the model learning to hide its reasoning.

#### 4.2.5 Harmony Chat Format Training

**Role hierarchy (strict ordering):**

$$
\text{System} > \text{Developer} > \text{User} > \text{Assistant} > \text{Tool}
$$

When instructions from different roles conflict, the model is trained to follow the higher-priority role.

**Channels:**
- `analysis`: CoT tokens (not shown to users by default)
- `commentary`: Function tool calling, intermediate reasoning
- `final`: Answers shown to users

**Training data includes:**
- Examples of role conflicts at each pair of hierarchy levels
- Supervised fine-tuning on correct resolution of conflicts
- RL reward for correct hierarchy adherence

#### 4.2.6 Tool Use Training

Three tool types:

1. **Browsing tool:** `search(query)` and `open(url)` functions for web interaction.
2. **Python tool:** Stateful Jupyter notebook environment for code execution.
3. **Developer-defined functions:** Arbitrary function schemas specified in Developer messages.

The model learns to interleave: CoT → tool call → tool response → more CoT → intermediate user message → final answer.

**Training procedure:**
- Mixed RL training with tool-augmented and tool-free problems.
- System prompt specifies available tools; model must decide when/whether to use them.
- Tool call syntax follows harmony format conventions.

#### 4.2.7 QAT Integration During Post-Training

MoE weights are quantized to MXFP4 during post-training via straight-through estimation, ensuring the final released checkpoint operates natively at 4.25 bits/param for MoE components.

#### 4.2.8 Post-Training Pseudo-Algorithm

```
ALGORITHM: Post-Training (CoT RL with Tool Use)
INPUT: Pretrained model θ_0, RL dataset D_RL, reward functions {r_*},
       tool environments {browsing, python, functions}, QAT config
OUTPUT: Post-trained model θ*

1. θ ← θ_0, π_ref ← θ_0
2. FOR each RL iteration i = 1, 2, ..., I:
   a. Sample batch of problems {x_j} from D_RL
   b. FOR each x_j:
      i.   Set reasoning level ∈ {low, medium, high} (sampled or scheduled)
      ii.  Set available tools based on system prompt
      iii. Generate K rollouts: {y_j^k} ~ π_θ(·|x_j)
           - Rollouts may include tool calls; execute tools and append responses
      iv.  Compute rewards: r_j^k = r(x_j, y_j^k)
   c. Compute RL loss with KL regularization:
      L_RL = -E[r] + β · D_KL(π_θ || π_ref)
   d. Add auxiliary losses:
      L_total = L_RL + λ_balance · L_balance + λ_safety · L_safety_hierarchy
   e. Backward pass with STE for MXFP4 quantized MoE weights
   f. Gradient clipping and optimizer step
   g. Periodically update π_ref ← θ (or keep fixed)
3. Final QAT calibration pass
4. Export quantized checkpoint
5. RETURN θ*
```

---

## 5. Optimization Strategy

### 5.1 Optimizer

Standard for large-scale Transformer training: **AdamW** with decoupled weight decay.

$$
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
$$

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda_{\text{wd}} \boldsymbol{\theta}_t\right)
$$

### 5.2 Learning Rate Scheduling

- Warmup phase → cosine decay (standard for pretraining).
- RL post-training typically uses a smaller, possibly constant or linearly decaying learning rate.

### 5.3 Gradient Clipping

Global gradient norm clipping with threshold $\tau_{\text{clip}}$:

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1, \frac{\tau_{\text{clip}}}{\|\mathbf{g}\|_2}\right)
$$

### 5.4 Mixed Precision

- Forward pass: BF16 activations, MXFP4 MoE weights (during QAT stages), BF16/FP32 for non-MoE weights.
- Backward pass: BF16 gradients with FP32 master weights for optimizer state accumulation.
- FlashAttention operates in BF16 with numerically stable online softmax.

### 5.5 Numerical Stability Considerations

- **RMSNorm:** Uses $\epsilon = 10^{-5}$ or similar to prevent division by zero.
- **Clamped SwiGLU:** Explicit clamping in expert FFN prevents activation explosion under quantized weights.
- **Learned softmax bias:** Must be initialized near zero to avoid initial attention collapse.
- **Expert residual connection:** Stabilizes gradient flow through quantized experts.

---

## 6. Inference Path

### 6.1 Autoregressive Decoding

Standard left-to-right token generation with KV caching.

#### 6.1.1 KV Cache Management

**GQA KV cache size per token per layer:**

$$
\text{KV}_{\text{per\_token\_per\_layer}} = 2 \times n_{kv} \times d_h = 2 \times 8 \times 64 = 1{,}024 \text{ elements}
$$

**Total KV cache for full context (dense layers, BF16):**

For $L_{\text{dense}} = 18$ dense layers (half of 36 for 120b):

$$
\text{KV}_{\text{dense}} = L_{\text{dense}} \times T \times 1{,}024 \times 2 \text{ bytes}
$$

At $T = 131{,}072$: $18 \times 131{,}072 \times 1{,}024 \times 2 \approx 4.83 \text{ GB}$

For banded-window layers ($L_{\text{banded}} = 18$), only $w = 128$ tokens need to be cached:

$$
\text{KV}_{\text{banded}} = L_{\text{banded}} \times w \times 1{,}024 \times 2 = 18 \times 128 \times 1{,}024 \times 2 \approx 4.72 \text{ MB}
$$

**Total KV cache is dominated by dense layers.** The alternating banded-window design reduces total KV cache by ~50% compared to all-dense attention.

#### 6.1.2 MoE Inference Efficiency

- Only $k = 4$ experts are activated per token per layer.
- Expert weights must be loaded from memory for each selected expert.
- With MXFP4, expert weight loading bandwidth is reduced by ~3.7× compared to FP16.

### 6.2 Memory Budget Analysis (Single GPU)

**gpt-oss-120b on 80GB GPU:**

| Component | Size |
|---|---|
| Model weights (MXFP4 MoE + BF16 others) | ~60.8 GiB |
| KV cache (varies with context) | ~4.8 GB at max context |
| Activations & workspace | ~10-14 GB |
| **Total** | **~76-80 GB** ✓ fits on 80GB |

**gpt-oss-20b on 16GB system:**

| Component | Size |
|---|---|
| Model weights | ~12.8 GiB |
| KV cache | ~3.2 GB (with 24 layers, proportional) |
| Activations & workspace | ~1-2 GB |
| **Total** | **~17 GB** — tight fit; requires careful memory management |

### 6.3 Tool-Augmented Inference

```
ALGORITHM: Tool-Augmented Autoregressive Generation
INPUT: prompt (in harmony format), available_tools, reasoning_level
OUTPUT: final_answer, full_trace (CoT + tool calls + tool responses)

1. context ← [system_prompt(reasoning_level, available_tools), user_message]
2. WHILE not end_of_generation:
   a. Generate tokens autoregressively using KV-cached decoding
   b. IF generated token sequence matches tool_call_pattern:
      i.   Parse tool call (function name, arguments)
      ii.  Execute tool in appropriate environment:
           - browsing: HTTP request via search/open functions
           - python: Execute in stateful Jupyter kernel
           - function: Call developer-defined function schema
      iii. Append tool_response to context in Tool role
      iv.  Continue generation
   c. IF channel marker is 'final':
      Mark as user-visible output
   d. IF end-of-sequence token generated:
      BREAK
3. Strip analysis-channel tokens (CoT) from user-visible output
4. RETURN final_answer, full_trace
```

### 6.4 Multi-Turn Conversation Handling

**Critical implementation detail:** In multi-turn conversations, reasoning traces (CoT) from past assistant turns **must be removed** from the context to:
1. Prevent context window exhaustion (CoT can be 20k+ tokens per turn).
2. Avoid confusing the model with stale reasoning.
3. Maintain inference efficiency.

### 6.5 Inference Failure Modes

- **Hallucination:** Model generates factually incorrect information. Particularly pronounced for smaller model (gpt-oss-20b: 91.4% hallucination rate on SimpleQA vs. 78.2% for 120b). Mitigated by tool use (browsing).
- **CoT hallucination:** Chain-of-thought contains fabricated reasoning steps or factually incorrect intermediate claims. No direct optimization pressure on CoT means this is unmitigated by design (to preserve monitorability).
- **Tool invocation errors:** Malformed tool calls, incorrect argument formatting. Mitigated by training on tool-use examples.
- **Expert routing imbalance at inference:** Some tokens may experience degraded quality if routed to poorly-quantized experts.

---

## 7. Evaluation Protocol

### 7.1 Benchmark Suite

#### 7.1.1 Reasoning and Factuality

| Benchmark | Metric | Description |
|---|---|---|
| AIME 2024/2025 | Accuracy (pass@1) | Competition math, with/without tools |
| GPQA Diamond | Accuracy | PhD-level science questions |
| MMLU | Accuracy | College-level exam questions |
| HLE | Accuracy | Expert-level questions |

#### 7.1.2 Coding

| Benchmark | Metric | Description |
|---|---|---|
| Codeforces | Elo rating | Competition programming |
| SWE-bench Verified | Accuracy (pass@1) | Real-world GitHub issue resolution |
| Aider Polyglot | Accuracy | Multi-language code generation |

#### 7.1.3 Tool Use

| Benchmark | Metric | Description |
|---|---|---|
| τ-Bench Retail | Accuracy | Function calling in retail domain |
| τ-Bench Airline | Accuracy | Function calling in airline domain |

#### 7.1.4 Health

| Benchmark | Metric | Description |
|---|---|---|
| HealthBench | Score (%) | Realistic health conversations |
| HealthBench Hard | Score (%) | Challenging health conversations |
| HealthBench Consensus | Score (%) | Physician-validated subset |

#### 7.1.5 Multilingual

| Benchmark | Metric | Description |
|---|---|---|
| MMMLU | Accuracy | Human-translated MMLU in 14 languages |

#### 7.1.6 Safety

| Benchmark | Metric | Description |
|---|---|---|
| Standard Disallowed Content | not_unsafe rate | LLM-graded safety compliance |
| Production Benchmarks | not_unsafe rate | Multi-turn challenging safety scenarios |
| StrongReject | not_unsafe rate | Jailbreak robustness |
| Instruction Hierarchy | Success rate | System > Developer > User adherence |
| SimpleQA | Accuracy / Hallucination rate | Factual accuracy on short-answer questions |
| PersonQA | Accuracy / Hallucination rate | Accuracy on person-related facts |
| BBQ | Accuracy | Fairness and bias evaluation |

### 7.2 Evaluation Configuration

- **Reasoning level:** All main results reported at `high` unless otherwise specified.
- **System prompt:** Model's default system prompt.
- **Decoding:** pass@1 (single sample, greedy or near-greedy).
- **Tool access:** Explicitly noted per benchmark (with/without tools).
- **Browsing contamination control:** Domain blocklist + classifier-based cheating detection + manual review of flagged rollouts.
- **Variable-effort evaluation:** Sweep over low/medium/high and plot accuracy vs. CoT+Answer length.

### 7.3 Evaluation Pseudo-Algorithm

```
ALGORITHM: Benchmark Evaluation
INPUT: model θ, benchmark B, reasoning_level, tool_config
OUTPUT: metrics (accuracy, Elo, score, hallucination_rate)

1. FOR each (prompt, ground_truth) in B:
   a. Construct input in harmony format:
      - System prompt with reasoning_level
      - Tool specifications (if tool_config enabled)
      - User prompt
   b. Generate response y ~ π_θ(·|input) with pass@1
   c. IF tool calls present in y:
      Execute tools, append responses, continue generation
   d. Extract final answer from 'final' channel
   e. Grade:
      - Exact match / rubric-based / LLM-graded (benchmark-specific)
      - For safety: not_unsafe classification
      - For hallucination: correct / incorrect / abstained
2. Aggregate metrics:
   accuracy = correct / total
   hallucination_rate = incorrect / (correct + incorrect)
3. RETURN metrics
```

### 7.4 Key Benchmark Results (High Reasoning)

| Benchmark | gpt-oss-120b | gpt-oss-20b | o4-mini | o3 |
|---|---|---|---|---|
| AIME 2025 (tools) | 97.9% | 98.7% | 99.5% | 98.4% |
| GPQA Diamond (no tools) | 80.1% | 71.5% | 81.4% | 83.3% |
| MMLU | 90.0% | 85.3% | — | 93.4% |
| HLE (tools) | 19.0% | 17.3% | 17.7% | 24.9% |
| SWE-bench Verified | 62.4% | 60.7% | 68.1% | 69.1% |
| Codeforces Elo (tools) | 2622 | 2516 | 2719 | 2706 |
| HealthBench | 57.6% | 42.5% | 50.1% | 59.8% |

**Observation:** gpt-oss-120b consistently exceeds o3-mini and approaches o4-mini. gpt-oss-20b is surprisingly competitive despite being 6× smaller than gpt-oss-120b.

---

## 8. Safety and Preparedness Evaluation

### 8.1 Preparedness Framework Tracked Categories

Three tracked categories: **Biological and Chemical**, **Cybersecurity**, **AI Self-Improvement**. Each is assessed against indicative thresholds for "High" capability.

**Result:** Default gpt-oss-120b does **not** reach High capability in any category.

### 8.2 Adversarial Fine-Tuning Methodology

To simulate worst-case open-weight misuse:

1. **Helpful-only training:** Additional RL stage rewarding compliance with unsafe prompts (removes safety refusals).
2. **Domain-specific capability maximization:**
   - **Bio:** End-to-end web browsing training + in-domain human expert biorisk data.
   - **Cyber:** Training on cybersecurity CTF challenge environments.

**Key finding:** Even adversarially fine-tuned gpt-oss-120b does **not** reach High capability in Bio/Chemical or Cyber categories.

**Frontier comparison:** Existing open models (DeepSeek R1-0528, Qwen 3 Thinking, Kimi K2) already achieve comparable biosecurity-relevant scores, meaning gpt-oss-120b does not significantly advance the open-weight frontier for hazardous biological capabilities.

### 8.3 Safety Training Components

| Component | Mechanism |
|---|---|
| CBRN pretraining filters | Classifier-based removal of harmful biosecurity content |
| Deliberative alignment | CoT-based policy reasoning before compliance decisions |
| Instruction hierarchy | Trained role-based priority resolution |
| Jailbreak robustness | StrongReject-evaluated adversarial prompt resistance |
| CoT non-interference | No optimization pressure on CoT content (preserves monitorability) |

### 8.4 Safety Failure Modes

- **Instruction hierarchy weakness:** gpt-oss models underperform o4-mini on system prompt extraction and prompt injection hijacking. Developers have less ability to use system messages as a mitigation layer.
- **Hallucinated CoT:** Chains of thought may contain content violating safety policies. Must not be shown to users without filtering.
- **Hallucination rate:** gpt-oss-120b: 78.2% on SimpleQA, 49.1% on PersonQA (without browsing). Significantly worse than o4-mini.
- **Open-weight fine-tuning risk:** Safety refusals can be removed by adversarial fine-tuning. Mitigated at the ecosystem level (no single model significantly advances hazardous capabilities beyond existing open-weight frontier).

---

## 9. Deployment Constraints

### 9.1 Hardware Requirements

| Model | Minimum Hardware | Recommended |
|---|---|---|
| gpt-oss-120b | Single 80GB GPU (A100/H100) | H100 80GB for full performance |
| gpt-oss-20b | 16GB GPU (RTX 4060 Ti 16GB, etc.) | 24GB+ for comfortable headroom |

### 9.2 Serving Considerations

- **License:** Apache 2.0 + gpt-oss usage policy.
- **API compatibility:** Designed for OpenAI Responses API.
- **Chat format:** Must use harmony chat format for correct behavior. Incorrect formatting significantly degrades capabilities.
- **Multi-turn context management:** Strip past CoT traces to prevent context exhaustion.
- **Tool harnesses:** Reference implementations provided for browsing, Python, and function calling.
- **Structured Outputs:** Supported natively.
- **Reasoning level selection:** System prompt configuration; users should balance accuracy vs. latency/cost.

### 9.3 Latency-Throughput-Cost Analysis

**Test-time compute scaling tradeoff:**

$$
\text{Cost} \propto \text{CoT tokens} + \text{Answer tokens}
$$

| Reasoning Level | Avg CoT (AIME, 120b) | AIME 2025 Accuracy | Relative Cost |
|---|---|---|---|
| Low | ~2k tokens | 50.4% | 1× |
| Medium | ~8k tokens | 80.0% | ~4× |
| High | ~20k tokens | 92.5% | ~10× |

The log-linear accuracy-cost relationship means users face a Pareto frontier:

$$
\text{Marginal accuracy gain} = \frac{\Delta \text{Acc}}{\Delta \log(\text{Cost})} \approx b \text{ (constant)}
$$

### 9.4 Deployment Pseudo-Algorithm

```
ALGORITHM: Production Serving Pipeline
INPUT: User request, system_config (reasoning_level, tools, developer_message)
OUTPUT: Response to user

1. VALIDATE harmony format compliance of input
2. CONSTRUCT full prompt:
   - System message (with reasoning level, tool specifications)
   - Developer message (custom guardrails, function schemas)
   - Conversation history (with CoT stripped from prior turns)
   - Current user message
3. CHECK context length ≤ 131,072 tokens
4. LOAD model weights (MXFP4 MoE + BF16 others)
5. INITIALIZE KV cache
6. GENERATE response via tool-augmented autoregressive decoding
7. POST-PROCESS:
   a. Separate analysis (CoT) from final (user-visible) content
   b. Apply content moderation / filtering on final output
   c. Optionally: monitor CoT for safety signals (without showing to user)
   d. Apply Structured Output constraints if specified
8. RETURN final response to user
9. LOG: latency, token counts, tool calls, safety flags
```

### 9.5 Production Failure Modes and Mitigations

| Failure Mode | Impact | Mitigation |
|---|---|---|
| OOM on long context | Inference crash | Dynamic context length limiting, KV cache eviction |
| Tool execution timeout | Stalled generation | Timeout limits per tool call, fallback to no-tool response |
| Harmony format misuse | Degraded model quality | Strict input validation, reference implementation |
| CoT shown to users | Safety-violating content exposure | Mandatory CoT filtering before user display |
| Instruction hierarchy bypass | Developer guardrails ineffective | Fine-tune for specific deployment; add output moderation |
| Expert routing hot-spot | Latency spike | Batch-level expert load monitoring, request routing |

---

## 10. Convergence Dynamics and Scaling Laws

### 10.1 MoE Scaling Properties

For MoE models, the effective compute scales with active parameters $N_a$ rather than total parameters $N$:

$$
L(N_a, D) \approx \left(\frac{N_c}{N_a}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}
$$

where $L$ is loss, $D$ is data size in tokens, and $\alpha_N, \alpha_D$ are scaling exponents. The MoE architecture achieves better loss per FLOP than dense models by increasing $N$ (capacity) while keeping $N_a$ (compute per token) fixed.

**gpt-oss-120b regime:**
- $N/N_a = 116.8/5.13 \approx 22.7$ — high expert multiplier
- This ratio is in the favorable regime where expert specialization provides significant quality uplift over a 5B-parameter dense model.

### 10.2 Test-Time Compute Scaling

The variable-effort reasoning training creates a controllable test-time compute axis:

$$
\text{Accuracy}(\text{effort}) \approx a + b \cdot \log\!\left(\text{CoT length}(\text{effort})\right)
$$

Empirically validated across AIME and GPQA (Figure 3 of model card). This is consistent with the theoretical framework where additional reasoning tokens provide information-theoretic gains in problem solving, subject to diminishing returns.

### 10.3 Size Scaling: 120b vs. 20b

| Property | 120b | 20b | Ratio |
|---|---|---|---|
| Total params | 116.83B | 20.91B | 5.59× |
| Active params | 5.13B | 3.61B | 1.42× |
| Experts | 128 | 32 | 4× |
| AIME 2025 (high, tools) | 97.9% | 98.7% | 20b wins |
| GPQA Diamond (no tools) | 80.1% | 71.5% | 120b wins by 8.6pp |
| SimpleQA hallucination | 78.2% | 91.4% | 120b much better |

**Key observation:** On reasoning-heavy tasks (AIME), the smaller model is competitive because long CoT compensates for reduced knowledge. On knowledge-intensive tasks (GPQA, SimpleQA), the 120b model's larger expert capacity provides significant advantage. This is consistent with the hypothesis that MoE expert count primarily contributes stored knowledge rather than reasoning ability.

---

## 11. Summary of Mathematical Formulations

| Component | Equation |
|---|---|
| RMSNorm | $\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$ |
| GQA Attention | $\alpha_{s,t} = \frac{\exp(q_s^T k_t / \sqrt{d_h} + M_{s,t})}{\sum_{t'} \exp(q_s^T k_{t'}/\sqrt{d_h} + M_{s,t'}) + \exp(b)}$ |
| RoPE | $\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(m\boldsymbol{\theta}) + \text{rotate\_half}(\mathbf{x}) \odot \sin(m\boldsymbol{\theta})$ |
| MoE Routing | $g_e = \frac{\exp(s_e)}{\sum_{e' \in \mathcal{S}} \exp(s_{e'})}$, $\mathcal{S} = \text{Top-}k(\mathbf{s})$ |
| SwiGLU (modified) | $\text{Expert}(\mathbf{x}) = \text{clamp}(\text{SiLU}(\mathbf{x}\mathbf{W}_1) \odot (\mathbf{x}\mathbf{W}_3))\mathbf{W}_2 + \alpha\mathbf{x}$ |
| MXFP4 Quantization | $\hat{w}_i = s_b \cdot Q_{\text{FP4}}(w_i / s_b)$ |
| Pretraining Loss | $\mathcal{L} = -\frac{1}{T}\sum_t \log P_\theta(x_t|x_{<t})$ |
| RL Objective | $\mathcal{L}_{\text{RL}} = -\mathbb{E}[r(x,y)] + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$ |
| Test-Time Scaling | $\text{Acc} \approx a + b \cdot \log(\text{CoT length})$ |