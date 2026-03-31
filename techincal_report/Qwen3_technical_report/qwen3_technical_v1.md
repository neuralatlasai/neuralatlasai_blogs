

# Qwen3: End-to-End Technical Report — Stage-Wise Reconstruction

---

## 1. Data Pipeline

### 1.1 Formal Problem Statement

**Objective:** Construct a pre-training corpus $\mathcal{D} = \{x_1, x_2, \dots, x_N\}$ of $N \approx 36 \times 10^{12}$ tokens spanning $L = 119$ languages/dialects across domains $\mathcal{G} = \{\text{code}, \text{STEM}, \text{reasoning}, \text{books}, \text{multilingual}, \text{synthetic}\}$, with instance-level quality annotations enabling fine-grained mixture optimization.

**Inputs:**
- Raw web crawl, books, code repositories, PDF-format documents
- Pre-existing Qwen2.5 model family (Qwen2.5-VL, Qwen2.5, Qwen2.5-Math, Qwen2.5-Coder)
- Language identification classifiers for 119 languages

**Outputs:**
- Tokenized corpus $\mathcal{D}$ with per-instance annotations $\{(\text{educational\_value}, \text{field}, \text{domain}, \text{safety})\}_i$ for each token sequence $x_i$
- Instance-level mixture weights $w_i$ optimized via proxy-model ablations

**Invariants:**
- Deduplication ensures $\forall i \neq j: \text{sim}(x_i, x_j) < \tau_{\text{dedup}}$
- Safety annotations filter sequences $x_i$ where $\text{safety}(x_i) < \tau_{\text{safe}}$
- Language balance constraints: $\forall l \in [L], \; n_l \geq n_l^{\min}$

---

### 1.2 PDF Text Extraction Pipeline

**Stage 1 — Visual OCR via Qwen2.5-VL:**

Given a PDF page rendered as image $I \in \mathbb{R}^{H \times W \times 3}$:

$$\hat{t}_{\text{raw}} = \text{Qwen2.5-VL}(I)$$

where $\hat{t}_{\text{raw}}$ is the raw OCR text extraction.

**Stage 2 — Text Refinement via Qwen2.5:**

$$\hat{t}_{\text{refined}} = \text{Qwen2.5}(\hat{t}_{\text{raw}}, \; \text{prompt}_{\text{refine}})$$

This two-step process yields **trillions** of additional high-quality text tokens from previously inaccessible PDF-locked content.

**Failure Modes:**
- OCR hallucination on mathematical notation, tables, multi-column layouts
- Language confusion in multilingual PDFs
- Duplicated content from repeated document scraping

---

### 1.3 Synthetic Data Generation

Three specialized generators produce synthetic tokens:

| Generator | Domain | Output Formats |
|---|---|---|
| Qwen2.5 | General | Textbooks, QA, instructions |
| Qwen2.5-Math | Mathematics | Problems, solutions, proofs |
| Qwen2.5-Coder | Code | Snippets, documentation, test cases |

Each generator produces data in multiple formats: textbooks, question-answering pairs, instruction-response pairs, and code snippets across dozens of domains. Total synthetic contribution: **trillions of tokens**.

**Quality Control:**
- Each synthetic instance is verified against ground-truth or test-case execution
- Contamination detection against downstream evaluation benchmarks
- Deduplication against web-sourced corpus

---

### 1.4 Multilingual Data Annotation System

**Annotation Dimensions:** Each instance $x_i$ receives a multi-dimensional label vector:

$$\ell_i = (\text{edu\_value}_i, \; \text{field}_i, \; \text{domain}_i, \; \text{safety}_i, \; \text{lang}_i)$$

annotated across $>30 \times 10^{12}$ tokens.

**Instance-Level Mixture Optimization:**

Unlike prior work optimizing at source/domain granularity, Qwen3 solves:

$$w^* = \arg\min_{w \in \Delta^N} \; \mathcal{L}_{\text{proxy}}\!\left(\theta; \; \sum_{i=1}^{N} w_i \cdot x_i\right)$$

where $\mathcal{L}_{\text{proxy}}$ is evaluated on small proxy models via extensive ablation experiments, and $\Delta^N$ is the probability simplex. Fine-grained labels $\ell_i$ enable combinatorial data mixture strategies unreachable by coarse domain-level mixing.

---

### 1.5 Tokenization

**Algorithm:** Byte-Level Byte-Pair Encoding (BBPE)

**Vocabulary size:** $|\mathcal{V}| = 151{,}669$

**Properties:**
- Byte-level fallback ensures zero out-of-vocabulary tokens for any Unicode input
- Covers 119 languages without language-specific tokenizer fragmentation
- Shared across all Qwen3 model sizes

**Formal Tokenization Map:**

$$\text{Tokenize}: \Sigma^* \rightarrow \mathcal{V}^*$$

where $\Sigma = \{0, 1, \dots, 255\}$ is the byte alphabet.

**Compression Ratio:** For language $l$:

$$\rho_l = \frac{|\text{Tokenize}(x_l)|}{|x_l|_{\text{bytes}}}$$

Optimized such that $\rho_l$ is approximately uniform across high-resource and low-resource languages.

---

### 1.6 Pseudo-Algorithm: Data Pipeline

```
ALGORITHM: Qwen3DataPipeline
INPUT: Raw corpus sources S, PDF documents P, language list L[119]
OUTPUT: Annotated tokenized corpus D

1.  FOR each PDF p ∈ P:
      I ← Render(p)
      t_raw ← Qwen2.5_VL(I)
      t_refined ← Qwen2.5(t_raw, prompt_refine)
      ADD t_refined TO D_pdf

2.  FOR domain ∈ {math, code, general}:
      generator ← SELECT(Qwen2.5-Math, Qwen2.5-Coder, Qwen2.5)
      D_synth[domain] ← generator.generate(formats=[textbook, QA, instruction, code])
      VERIFY D_synth[domain] against test cases / ground truth

3.  D_raw ← WebCrawl(S) ∪ D_pdf ∪ D_synth
4.  D_dedup ← MinHashDedup(D_raw, threshold=τ_dedup)
5.  FOR each x_i ∈ D_dedup:
      ℓ_i ← AnnotationSystem(x_i)  // (edu_value, field, domain, safety, lang)
      IF safety(x_i) < τ_safe: REMOVE x_i

6.  w* ← InstanceMixtureOptimization(D_dedup, ℓ, proxy_models)
7.  D ← BBPE_Tokenize(D_dedup, vocab_size=151669)
8.  RETURN D, w*
```

---

## 2. Model Architecture

### 2.1 Dense Transformer Architecture

#### 2.1.1 Formal Specification

Each dense Qwen3 model is a decoder-only autoregressive Transformer parameterized by:

$$\theta = \{W_E, \; \{W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}, W_O^{(l)}, W_{\text{up}}^{(l)}, W_{\text{gate}}^{(l)}, W_{\text{down}}^{(l)}, \gamma_{\text{attn}}^{(l)}, \gamma_{\text{ffn}}^{(l)}, \gamma_{Q}^{(l)}, \gamma_{K}^{(l)}\}_{l=1}^{L}, \; W_{\text{head}}\}$$

where $L$ is the number of Transformer layers.

**Model Configurations:**

| Model | $L$ | $d_{\text{model}}$ | $n_Q$ | $n_{KV}$ | $d_h$ | Tie Emb | Context |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 28 | — | 16 | 8 | — | Yes | 32K |
| Qwen3-1.7B | 28 | — | 16 | 8 | — | Yes | 32K |
| Qwen3-4B | 36 | — | 32 | 8 | — | Yes | 128K |
| Qwen3-8B | 36 | — | 32 | 8 | — | No | 128K |
| Qwen3-14B | 40 | — | 40 | 8 | — | No | 128K |
| Qwen3-32B | 64 | — | 64 | 8 | — | No | 128K |

Where $n_Q$ = number of query heads, $n_{KV}$ = number of key-value heads (GQA groups).

---

#### 2.1.2 Layer-Wise Computation

For layer $l$, given input hidden states $H^{(l-1)} \in \mathbb{R}^{T \times d}$:

**Step 1 — Pre-Normalization (RMSNorm) for Attention:**

$$\hat{H}^{(l)} = \text{RMSNorm}(H^{(l-1)}; \gamma_{\text{attn}}^{(l)})$$

where:

$$\text{RMSNorm}(x; \gamma) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

with $\gamma \in \mathbb{R}^d$ learnable scale, $\epsilon$ numerical stability constant.

**Step 2 — Grouped Query Attention (GQA) with QK-Norm:**

Projections:

$$Q = \hat{H}^{(l)} W_Q^{(l)} \in \mathbb{R}^{T \times n_Q d_h}$$
$$K = \hat{H}^{(l)} W_K^{(l)} \in \mathbb{R}^{T \times n_{KV} d_h}$$
$$V = \hat{H}^{(l)} W_V^{(l)} \in \mathbb{R}^{T \times n_{KV} d_h}$$

**QK-Norm (replacing QKV-bias from Qwen2):**

$$\hat{Q} = \text{RMSNorm}(Q; \gamma_Q^{(l)}), \quad \hat{K} = \text{RMSNorm}(K; \gamma_K^{(l)})$$

This stabilizes the dot-product attention logits by controlling the norm of $Q$ and $K$, preventing attention entropy collapse during training at scale.

**Rationale for removing QKV-bias and introducing QK-Norm:**
- QKV-bias adds $3 \times d_{\text{model}} \times n_{\text{heads}} \times d_h$ parameters with marginal benefit
- QK-Norm directly constrains $\|q\| \cdot \|k\|$, preventing logit explosion: $|q^Tk| \leq \|q\|\|k\| \cdot \cos\alpha$, bounded after normalization
- Empirically yields more stable gradients at $>100$B scale

**RoPE Application:**

For position $t$, head dimension index $i$:

$$\text{RoPE}(x, t)_{2i} = x_{2i} \cos(t \cdot \omega_i) - x_{2i+1} \sin(t \cdot \omega_i)$$
$$\text{RoPE}(x, t)_{2i+1} = x_{2i} \sin(t \cdot \omega_i) + x_{2i+1} \cos(t \cdot \omega_i)$$

where the frequency:

$$\omega_i = \theta_{\text{base}}^{-2i/d_h}$$

with $\theta_{\text{base}} = 10{,}000$ (S1, S2) or $\theta_{\text{base}} = 1{,}000{,}000$ (S3, long-context stage via ABF).

**Positional Encoding Properties:**
- Relative: $\text{RoPE}(q, m)^T \text{RoPE}(k, n) = f(q, k, m-n)$
- Decays with distance: attention naturally attenuates for distant tokens
- ABF (Adjusted Base Frequency): increasing $\theta_{\text{base}}$ compresses rotational frequencies, enabling extrapolation to longer sequences

**GQA Head Grouping:**

Each KV head is shared across $G = n_Q / n_{KV}$ query heads:

$$\text{Attn}_g(Q_g, K_j, V_j) = \text{softmax}\!\left(\frac{\hat{Q}_g \hat{K}_j^T}{\sqrt{d_h}} + M_{\text{causal}}\right) V_j$$

where $g \in \{1, \dots, G\}$ indexes query heads sharing KV head $j$, and $M_{\text{causal}} \in \{0, -\infty\}^{T \times T}$ is the causal mask.

**GQA Complexity Analysis:**

Standard MHA KV-cache per layer: $2 \times n_Q \times d_h \times T$ elements.

GQA KV-cache per layer: $2 \times n_{KV} \times d_h \times T$ elements.

**Compression ratio:**

$$\rho_{\text{KV}} = \frac{n_{KV}}{n_Q}$$

For Qwen3-32B: $\rho_{\text{KV}} = 8/64 = 0.125$, i.e., **8× KV-cache reduction** versus MHA.

**Output Projection:**

$$O^{(l)} = \text{Concat}(\text{Attn}_1, \dots, \text{Attn}_{n_Q}) \cdot W_O^{(l)}$$

**Step 3 — Residual Connection:**

$$H_{\text{mid}}^{(l)} = H^{(l-1)} + O^{(l)}$$

**Step 4 — Pre-Normalization (RMSNorm) for FFN:**

$$\hat{H}_{\text{mid}}^{(l)} = \text{RMSNorm}(H_{\text{mid}}^{(l)}; \gamma_{\text{ffn}}^{(l)})$$

**Step 5 — SwiGLU Feed-Forward Network:**

$$\text{FFN}(x) = \left[\text{SiLU}(x W_{\text{gate}}^{(l)}) \odot (x W_{\text{up}}^{(l)})\right] W_{\text{down}}^{(l)}$$

where:

$$\text{SiLU}(z) = z \cdot \sigma(z), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

Dimensions: $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ffn}}}$, $W_{\text{down}} \in \mathbb{R}^{d_{\text{ffn}} \times d}$.

SwiGLU uses $\frac{8}{3}d$ intermediate dimension (typically rounded to a multiple of 256 for hardware alignment), yielding parameter count per FFN layer: $3 \times d \times d_{\text{ffn}}$.

**Step 6 — Residual Connection:**

$$H^{(l)} = H_{\text{mid}}^{(l)} + \text{FFN}(\hat{H}_{\text{mid}}^{(l)})$$

---

#### 2.1.3 Embedding and Output Head

**Input Embedding:**

$$H^{(0)} = \text{Embed}(x) = W_E[x_1; x_2; \dots; x_T] \in \mathbb{R}^{T \times d}$$

where $W_E \in \mathbb{R}^{|\mathcal{V}| \times d}$.

**Output Logits:**

$$z_t = \text{RMSNorm}(H^{(L)}_t) \cdot W_{\text{head}}^T$$

where $W_{\text{head}} \in \mathbb{R}^{|\mathcal{V}| \times d}$.

**Tied Embedding** (for 0.6B, 1.7B, 4B): $W_{\text{head}} = W_E$, reducing parameter count by $|\mathcal{V}| \times d$.

**Untied Embedding** (for 8B, 14B, 32B): $W_{\text{head}} \neq W_E$, allowing independent specialization of input representation and output prediction.

---

### 2.2 Mixture-of-Experts (MoE) Architecture

#### 2.2.1 Structural Specification

| Model | $L$ | $n_Q / n_{KV}$ | Total Experts $E$ | Active Experts $k$ | Context |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48 | 32/4 | 128 | 8 | 128K |
| Qwen3-235B-A22B | 94 | 64/4 | 128 | 8 | 128K |

**Key Design Decisions:**
- **Fine-grained expert segmentation:** 128 total experts with 8 active, following Dai et al. (2024). Each expert is smaller than a standard FFN, enabling finer routing granularity.
- **No shared experts:** Unlike Qwen2.5-MoE, all experts are routed; no unconditional shared computation. This forces routing to fully specialize.
- **Global-batch load balancing loss:** Replaces per-sample auxiliary loss.

---

#### 2.2.2 MoE Layer Computation

Each MoE FFN layer replaces the dense FFN with:

**Router:**

$$g(x) = \text{TopK}\!\left(\text{softmax}(x \cdot W_r), \; k=8\right)$$

where $W_r \in \mathbb{R}^{d \times E}$, $E = 128$.

Let $\mathcal{S}_t = \{e_1, \dots, e_k\}$ denote the set of selected expert indices for token $t$, and $\alpha_{t,e}$ the routing weight for expert $e$:

$$\alpha_{t,e} = \frac{\exp(x_t \cdot w_r^{(e)})}{\sum_{e' \in \mathcal{S}_t} \exp(x_t \cdot w_r^{(e')})}$$

**Expert Computation:**

$$\text{MoE-FFN}(x_t) = \sum_{e \in \mathcal{S}_t} \alpha_{t,e} \cdot \text{FFN}_e(x_t)$$

where each $\text{FFN}_e$ is a SwiGLU network with reduced dimensions (fine-grained segmentation):

$$\text{FFN}_e(x) = \left[\text{SiLU}(x W_{\text{gate}}^{(e)}) \odot (x W_{\text{up}}^{(e)})\right] W_{\text{down}}^{(e)}$$

---

#### 2.2.3 Global-Batch Load Balancing Loss

**Problem:** Per-sample auxiliary losses cause expert collapse or load imbalance at scale. Token-level balancing also introduces noise.

**Solution (Qiu et al., 2025):** Compute load balance across the entire global batch $\mathcal{B}$:

$$\mathcal{L}_{\text{bal}} = E \cdot \sum_{e=1}^{E} f_e \cdot p_e$$

where:

$$f_e = \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \mathbb{1}[e \in \mathcal{S}_t]$$

is the fraction of tokens routed to expert $e$ across the global batch, and:

$$p_e = \frac{1}{|\mathcal{B}|} \sum_{t \in \mathcal{B}} \text{softmax}(x_t \cdot W_r)_e$$

is the average routing probability for expert $e$.

**Properties:**
- Encourages $f_e \approx k/E = 8/128 = 1/16$ for all $e$
- Global aggregation reduces variance compared to per-sample or micro-batch balancing
- Promotes expert specialization by avoiding oscillatory routing

---

#### 2.2.4 Computational Complexity

**Dense model per-layer FLOPs (forward pass, per token):**

$$\text{FLOPs}_{\text{dense}} = \underbrace{4 d^2 + 4 n_{KV} d d_h}_{\text{Attention}} + \underbrace{3 d \cdot d_{\text{ffn}}}_{\text{FFN}} + \underbrace{2T d}_{\text{Attn logits}} $$

**MoE model per-layer FLOPs (forward pass, per token):**

$$\text{FLOPs}_{\text{MoE}} = \underbrace{4 d^2 + 4 n_{KV} d d_h}_{\text{Attention (same)}} + \underbrace{k \cdot 3 d \cdot d_{\text{ffn}}^{(e)}}_{\text{Active experts}} + \underbrace{d \cdot E}_{\text{Router}}$$

Since $k \cdot d_{\text{ffn}}^{(e)} \ll d_{\text{ffn}}^{\text{dense-equivalent}}$ due to fine-grained segmentation:

**Qwen3-235B-A22B:** 235B total parameters, 22B activated per token → **~10.7× parameter efficiency** over equivalent dense model at inference.

---

#### 2.2.5 Memory Flow Analysis

**Per-Layer KV-Cache (at inference):**

$$\text{KV}_{\text{mem}}^{(l)} = 2 \times n_{KV} \times d_h \times T \times b_{\text{bytes}}$$

For Qwen3-235B-A22B with $n_{KV}=4$, $d_h = d/n_Q$, at sequence length $T$:

$$\text{Total KV} = L \times 2 \times n_{KV} \times d_h \times T \times b_{\text{bytes}} = 94 \times 2 \times 4 \times d_h \times T \times 2$$

(at FP16, $b_{\text{bytes}} = 2$).

GQA with $n_{KV}=4$ provides **16× compression** versus MHA ($n_{KV}=n_Q=64$) for the flagship model.

**Expert Parameter Storage:**

All 128 experts per MoE layer must reside in memory (or be paged), even though only 8 are activated. Total expert parameters per layer:

$$\text{Expert}_{\text{params}}^{(l)} = E \times 3 \times d \times d_{\text{ffn}}^{(e)}$$

This dominates memory footprint and motivates tensor parallelism and expert parallelism strategies.

---

### 2.3 Pseudo-Algorithm: Single Forward Pass

```
ALGORITHM: Qwen3ForwardPass
INPUT: Token IDs x[1..T], model parameters θ
OUTPUT: Logits z[1..T] ∈ ℝ^{T × |V|}

1.  H ← W_E[x]                              // [T, d]

2.  FOR l = 1 TO L:
      // ---- Attention Block ----
      H_norm ← RMSNorm(H, γ_attn[l])
      Q ← H_norm · W_Q[l]                   // [T, n_Q · d_h]
      K ← H_norm · W_K[l]                   // [T, n_KV · d_h]
      V ← H_norm · W_V[l]                   // [T, n_KV · d_h]
      Q ← RMSNorm_per_head(Q, γ_Q[l])       // QK-Norm
      K ← RMSNorm_per_head(K, γ_K[l])       // QK-Norm
      Q ← Apply_RoPE(Q, positions, θ_base)
      K ← Apply_RoPE(K, positions, θ_base)
      // GQA: expand K, V to match Q head count
      K_exp ← Repeat_KV(K, n_Q / n_KV)
      V_exp ← Repeat_KV(V, n_Q / n_KV)
      A ← FlashAttention(Q, K_exp, V_exp, causal=True)
      O ← A · W_O[l]
      H ← H + O                             // Residual

      // ---- FFN Block (Dense or MoE) ----
      H_norm ← RMSNorm(H, γ_ffn[l])
      IF model.is_MoE AND layer_is_MoE(l):
        scores ← softmax(H_norm · W_r[l])   // [T, E]
        top_k_indices, top_k_weights ← TopK(scores, k=8)
        Renormalize top_k_weights
        FFN_out ← 0
        FOR each active expert e ∈ top_k_indices:
          FFN_out += top_k_weights[e] · SwiGLU_e(H_norm)
      ELSE:
        FFN_out ← SwiGLU(H_norm)
      H ← H + FFN_out                       // Residual

3.  z ← RMSNorm(H, γ_final) · W_head^T     // [T, |V|]
4.  RETURN z
```

---

## 3. Optimization Strategy

### 3.1 Pre-Training Objective

**Standard Causal Language Modeling:**

$$\mathcal{L}_{\text{CLM}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

where:

$$p_\theta(x_t \mid x_{<t}) = \text{softmax}(z_t)_{x_t}$$

**Total loss with MoE balancing:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CLM}} + \lambda_{\text{bal}} \cdot \mathcal{L}_{\text{bal}}$$

where $\lambda_{\text{bal}}$ is the load-balancing coefficient (typically $\lambda_{\text{bal}} \in [0.001, 0.01]$).

---

### 3.2 Scaling Laws for Hyperparameter Prediction

Qwen3 develops scaling laws that model the relationship:

$$\eta^*(\text{model\_size}, \text{stage}, \text{arch}) = f(N_{\text{params}}, \; D_{\text{stage}}, \; \text{dense/MoE})$$
$$B^*(\text{model\_size}, \text{stage}, \text{arch}) = g(N_{\text{params}}, \; D_{\text{stage}}, \; \text{dense/MoE})$$

where $\eta^*$ is the optimal peak learning rate and $B^*$ is the optimal batch size. These are fit via **extensive experiments on proxy models** across the three pre-training stages.

**Learning Rate Schedule:**

$$\eta(s) = \begin{cases}
\eta^* \cdot \frac{s}{s_{\text{warmup}}} & s \leq s_{\text{warmup}} \\
\eta^* \cdot \text{decay}(s) & s > s_{\text{warmup}}
\end{cases}$$

In S2 (Reasoning Stage): the learning rate decay is **accelerated** to enable sharper convergence on the higher-quality reasoning-intensive distribution.

**Optimizer:** AdamW with:

$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda_w \theta_t\right)$$

where $\hat{m}_t, \hat{v}_t$ are bias-corrected first and second moment estimates, and $\lambda_w$ is weight decay.

---

### 3.3 Numerical Stability Mechanisms

**QK-Norm Gradient Analysis:**

Without QK-Norm, attention logit magnitude scales as:

$$|q^T k| \leq \|q\| \cdot \|k\| \cdot |\cos\alpha|$$

where $\|q\|, \|k\|$ can grow unboundedly during training, causing:
- Softmax saturation ($\nabla \rightarrow 0$)
- Attention entropy collapse
- Training loss spikes at large scale

With QK-Norm:

$$\|\hat{q}\| = \|\hat{k}\| = \sqrt{d_h} \cdot \frac{\|\gamma\|}{\sqrt{d_h}} = \|\gamma\|$$

which is **bounded by a learnable parameter**, preventing logit divergence while preserving expressivity.

**Mixed Precision (BF16):**

- All matrix multiplications in BF16 (8-bit exponent, 7-bit mantissa)
- Loss scaling for gradient underflow prevention
- RMSNorm computed in FP32 for numerical fidelity
- Gradient accumulation in FP32

---

## 4. Training Stages

### 4.1 Pre-Training: Three-Stage Process

#### Stage S1 — General Pre-Training

**Objective:** Build broad language proficiency and world knowledge.

| Parameter | Value |
|---|---|
| Tokens | $\sim 30 \times 10^{12}$ |
| Sequence Length | 4,096 |
| Languages | 119 |
| RoPE $\theta_{\text{base}}$ | 10,000 |
| LR Schedule | Predicted via scaling laws |
| Batch Size | Predicted via scaling laws |

**Invariant:** Cross-entropy loss converges monotonically on held-out validation across all 119 languages.

**Output:** Base model checkpoint $\theta_{\text{S1}}$ with general capabilities.

---

#### Stage S2 — Reasoning Enhancement

**Objective:** Sharpen STEM, coding, and reasoning capabilities.

| Parameter | Value |
|---|---|
| Tokens | $\sim 5 \times 10^{12}$ (higher-quality subset) |
| Sequence Length | 4,096 |
| Data Composition | Increased STEM, coding, reasoning, synthetic data |
| RoPE $\theta_{\text{base}}$ | 10,000 |
| LR Decay | Accelerated schedule |

**Formal data distribution shift:**

$$p_{\text{S2}}(x) = \sum_{d \in \mathcal{G}} \alpha_d^{\text{S2}} \cdot p_d(x), \quad \alpha_{\text{STEM}}^{\text{S2}} \gg \alpha_{\text{STEM}}^{\text{S1}}, \; \alpha_{\text{code}}^{\text{S2}} \gg \alpha_{\text{code}}^{\text{S1}}$$

**Output:** Checkpoint $\theta_{\text{S2}}$ with enhanced analytical reasoning.

---

#### Stage S3 — Long-Context Extension

**Objective:** Extend effective context from 4,096 to 32,768 tokens (inference up to 131,072 via YARN + DCA).

| Parameter | Value |
|---|---|
| Tokens | Hundreds of billions |
| Sequence Length | 32,768 |
| Data Composition | 75% length ∈ [16K, 32K], 25% length ∈ [4K, 16K] |
| RoPE $\theta_{\text{base}}$ | $1{,}000{,}000$ (ABF, 100× increase) |

**ABF (Adjusted Base Frequency):**

$$\omega_i^{\text{ABF}} = \left(\theta_{\text{base}}^{\text{new}}\right)^{-2i/d_h}, \quad \theta_{\text{base}}^{\text{new}} = 1{,}000{,}000$$

This lowers the angular velocity of each rotary dimension, enabling the model to distinguish positions at longer ranges without aliasing.

**YARN (Yet Another RoPE Extension):**

Applies dimension-dependent scaling:

$$\omega_i^{\text{YARN}} = \begin{cases}
\omega_i & \text{if } \lambda_i > \lambda_{\text{max}} \\
\omega_i / s & \text{if } \lambda_i < \lambda_{\text{min}} \\
\text{interpolate} & \text{otherwise}
\end{cases}$$

where $\lambda_i = 2\pi / \omega_i$ is the wavelength, $s$ is the scaling factor, and interpolation handles the transition band.

Combined with an attention-magnitude correction factor:

$$\text{Attn}_{\text{YARN}} = \frac{1}{\sqrt{d_h}} \cdot \frac{1}{c(s)} \cdot Q K^T$$

where $c(s)$ corrects for the reduced entropy caused by position interpolation.

**DCA (Dual Chunk Attention):**

Partitions long sequences into chunks, combining intra-chunk and inter-chunk attention patterns:

$$\text{DCA}(Q, K, V) = \text{IntraChunk}(Q, K, V) + \text{InterChunk}(Q, K, V)$$

This achieves a **4× effective context extension** at inference time (32K training → 128K inference for applicable models).

**Output:** Final pre-trained checkpoint $\theta_{\text{S3}}$.

---

### 4.1.1 Pseudo-Algorithm: Pre-Training

```
ALGORITHM: Qwen3PreTraining
INPUT: Raw corpus D, model config C, scaling law predictions (η*, B*)
OUTPUT: Pre-trained checkpoint θ_S3

// ===== Stage S1: General =====
1.  θ ← RandomInit(C)
2.  η, B ← ScalingLawPredict(C, stage=S1)
3.  D_S1 ← Sample(D, tokens=30T, seq_len=4096, mixture=w*_S1)
4.  FOR step s = 1 TO S_S1:
      batch ← Sample(D_S1, batch_size=B)
      L ← CLM_Loss(θ, batch)
      IF MoE: L += λ_bal · GlobalBatchBalanceLoss(θ, batch)
      θ ← AdamW_Update(θ, ∇L, η(s))
5.  θ_S1 ← θ

// ===== Stage S2: Reasoning =====
6.  η, B ← ScalingLawPredict(C, stage=S2)
7.  D_S2 ← Sample(D, tokens=5T, seq_len=4096, mixture=w*_S2)
      // Increase STEM, code, reasoning, synthetic proportions
8.  FOR step s = 1 TO S_S2:
      batch ← Sample(D_S2, batch_size=B)
      L ← CLM_Loss(θ, batch)
      IF MoE: L += λ_bal · GlobalBatchBalanceLoss(θ, batch)
      θ ← AdamW_Update(θ, ∇L, η_accelerated(s))
9.  θ_S2 ← θ

// ===== Stage S3: Long Context =====
10. Set RoPE θ_base ← 1,000,000 (ABF)
11. Enable YARN + DCA
12. D_S3 ← LongContextCorpus(seq_len=32768)
      // 75% in [16K, 32K], 25% in [4K, 16K]
13. FOR step s = 1 TO S_S3:
      batch ← Sample(D_S3, batch_size=B_long)
      L ← CLM_Loss(θ, batch)
      θ ← AdamW_Update(θ, ∇L, η_S3(s))
14. θ_S3 ← θ
15. RETURN θ_S3
```

---

### 4.2 Post-Training: Four-Stage Process

The post-training pipeline converts the base model $\theta_{\text{S3}}$ into an instruction-following model with dual thinking/non-thinking capabilities.

---

#### 4.2.1 Stage 1: Long-CoT Cold Start

**Objective:** Instill foundational long chain-of-thought reasoning patterns without saturating the model's capacity (preserving headroom for RL).

**Data Construction:**

*Query Filtering (Phase 1):*
1. Qwen2.5-72B-Instruct removes non-verifiable queries (multiple sub-questions, open-ended generation)
2. Queries solvable without CoT by Qwen2.5-72B-Instruct are excluded
3. Domain annotation via Qwen2.5-72B-Instruct for balanced representation

*Response Filtering (Phase 2):*
- Generate $N$ candidate responses per query via QwQ-32B
- For queries where QwQ-32B consistently fails: human assessment
- For queries with positive Pass@$N$, filter responses exhibiting:
  - (a) Incorrect final answers
  - (b) Substantial repetition
  - (c) Guesswork without adequate reasoning
  - (d) Thinking/summary inconsistencies
  - (e) Inappropriate language mixing or style shifts
  - (f) Suspiciously similar to validation items

**Training Objective (SFT):**

$$\mathcal{L}_{\text{cold}} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

where $y$ includes the full thinking trace: `<think>...</think>` followed by the answer.

**Design Principle:** Minimize both the number of training samples and training steps. The goal is to seed reasoning patterns, **not** to maximize immediate performance. This preserves exploration capacity for Stage 2 RL.

---

#### 4.2.2 Stage 2: Reasoning RL

**Objective:** Maximize verifiable reasoning performance via reinforcement learning on code and mathematics tasks.

**Query-Verifier Pair Selection Criteria:**
1. Not used during cold-start (no data leakage)
2. Learnable for the cold-start model (not trivial, not impossible)
3. As challenging as possible
4. Broad sub-domain coverage

**Dataset:** 3,995 query-verifier pairs total.

**Algorithm: GRPO (Group Relative Policy Optimization)**

Given query $q$, generate $G$ rollout responses $\{y_1, \dots, y_G\}$ from current policy $\pi_\theta$:

$$y_i \sim \pi_\theta(\cdot \mid q), \quad i = 1, \dots, G$$

Compute rewards $\{r_1, \dots, r_G\}$ via the verifier, then normalize within the group:

$$\hat{r}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

where $\mu_r = \frac{1}{G}\sum_i r_i$ and $\sigma_r = \sqrt{\frac{1}{G}\sum_i (r_i - \mu_r)^2}$.

**GRPO Objective:**

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\mathbb{E}_{q \sim \mathcal{D}} \left[\frac{1}{G} \sum_{i=1}^{G} \hat{r}_i \cdot \sum_{t=1}^{|y_i|} \min\!\left(\rho_t^{(i)} \hat{A}_i, \; \text{clip}(\rho_t^{(i)}, 1-\epsilon, 1+\epsilon) \hat{A}_i\right)\right] + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

where:

$$\rho_t^{(i)} = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid q, y_{i,<t})}$$

$\hat{A}_i = \hat{r}_i$ is the group-relative advantage, and $\beta$ controls KL regularization against the reference policy $\pi_{\text{ref}}$ (cold-start checkpoint).

**Key Observations:**
- **Large batch size** and **high number of rollouts per query** are beneficial
- **Off-policy training** improves sample efficiency (reuse rollouts across updates)
- **Entropy control:** Model entropy is managed to increase steadily or remain stable, balancing exploration vs. exploitation

**Entropy Management:**

$$\mathcal{H}[\pi_\theta] = -\sum_{v \in \mathcal{V}} \pi_\theta(v \mid \cdot) \log \pi_\theta(v \mid \cdot)$$

Monotonically non-decreasing or stable entropy prevents premature collapse and maintains diverse reasoning strategies.

**Result:** AIME'24 score of Qwen3-235B-A22B increases from 70.1 → 85.1 over 170 RL steps, without manual hyperparameter intervention.

---

#### 4.2.3 Stage 3: Thinking Mode Fusion

**Objective:** Integrate non-thinking capabilities into the thinking model via continual SFT, enabling dynamic mode switching within a single model.

**Chat Template Design:**

| Mode | Template |
|---|---|
| Thinking (default) | `<\|im_start\|>user\n{query}\n/think<\|im_end\|>` → `<\|im_start\|>assistant\n<think>{thinking}</think>\n{response}<\|im_end\|>` |
| Non-thinking | `<\|im_start\|>user\n{query}\n/no_think<\|im_end\|>` → `<\|im_start\|>assistant\n<think>\n</think>\n{response}<\|im_end\|>` |

**Key design choices:**
- Non-thinking mode retains **empty `<think></think>` block** for format consistency
- Default is thinking mode (some training samples omit the `/think` flag)
- Multi-turn: random `/think` and `/no_think` flags inserted; model follows the **last flag**

**SFT Data Construction:**

*Thinking data:* Rejection sampling on Stage 1 queries using the Stage 2 model itself.

*Non-thinking data:* Curated to cover:
- Coding, mathematics, instruction-following
- Multilingual tasks, creative writing, QA, role-playing
- Low-resource language translation tasks (increased proportion)
- Quality assessed via automatically generated checklists

**SFT Loss:**

$$\mathcal{L}_{\text{fusion}} = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

applied to the combined thinking + non-thinking dataset.

**Emergent Capability — Thinking Budget:**

After Thinking Mode Fusion, the model naturally handles **partial thinking** — generating responses from incomplete reasoning. This is exploited at inference:

When thinking length exceeds user-defined threshold $B_{\text{think}}$:
1. Halt generation
2. Insert: `Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n`
3. Model generates final response conditioned on accumulated partial reasoning

This **is not explicitly trained** but emerges from the fusion of complete thinking and empty thinking modes, with the model interpolating between extremes.

---

#### 4.2.4 Stage 4: General RL

**Objective:** Broadly enhance capabilities and stability across 20+ distinct tasks.

**Task Categories and Reward Types:**

| Capability | Examples | Reward Type |
|---|---|---|
| Instruction Following | Content, format, length, structured output | Rule-based |
| Format Following | `/think` ↔ `/no_think` switching, `<think>`/`</think>` usage | Rule-based |
| Preference Alignment | Helpfulness, engagement, style | Model-based (no reference) |
| Agent Ability | Tool calling, multi-turn interaction with environment | Rule-based + Environment |
| RAG | Grounded generation, hallucination minimization | Model-based (with reference) |

**Three Reward Signal Types:**

**(1) Rule-based Reward $r_{\text{rule}}$:**

$$r_{\text{rule}}(y, y^*) = \begin{cases}
1 & \text{if } \text{Match}(y, y^*) \\
0 & \text{otherwise}
\end{cases}$$

Handles instruction following, format adherence, verifiable outputs. High precision, prevents reward hacking.

**(2) Model-based Reward with Reference $r_{\text{ref}}$:**

$$r_{\text{ref}}(y, y^*, q) = \text{Qwen2.5-72B-Instruct}(q, y, y^*)$$

Provides scalar score given query $q$, response $y$, and reference $y^*$. More flexible than rule-based, handles diverse formats, avoids false negatives.

**(3) Model-based Reward without Reference $r_{\text{pref}}$:**

$$r_{\text{pref}}(y, q) = \text{RewardModel}_\phi(q, y)$$

Trained on human preference data. Handles open-ended queries, enhances engagement and helpfulness. No dependency on reference answers.

**Combined Reward:**

$$r(y, q) = \sum_{j=1}^{J} w_j \cdot r_j(y, q)$$

where $J$ indexes the task-specific reward channels with weights $w_j$.

**Agent RL — Multi-Turn Execution:**

During rollout, the model performs complete multi-turn interaction cycles with real environment execution feedback:

$$y_1 \sim \pi_\theta(\cdot \mid q), \quad o_1 = \text{Env}(y_1), \quad y_2 \sim \pi_\theta(\cdot \mid q, y_1, o_1), \quad \dots$$

Terminal reward is computed at the end of the full interaction trajectory, improving long-horizon decision-making stability.

---

#### 4.2.5 Stage Ablation Analysis

From Table 22 ablation on Qwen3-32B:

| Metric | S2→S3 Δ | S3→S4 Δ | Interpretation |
|---|---|---|---|
| LiveBench (Think) | +2.3 | +4.0 | General capability improves across stages |
| Arena-Hard (Think) | +2.6 | +4.4 | Alignment improves substantially |
| ThinkFollow | — → 88.7 | → 98.9 | Mode switching accuracy nearly perfect after S4 |
| ToolUse | +7.1 | +15.1 | Agent capabilities dramatically improve |
| CounterFactQA | +10.9 | +6.8 | Hallucination resistance improves |
| AIME'24 (Think) | -1.9 | -0.5 | **Degradation**: general training dilutes specialized reasoning |
| LiveCodeBench (Think) | -1.2 | -1.5 | **Degradation**: same trade-off |

**Failure Mode:** Stages 3 and 4 slightly degrade peak reasoning performance on the hardest benchmarks (AIME, LiveCodeBench) while dramatically improving general versatility, format adherence, and agent capabilities. This is an **accepted Pareto trade-off**: broader capability envelope at the cost of narrow peak specialization.

---

### 4.3 Strong-to-Weak Distillation

**Scope:** 5 dense models (0.6B, 1.7B, 4B, 8B, 14B) + 1 MoE model (30B-A3B).

**Teacher Models:** Qwen3-32B or Qwen3-235B-A22B.

#### Phase 1: Off-Policy Distillation

Teacher generates responses in both `/think` and `/no_think` modes. Student is supervised on these teacher outputs:

$$\mathcal{L}_{\text{off-policy}} = -\sum_{t} \log p_{\theta_S}(y_t^{\text{teacher}} \mid x, y_{<t}^{\text{teacher}})$$

This instills:
- Basic reasoning skills
- Mode-switching capability
- Foundation for on-policy refinement

#### Phase 2: On-Policy Distillation

Student generates its own responses; teacher provides target logits:

$$\mathcal{L}_{\text{on-policy}} = D_{\text{KL}}\!\left[p_{\theta_T}(\cdot \mid x, y_{<t}^{\text{student}}) \;\|\; p_{\theta_S}(\cdot \mid x, y_{<t}^{\text{student}})\right]$$

$$= \sum_{v \in \mathcal{V}} p_{\theta_T}(v \mid x, y_{<t}^{\text{student}}) \log \frac{p_{\theta_T}(v \mid x, y_{<t}^{\text{student}})}{p_{\theta_S}(v \mid x, y_{<t}^{\text{student}})}$$

where $y^{\text{student}} \sim \pi_{\theta_S}(\cdot \mid x)$ and teacher logits are computed for student-generated continuations.

**Critical Advantage over RL (from Table 21 on Qwen3-8B):**

| Method | AIME'24 | AIME'25 | MATH-500 | LCB v5 | GPU Hours |
|---|---|---|---|---|---|
| Off-policy → RL | 67.6 (90.0) | 55.5 (83.3) | 94.8 | 52.9 | 17,920 |
| Off-policy → On-policy Distill | **74.4** (**93.3**) | **65.5** (**86.7**) | **97.0** | **60.3** | **1,800** |

**Key observations:**
- On-policy distillation achieves **~10× compute efficiency** (1,800 vs 17,920 GPU hours)
- **Higher Pass@1** across all benchmarks
- **Higher Pass@64** (93.3 vs 90.0 on AIME'24), indicating **expanded exploration space** — the student discovers reasoning paths beyond those found by RL alone
- RL shows **no improvement in Pass@64**, suggesting it exploits existing capabilities without expanding the reasoning frontier

**Information-Theoretic Interpretation:**

Distillation transfers the teacher's **full predictive distribution** $p_{\theta_T}$, not just argmax labels. The student receives gradient signal from the entire vocabulary at every position, providing:

$$\text{Effective gradient info per token} \propto |\mathcal{V}| \quad \text{(distillation)} \quad \text{vs.} \quad 1 \quad \text{(RL reward)}$$

This explains both the efficiency and the exploration benefits.

---

### 4.3.1 Pseudo-Algorithm: Full Post-Training

```
ALGORITHM: Qwen3PostTraining
INPUT: Pre-trained checkpoint θ_S3, teacher model θ_T
OUTPUT: Instruction-tuned model θ_final

// ===== FLAGSHIP MODELS (235B, 32B) =====

// Stage 1: Long-CoT Cold Start
1.  D_cold ← QueryFilter(D_raw, Qwen2.5-72B-Instruct)
      Filter: non-verifiable, solvable-without-CoT, domain-balance
2.  FOR each query q ∈ D_cold:
      responses ← Generate_N(QwQ-32B, q)
      IF Pass@N = 0: human_assess(responses)
      responses ← ResponseFilter(responses)
        // Remove: wrong answer, repetition, guesswork,
        // inconsistency, language mixing, validation similarity
3.  D_cold_final ← Select_Subset(D_cold, minimize_samples=True)
4.  θ_S1 ← SFT(θ_S3, D_cold_final, few_steps=True)

// Stage 2: Reasoning RL
5.  D_RL ← Select_QueryVerifier_Pairs(criteria=[
        not_in_cold_start, learnable, challenging, broad_coverage
      ], count=3995)
6.  θ_ref ← θ_S1
7.  FOR step s = 1 TO 170:
      FOR each q ∈ batch:
        rollouts ← Generate_G(π_{θ}, q)  // large G
        rewards ← Verify(rollouts)
        advantages ← GroupNormalize(rewards)
      θ ← GRPO_Update(θ, rollouts, advantages, θ_ref, β)
      Monitor: entropy(π_θ) should be steady/increasing
8.  θ_S2 ← θ

// Stage 3: Thinking Mode Fusion
9.  D_think ← RejectionSample(θ_S2, D_cold_queries)
10. D_nothink ← Curate(code, math, IF, multilingual, writing, QA, RP)
      Apply quality checklists
      Increase low-resource translation proportion
11. D_fusion ← Mix(D_think, D_nothink, with_chat_template)
      Insert /think and /no_think flags
      Multi-turn: random flag insertion, follow last flag
12. θ_S3_post ← ContinualSFT(θ_S2, D_fusion)

// Stage 4: General RL
13. Setup reward system: 20+ tasks with
      r_rule, r_ref(Qwen2.5-72B), r_pref(RewardModel)
14. FOR step s = 1 TO S_general:
      FOR each task:
        q ← SampleQuery(task)
        y ← Rollout(π_θ, q, multi_turn_if_agent=True)
        r ← CombinedReward(y, q, task)
      θ ← RL_Update(θ, rollouts, rewards)
15. θ_final_flagship ← θ

// ===== LIGHTWEIGHT MODELS (0.6B–14B, 30B-A3B) =====

// Off-Policy Distillation
16. D_off ← Generate(θ_T, prompts, modes=[/think, /no_think])
17. θ_student ← SFT(θ_S3_student, D_off)

// On-Policy Distillation
18. FOR step s = 1 TO S_distill:
      y_student ← Generate(π_{θ_student}, prompts, mode=random)
      logits_teacher ← θ_T.forward(prompts, y_student)
      L ← KL(logits_teacher || logits_student)
      θ_student ← Update(θ_student, ∇L)
19. θ_final_lightweight ← θ_student

20. RETURN θ_final_flagship, θ_final_lightweight
```

---

## 5. Inference Path

### 5.1 Autoregressive Decoding with Mode Control

**Mode Selection:**

At inference time, the system prompt or user message contains either `/think` or `/no_think`. The model tokenizes this flag and conditions generation accordingly.

**Thinking Mode — Standard Decoding:**

$$y_t \sim \text{TopK}\!\left(\text{TopP}\!\left(\text{softmax}\!\left(\frac{z_t}{\tau}\right), p\right), k\right)$$

with $\tau = 0.6$, $p = 0.95$, $k = 20$ (thinking mode defaults).

**Non-Thinking Mode:**

$$\tau = 0.7, \quad p = 0.8, \quad k = 20, \quad \text{presence\_penalty} = 1.5$$

**Max Output Length:** 32,768 tokens (38,912 for AIME tasks).

---

### 5.2 Thinking Budget Mechanism

**Formal Definition:** Given budget $B_{\text{think}} \in \mathbb{Z}^+$:

$$\text{Generate}(q, B_{\text{think}}) = \begin{cases}
\text{standard thinking} & \text{if } |\text{thinking\_tokens}| < B_{\text{think}} \\
\text{inject stop-thinking + generate response} & \text{if } |\text{thinking\_tokens}| \geq B_{\text{think}}
\end{cases}$$

**Stop-thinking injection string:**

`Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n`

**Property:** Performance scales smoothly and monotonically with $B_{\text{think}}$ (Figure 2 in original report), confirming that the model extracts value from partial reasoning chains.

**Complexity Trade-off:**

$$\text{Latency}(B_{\text{think}}) \approx B_{\text{think}} \cdot t_{\text{decode}} + t_{\text{prefill}}$$

Users can tune the latency–quality Pareto frontier per-query.

---

### 5.3 KV-Cache Management

**Per-Layer KV-Cache Size:**

$$\text{KV}^{(l)}_{\text{bytes}} = 2 \times n_{KV} \times d_h \times T_{\text{current}} \times b$$

where $b = 2$ bytes (FP16) or $b = 1$ byte (INT8 quantized KV-cache).

**Total KV-Cache:**

$$\text{KV}_{\text{total}} = \sum_{l=1}^{L} \text{KV}^{(l)}_{\text{bytes}}$$

For Qwen3-235B-A22B at $T = 32{,}768$, $n_{KV} = 4$, $L = 94$:

$$\text{KV}_{\text{total}} = 94 \times 2 \times 4 \times d_h \times 32768 \times 2 \;\text{bytes}$$

GQA's $n_{KV} = 4$ (vs. $n_Q = 64$) provides **16× KV-cache compression**.

---

### 5.4 Long-Context Inference

**Effective Context at Inference:**

- Training context: 32,768 tokens
- Inference extension via YARN + DCA: up to **131,072 tokens** (4× extension)

**YARN + DCA Combined Effect:**

$$T_{\text{effective}} = s \times T_{\text{train}} = 4 \times 32{,}768 = 131{,}072$$

where $s = 4$ is the extension factor achieved through:
1. YARN frequency rescaling of RoPE dimensions
2. DCA chunk-based attention that separates intra-chunk and inter-chunk computations

---

### 5.5 MoE Inference Considerations

**Expert Routing at Inference:**

For each token, only $k = 8$ of $E = 128$ experts are activated:

$$\text{Active FLOPs} = \frac{k}{E} \times \text{Total Expert FLOPs} = \frac{8}{128} = \frac{1}{16}$$

However, **all 128 expert weights must be accessible** in memory (or via paged/offloaded storage).

**Expert Parallelism:**

Experts are distributed across devices. All-to-all communication routes tokens to their assigned expert devices:

$$\text{Communication Cost} \propto k \times B \times d \times \text{num\_devices}$$

where $B$ is the batch size. This is the dominant latency bottleneck in MoE inference at high batch sizes.

---

### 5.6 Pseudo-Algorithm: Inference

```
ALGORITHM: Qwen3Inference
INPUT: Query q, mode ∈ {think, no_think}, budget B_think, params (τ, p, k)
OUTPUT: Response y

1.  tokens ← Tokenize(q + mode_flag)
2.  KV_cache ← Initialize()

3.  // Prefill phase
    logits, KV_cache ← ForwardPass(tokens, KV_cache)

4.  thinking_tokens ← 0
    output ← []
    IF mode = think:
      output.append("<think>")

5.  WHILE NOT EOS AND len(output) < max_len:
      next_token ← Sample(logits, τ, p, k, presence_penalty)
      output.append(next_token)
      
      IF mode = think AND in_thinking_block:
        thinking_tokens += 1
        IF thinking_tokens >= B_think:
          // Inject stop-thinking
          stop_str ← "Considering the limited time..."
          output.extend(Tokenize(stop_str))
          in_thinking_block ← False
      
      IF next_token = "</think>":
        in_thinking_block ← False
      
      // Decode step
      logits, KV_cache ← ForwardPass([next_token], KV_cache)

6.  y ← Detokenize(output)
7.  RETURN y
```

---

## 6. Evaluation Protocol

### 6.1 Benchmark Taxonomy

#### 6.1.1 General Tasks

| Benchmark | Setting | Metric |
|---|---|---|
| MMLU-Redux | 5-shot | Accuracy |
| GPQA-Diamond | 10× sampling, average | Accuracy |
| C-Eval | — | Accuracy |
| LiveBench (2024-11-25) | — | Composite |

#### 6.1.2 Alignment Tasks

| Benchmark | Metric |
|---|---|
| IFEval | Strict-prompt accuracy |
| Arena-Hard | Win rate |
| AlignBench v1.1 | Score (1–10) |
| Creative Writing v3 | Score |
| WritingBench | Score (1–10) |

#### 6.1.3 Math & Text Reasoning

| Benchmark | Setting | Metric |
|---|---|---|
| MATH-500 | — | Accuracy |
| AIME'24 / AIME'25 | 64× sampling, average | Accuracy |
| ZebraLogic | — | Accuracy |
| AutoLogi | — | Accuracy |

#### 6.1.4 Agent & Coding

| Benchmark | Setting | Metric |
|---|---|---|
| BFCL v3 | FC format, 64K context via YARN | Accuracy |
| LiveCodeBench v5 | Adjusted prompt for thinking mode | Pass rate |
| CodeForces | 8 attempts per problem | Elo rating |

#### 6.1.5 Multilingual Tasks (119 languages, 6 benchmarks)

| Benchmark | # Languages | Task |
|---|---|---|
| Multi-IF | 8 | Instruction following |
| INCLUDE | 44 | Regional knowledge |
| MMMLU | 14 | General knowledge |
| MT-AIME2024 | 55 | Mathematics |
| PolyMath | 18 | Mathematics |
| MLogiQA | 10 | Logical reasoning |

---

### 6.2 Pre-Training Evaluation (Base Models)

**15 benchmarks** across general, math & STEM, coding, and multilingual dimensions.

**Key Quantitative Results (Qwen3-235B-A22B-Base):**

$$\text{vs. DeepSeek-V3-Base:} \quad 14/15 \text{ benchmarks superior, with } 1/3 \text{ total params, } 2/3 \text{ activated params}$$

$$\text{vs. Qwen2.5-72B-Base:} \quad 15/15 \text{ benchmarks superior, with } < 1/3 \text{ activated params}$$

$$\text{vs. Llama-4-Maverick-Base:} \quad \text{Superior on most benchmarks with } \sim 1/2 \text{ total params}$$

**MoE Efficiency Results:**

$$\text{Qwen3 MoE} \approx \text{Qwen3 Dense (same data)} \text{ with only } 1/5 \text{ activated params}$$

$$\text{Qwen3 MoE} > \text{Qwen2.5 MoE} \text{ with } < 1/2 \text{ activated params and fewer total params}$$

$$\text{Qwen3 MoE} \approx \text{Qwen2.5 Dense} \text{ with } 1/10 \text{ activated params}$$

**Cross-Scale Equivalence:**

$$\text{Qwen3-1.7B} \approx \text{Qwen2.5-3B}, \quad \text{Qwen3-4B} \approx \text{Qwen2.5-7B}$$
$$\text{Qwen3-8B} \approx \text{Qwen2.5-14B}, \quad \text{Qwen3-14B} \approx \text{Qwen2.5-32B}$$
$$\text{Qwen3-32B} \approx \text{Qwen2.5-72B}$$

This demonstrates approximately **2× parameter efficiency improvement** per generation.

---

### 6.3 Post-Training Evaluation (Instruction-Tuned Models)

**Thinking Mode Sampling:** $\tau = 0.6, \; p = 0.95, \; k = 20$

**Non-Thinking Mode Sampling:** $\tau = 0.7, \; p = 0.8, \; k = 20, \; \text{presence\_penalty} = 1.5$

**Creative Writing / WritingBench (both modes):** $\text{presence\_penalty} = 1.5$

**Output length:** 32,768 tokens (38,912 for AIME)

**Flagship Results (Qwen3-235B-A22B Thinking):**

| Benchmark | Score |
|---|---|
| AIME'24 | **85.7** |
| AIME'25 | **81.5** |
| LiveCodeBench v5 | **70.7** |
| CodeForces | **2056** (98.2nd percentile) |
| BFCL v3 | **70.8** |
| Arena-Hard | **95.6** |

**vs. DeepSeek-R1:** Superior on 17/23 benchmarks with 60% activated params, 35% total params.

---

### 6.4 Thinking Budget Scaling Analysis

From Figure 2, Qwen3-235B-A22B exhibits **monotonically increasing performance** as thinking budget increases, across mathematics, coding, and STEM benchmarks. The scaling is **smooth** (no discontinuities), confirming:

1. The model extracts marginal value from each additional thinking token
2. Partial reasoning chains contain actionable intermediate conclusions
3. The thinking budget mechanism enables **continuous latency-quality trade-off** without model switching

**Conjecture:** Further extension beyond 32K output tokens would yield additional performance gains (stated as future work).

---

## 7. Deployment Constraints

### 7.1 Memory Requirements

**Model Weight Storage (FP16):**

| Model | Total Params | FP16 Size | Activated Params | Active FP16 |
|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | ~1.2 GB | 0.6B | 1.2 GB |
| Qwen3-8B | 8B | ~16 GB | 8B | 16 GB |
| Qwen3-32B | 32B | ~64 GB | 32B | 64 GB |
| Qwen3-30B-A3B | 30B | ~60 GB | 3B | ~6 GB (compute) |
| Qwen3-235B-A22B | 235B | ~470 GB | 22B | ~44 GB (compute) |

Note: MoE models require **full parameter storage** despite sparse activation.

---

### 7.2 KV-Cache Memory at Inference

For sequence length $T$ and batch size $B$:

$$\text{KV}_{\text{total}} = B \times L \times 2 \times n_{KV} \times d_h \times T \times b_{\text{bytes}}$$

**Critical constraint for long-context:** At $T = 128$K with Qwen3-235B-A22B ($L = 94, n_{KV} = 4$), KV-cache can dominate GPU memory, necessitating:
- Paged KV-cache (vLLM-style)
- KV-cache quantization (FP8 or INT8)
- Prefix caching for repeated prompts

---

### 7.3 Throughput and Latency Considerations

**Prefill throughput:** Compute-bound, scales with FLOPs:

$$\text{Prefill FLOPs} \approx 2 \times N_{\text{active}} \times T_{\text{prompt}}$$

For Qwen3-235B-A22B: $2 \times 22 \times 10^9 \times T_{\text{prompt}}$.

**Decode throughput:** Memory-bandwidth-bound:

$$\text{Decode tokens/s} \approx \frac{\text{Memory Bandwidth (GB/s)}}{N_{\text{active}} \times b_{\text{bytes}} + \text{KV\_read\_per\_step}}$$

**MoE-specific bottleneck:** All-to-all expert routing communication adds latency proportional to the number of active experts and the expert parallelism degree.

---

### 7.4 Distributed Serving

**Tensor Parallelism (TP):** Split attention heads and FFN columns across devices. For GQA with $n_{KV} = 4$, minimum TP degree constrained by $n_{KV}$: TP ≤ 4 for KV-head splitting, or replicate KV heads across TP ranks.

**Expert Parallelism (EP):** 128 experts distributed across $D_{\text{EP}}$ devices, with all-to-all dispatch:

$$\text{Experts per device} = \frac{128}{D_{\text{EP}}}$$

**Pipeline Parallelism (PP):** 94 layers (Qwen3-235B-A22B) partitioned across pipeline stages.

---

### 7.5 Quantization Considerations

For edge deployment (0.6B–4B models):
- INT4/INT8 weight-only quantization
- GPTQ / AWQ / GGUF formats
- Tied embedding models (0.6B, 1.7B, 4B) benefit from embedding table compression

For server deployment (32B, 235B):
- FP8 weight quantization (1.5–2× throughput improvement on H100)
- INT8 KV-cache quantization
- Quantization-aware training or post-training calibration

---

### 7.6 Failure Modes and Mitigations

| Failure Mode | Manifestation | Mitigation in Qwen3 |
|---|---|---|
| **Reward Hacking** | RL exploits verifier weaknesses | Well-designed rule-based rewards; diverse reward signals |
| **Mode Confusion** | Thinking when `/no_think`, or vice versa | ThinkFollow benchmark; Stage 4 RL drives ThinkFollow to 98.9% |
| **Reasoning Degradation** | General RL dilutes specialized reasoning | Accepted Pareto trade-off; AIME drops ~2 pts for broad capability gains |
| **Hallucination** | Fabricated facts in RAG/QA | CounterFactQA monitoring; RAG-specific reward signals |
| **Expert Collapse (MoE)** | Few experts receive all tokens | Global-batch load balancing loss |
| **Entropy Collapse (RL)** | Policy becomes deterministic too early | Entropy monitoring; controlled steady/increasing entropy |
| **Language Mixing** | Inappropriate code-switching in output | Explicit filtering in cold-start data; multilingual SFT balance |
| **Thinking Budget Artifacts** | Incoherent response after forced thinking halt | Emergent from fusion training; smooth degradation empirically observed |
| **KV-Cache Overflow** | OOM at long contexts | GQA compression (16×); paged KV-cache; quantized KV |
| **Distribution Shift (Distillation)** | Student distribution diverges from teacher | On-policy distillation corrects shift by training on student-generated sequences |

---

## 8. Convergence Dynamics and Training Stability

### 8.1 QK-Norm Stability Guarantee

**Without QK-Norm:** Attention logit variance scales with training:

$$\text{Var}[q^T k] \propto d_h \cdot \text{Var}[q_i] \cdot \text{Var}[k_i] \cdot (1 + \kappa_t)$$

where $\kappa_t$ grows with training step due to weight norm increase. This leads to softmax saturation:

$$\text{softmax}(\alpha z) \xrightarrow{\alpha \to \infty} \text{one-hot}$$

causing gradient vanishing through attention layers.

**With QK-Norm:**

$$\text{Var}[q^T k] = O(d_h)$$

bounded by the learnable scale $\gamma$, preventing logit explosion regardless of training duration.

### 8.2 GRPO Convergence Properties

**Monotonic Improvement Guarantee (empirical):**

Over 170 RL steps on Qwen3-235B-A22B:
- AIME'24: $70.1 \to 85.1$ (monotonic)
- No hyperparameter intervention required
- Entropy remains stable/increasing throughout

This indicates that the combination of:
1. Large batch size (low gradient variance)
2. High rollouts per query (accurate advantage estimation)
3. Off-policy reuse (sample efficiency)
4. Entropy control (exploration maintenance)

produces stable, intervention-free RL training at the 235B scale.

### 8.3 Distillation Convergence

**On-policy KL minimization convergence:**

$$D_{\text{KL}}[p_T \| p_S] \geq 0, \quad \nabla_{\theta_S} D_{\text{KL}} = -\mathbb{E}_{p_T}\!\left[\nabla_{\theta_S} \log p_{\theta_S}\right]$$

Gradient is well-defined and the objective is convex in $\log p_{\theta_S}$ (log-linear models), ensuring convergence to a local minimum of the KL surface. On-policy generation ensures the student is trained on its own distribution, preventing compounding errors from distribution mismatch.

---

## 9. Summary of Architectural and Training Innovations

| Innovation | Predecessor (Qwen2.5) | Qwen3 |
|---|---|---|
| QKV-bias | Present | **Removed** |
| QK-Norm | Absent | **Introduced** (training stability) |
| Shared experts (MoE) | Present | **Removed** (full specialization) |
| Load balancing | Per-sample/micro-batch | **Global-batch** (Qiu et al., 2025) |
| Languages | 29 | **119** |
| Pre-training tokens | ~18T | **36T** |
| Context length | 32K/128K | **32K train → 128K inference** |
| Post-training | SFT + RLHF | **4-stage: CoT cold-start → Reasoning RL → Thinking Fusion → General RL** |
| Mode control | Separate models | **Unified `/think` `/no_think`** |
| Thinking budget | N/A | **Emergent from fusion** |
| Lightweight training | Independent | **Strong-to-weak distillation (10× efficiency)** |
| Total expert count | Varies | **128 (fine-grained segmentation)** |
| Active experts | Varies | **8** |

---

## 10. Information Preservation Equations

### 10.1 Pre-Training Information Flow

**Mutual information between data and model:**

$$I(\mathcal{D}; \theta) = H(\mathcal{D}) - H(\mathcal{D} \mid \theta) = H(\mathcal{D}) + \mathbb{E}_{\mathcal{D}}\!\left[\log p_\theta(\mathcal{D})\right]$$

Pre-training maximizes $I(\mathcal{D}; \theta)$ by minimizing $\mathcal{L}_{\text{CLM}} = -\mathbb{E}[\log p_\theta]$.

### 10.2 Distillation Information Transfer

**Data Processing Inequality applied to distillation chain:**

$$I(X; \theta_T) \geq I(X; \theta_S)$$

Equality holds when the student perfectly replicates the teacher's distribution. The KL-based on-policy distillation minimizes the gap:

$$I(X; \theta_T) - I(X; \theta_S) = \mathbb{E}_X\!\left[D_{\text{KL}}[p_{\theta_T}(\cdot \mid X) \| p_{\theta_S}(\cdot \mid X)]\right]$$

### 10.3 Compression Efficiency (MoE)

**Information per FLOP:**

$$\eta_{\text{info}} = \frac{\text{Bits of mutual information stored}}{N_{\text{active}} \times \text{FLOPs per forward}}$$

Qwen3-235B-A22B achieves approximately **10.7×** higher parameter capacity per active FLOP compared to a dense 235B model, while maintaining competitive or superior information capture (as evidenced by benchmark parity or superiority versus dense baselines).

### 10.4 GQA KV-Cache Compression

**Compression without information loss (within the attention mechanism):**

The GQA shared KV representation satisfies:

$$\text{Attn}(Q_g, K_j, V_j) = \text{Attn}(Q_g, K_g, V_g) \quad \text{when} \quad K_j = K_g, \; V_j = V_g \; \forall g \in \text{group}(j)$$

The model learns to encode group-shared key-value representations during training, so the compression introduces **no post-hoc information loss** — the model is trained to operate with the compressed KV structure from initialization.

**Compression ratio:** $n_{KV}/n_Q$. For Qwen3-235B-A22B: $4/64 = 1/16$.