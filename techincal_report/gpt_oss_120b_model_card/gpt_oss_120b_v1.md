

# GPT-OSS-120B / GPT-OSS-20B: End-to-End Technical Report

---

## 1. Data Pipeline

### 1.1 Formal Definition

**Objective:** Construct a text-only pretraining corpus $\mathcal{D} = \{x^{(i)}\}_{i=1}^{N}$ with $|\mathcal{D}|$ in the trillions of tokens, domain-weighted toward STEM, coding, and general knowledge, with provable removal of hazardous biosecurity content.

**Inputs:** Raw web-scale text corpora, curated STEM/code datasets, domain-specific knowledge sources.

**Outputs:** Deduplicated, filtered, tokenized sequences of maximum length $L_{\max} = 131{,}072$ tokens, packed into training batches.

**Invariants:**
- Knowledge cutoff: June 2024.
- CBRN content is filtered using the same pre-training filters as GPT-4o.
- No image, audio, or video modalities—text-only throughout all stages.

### 1.2 Data Curation and Filtering

#### 1.2.1 Domain Weighting

Let $\mathcal{D} = \bigcup_{k=1}^{K} \mathcal{D}_k$ where $\mathcal{D}_k$ denotes domain $k \in \{\text{STEM}, \text{code}, \text{general}, \ldots\}$. The sampling probability for domain $k$ is:

$$p_k = \frac{w_k \cdot |\mathcal{D}_k|}{\sum_{j=1}^{K} w_j \cdot |\mathcal{D}_j|}$$

where $w_k > 0$ are domain weights tuned to emphasize STEM, coding, and general knowledge.

#### 1.2.2 CBRN Safety Filtering

A classifier $f_{\text{CBRN}}: \mathcal{X} \to [0,1]$ scores each document $x$ for Chemical, Biological, Radiological, and Nuclear hazard relevance:

$$\mathcal{D}_{\text{filtered}} = \{x \in \mathcal{D} : f_{\text{CBRN}}(x) < \tau_{\text{CBRN}}\}$$

- The same filter pipeline from GPT-4o is reused, ensuring consistency across model families.
- **Failure mode:** Overly aggressive filtering removes benign scientific content; under-filtering leaks hazardous knowledge into the model's parametric memory.

#### 1.2.3 Deduplication

Standard MinHash + LSH deduplication at the document level, followed by exact-match n-gram deduplication at the paragraph level, removing near-duplicate content to prevent memorization and training instability.

### 1.3 Tokenization

**Tokenizer:** `o200k_harmony` — a Byte Pair Encoding (BPE) tokenizer, open-sourced via the TikToken library.

**Formal specification:**
- Vocabulary size: $V = 201{,}088$ tokens
- Extension of the base `o200k` tokenizer (used in GPT-4o, o4-mini) with special tokens for the harmony chat format
- Special tokens delineate message boundaries, author roles (`System`, `Developer`, `User`, `Assistant`, `Tool`), and visibility channels (`analysis`, `commentary`, `final`)

**BPE Objective:**

Given a corpus $\mathcal{C}$, BPE iteratively merges the most frequent adjacent byte-pair $(a, b)$ to form a new token $ab$, maximizing compression:

$$\text{merge}^* = \arg\max_{(a,b)} \text{freq}(a, b \mid \mathcal{C})$$

repeated until $|V| = 201{,}088$.

**Encoding complexity:** $O(n \cdot |V|)$ per document in the worst case, $O(n)$ amortized with trie-based lookup, where $n$ is byte length.

### 1.4 Preprocessing Pseudo-Algorithm

```
ALGORITHM: DataPipeline
Input: Raw corpus R, CBRN classifier f, threshold τ, tokenizer T
Output: Training dataset D_train

1. D ← Deduplicate(R)                     // MinHash + LSH + n-gram
2. D ← {x ∈ D : f_CBRN(x) < τ}           // CBRN safety filtering
3. D ← DomainWeight(D, {w_k})             // Apply domain sampling weights
4. FOR each document x ∈ D:
     tokens ← T.encode(x)                  // BPE tokenization
     IF len(tokens) > L_max:
       chunks ← SplitWithOverlap(tokens, L_max, overlap=0)
     ELSE:
       chunks ← [tokens]
     APPEND chunks → D_tokenized
5. D_train ← PackSequences(D_tokenized, L_max)  // Concatenate + pack
6. RETURN D_train
```

**Failure modes:**
- Tokenization artifacts at document boundaries inside packed sequences
- Domain weight miscalibration leading to over-representation of narrow topics
- Contamination of evaluation benchmarks in training data

---

## 2. Model Architecture

### 2.1 Formal Definition

GPT-OSS is an autoregressive Mixture-of-Experts (MoE) Transformer, descended from GPT-2/GPT-3 architecture families. The model defines a conditional probability distribution over token sequences:

$$p_\theta(x_{1:T}) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})$$

where each conditional is parameterized by a deep MoE Transformer with parameters $\theta$.

**Two model sizes:**

| Specification | GPT-OSS-120B | GPT-OSS-20B |
|---|---|---|
| Layers $L$ | 36 | 24 |
| Residual stream dim $d$ | 2880 | 2880 |
| Experts per layer $N_e$ | 128 | 32 |
| Top-$k$ routing | 4 | 4 |
| Query heads $n_q$ | 64 | 64 |
| Head dim $d_h$ | 64 | 64 |
| KV heads $n_{kv}$ (GQA) | 8 | 8 |
| Vocabulary $V$ | 201,088 | 201,088 |
| Expert hidden dim $d_{ff}$ | 2880 | 2880 |
| Total params | 116.83B | 20.91B |
| Active params/token | 5.13B | 3.61B |
| Window bandwidth $w$ | 128 | 128 |
| Max context $L_{\max}$ | 131,072 | 131,072 |

### 2.2 Embedding Layer

#### 2.2.1 Token Embedding

$$\mathbf{h}_t^{(0)} = \mathbf{E}[x_t] \in \mathbb{R}^d$$

where $\mathbf{E} \in \mathbb{R}^{V \times d}$ is the embedding matrix.

**Parameter count:**

$$|\mathbf{E}| = V \times d = 201{,}088 \times 2{,}880 = 579{,}133{,}440 \approx 0.579\text{B}$$

#### 2.2.2 Unembedding

The output logits are computed via a separate unembedding matrix $\mathbf{U} \in \mathbb{R}^{d \times V}$:

$$\text{logits}_t = \mathbf{U}^T \, \text{RMSNorm}(\mathbf{h}_t^{(L)}) \in \mathbb{R}^V$$

$$|\mathbf{U}| = d \times V = 0.579\text{B}$$

**Total embedding + unembedding:** $1.158\text{B} \approx 1.16\text{B}$ ✓

**Note:** Unembedding parameters count toward active parameters; embedding parameters do not. Hence:

$$\text{Active}_{\text{embed}} = |\mathbf{U}| = 0.579\text{B}$$

### 2.3 Transformer Block

Each layer $\ell \in \{1, \ldots, L\}$ applies the following residual stream update:

$$\mathbf{h}^{(\ell)} = \mathbf{h}^{(\ell-1)} + \text{Attn}^{(\ell)}\!\left(\text{RMSNorm}(\mathbf{h}^{(\ell-1)})\right) + \text{MoE}^{(\ell)}\!\left(\text{RMSNorm}\!\left(\mathbf{h}^{(\ell-1)} + \text{Attn}^{(\ell)}(\cdot)\right)\right)$$

This is **Pre-LN** placement (GPT-2 style): normalization is applied *before* each sub-layer, with residual connections bypassing normalization.

#### 2.3.1 RMSNorm

**Definition:**

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \boldsymbol{\gamma}$$

where:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}$$

- $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learnable gain vector
- $\epsilon > 0$ is a numerical stability constant (typically $10^{-6}$)
- No learnable bias; no mean centering (unlike LayerNorm)

**Complexity:** $O(d)$ per token per normalization.

**Invariant:** $\|\text{RMSNorm}(\mathbf{x})\|_2 \approx \sqrt{d} \cdot \|\boldsymbol{\gamma}\|_{\text{rms}}$, stabilizing residual stream magnitudes across depth.

### 2.4 Attention Mechanism

#### 2.4.1 Grouped Query Attention (GQA)

**Configuration:** $n_q = 64$ query heads, $n_{kv} = 8$ key-value heads, head dimension $d_h = 64$. Each KV head is shared across $n_q / n_{kv} = 8$ query heads.

**Projections:**

$$\mathbf{Q} = \mathbf{x} \mathbf{W}_Q, \quad \mathbf{W}_Q \in \mathbb{R}^{d \times (n_q \cdot d_h)} = \mathbb{R}^{2880 \times 4096}$$

$$\mathbf{K} = \mathbf{x} \mathbf{W}_K, \quad \mathbf{W}_K \in \mathbb{R}^{d \times (n_{kv} \cdot d_h)} = \mathbb{R}^{2880 \times 512}$$

$$\mathbf{V} = \mathbf{x} \mathbf{W}_V, \quad \mathbf{W}_V \in \mathbb{R}^{d \times (n_{kv} \cdot d_h)} = \mathbb{R}^{2880 \times 512}$$

$$\mathbf{O} = \text{Concat}(\text{head}_1, \ldots, \text{head}_{n_q}) \, \mathbf{W}_O, \quad \mathbf{W}_O \in \mathbb{R}^{(n_q \cdot d_h) \times d} = \mathbb{R}^{4096 \times 2880}$$

**Parameter count per layer:**

$$|\text{Attn}^{(\ell)}| = d \cdot n_q \cdot d_h + 2 \cdot d \cdot n_{kv} \cdot d_h + n_q \cdot d_h \cdot d$$

$$= 2880 \times 4096 + 2 \times 2880 \times 512 + 4096 \times 2880$$

$$= 11{,}796{,}480 + 2{,}949{,}120 + 11{,}796{,}480 = 26{,}542{,}080 \approx 26.54\text{M}$$

**Total attention parameters:**
- 120B: $36 \times 26.54\text{M} = 955.4\text{M} \approx 0.96\text{B}$ ✓
- 20B: $24 \times 26.54\text{M} = 637.0\text{M} \approx 0.64\text{B}$ ✓

#### 2.4.2 Attention Computation with Learned Softmax Bias

For query head $i$ at position $t$, the attention weights are computed with a **learned bias in the softmax denominator** (off-by-one / attention-sink mechanism):

$$\alpha_{t,s}^{(i)} = \frac{\exp\!\left(\frac{\mathbf{q}_t^{(i)} \cdot \mathbf{k}_s^{(i)}}{\sqrt{d_h}}\right)}{\exp(b^{(i)}) + \sum_{s' \in \mathcal{S}_t} \exp\!\left(\frac{\mathbf{q}_t^{(i)} \cdot \mathbf{k}_{s'}^{(i)}}{\sqrt{d_h}}\right)}$$

where:
- $b^{(i)} \in \mathbb{R}$ is a **learned scalar bias per head** $i$ (total: $n_q = 64$ learned biases per layer)
- $\mathcal{S}_t$ is the set of positions attended to by position $t$ (determined by the attention pattern)

**Effective no-attention mechanism:** When $\exp(b^{(i)}) \gg \sum_{s'} \exp(\cdot)$, all attention weights $\alpha_{t,s}^{(i)} \to 0$, allowing the head to effectively attend to nothing. The "absorbed" attention mass produces a zero-weighted output contribution, enabling selective information routing.

**Residual attention output:**

$$\mathbf{o}_t^{(i)} = \sum_{s \in \mathcal{S}_t} \alpha_{t,s}^{(i)} \, \mathbf{v}_s^{(i)}$$

Note: $\sum_{s} \alpha_{t,s}^{(i)} \leq 1$ (strict inequality when $b^{(i)}$ captures mass), unlike standard softmax attention where $\sum_s \alpha_{t,s} = 1$.

#### 2.4.3 Alternating Attention Patterns

Layers alternate between two attention patterns:

- **Banded (windowed) attention:** $\mathcal{S}_t = \{s : |t - s| \leq w/2\} \cap \{1, \ldots, t\}$ with bandwidth $w = 128$
- **Dense (full causal) attention:** $\mathcal{S}_t = \{1, \ldots, t\}$

**Pattern assignment:**

$$\text{Pattern}(\ell) = \begin{cases} \text{banded} & \text{if } \ell \bmod 2 = 1 \\ \text{dense} & \text{if } \ell \bmod 2 = 0 \end{cases}$$

**Complexity per layer per token:**
- Banded: $O(w \cdot d_h \cdot n_q) = O(128 \times 64 \times 64) = O(524{,}288)$
- Dense: $O(T \cdot d_h \cdot n_q) = O(T \times 4{,}096)$

**Rationale:** Banded layers capture local patterns at $O(w)$ cost; dense layers capture global dependencies. Alternation provides effective receptive field coverage across the full context while halving the quadratic attention compute.

#### 2.4.4 Rotary Position Embeddings (RoPE) with YaRN Extension

**Base RoPE:**

For position $t$ and head dimension pair index $m \in \{0, \ldots, d_h/2 - 1\}$:

$$\theta_m = \beta^{-2m/d_h}$$

where $\beta = 10{,}000$ (base frequency). The rotation matrix applied to query/key pairs:

$$\mathbf{R}_t = \bigoplus_{m=0}^{d_h/2 - 1} \begin{pmatrix} \cos(t \cdot \theta_m) & -\sin(t \cdot \theta_m) \\ \sin(t \cdot \theta_m) & \cos(t \cdot \theta_m) \end{pmatrix}$$

$$\tilde{\mathbf{q}}_t = \mathbf{R}_t \, \mathbf{q}_t, \quad \tilde{\mathbf{k}}_s = \mathbf{R}_s \, \mathbf{k}_s$$

$$\tilde{\mathbf{q}}_t^T \tilde{\mathbf{k}}_s = \mathbf{q}_t^T \mathbf{R}_{t-s} \, \mathbf{k}_s$$

This encodes relative position $t - s$ into the dot-product attention.

**YaRN Extension to $L_{\max} = 131{,}072$:**

YaRN (Yet another RoPE extensioN) modifies RoPE frequencies to extrapolate beyond the pretraining context. Let $\lambda_m = 2\pi / \theta_m$ be the wavelength of frequency $m$, and $s = L_{\max} / L_{\text{pretrain}}$ be the scale factor. Define:

$$\theta_m' = \begin{cases} \theta_m & \text{if } \lambda_m > \lambda_{\text{long}} \\ \theta_m / s & \text{if } \lambda_m < \lambda_{\text{short}} \\ \left[(1 - \gamma_m) \cdot s^{-1} + \gamma_m\right] \cdot \theta_m & \text{otherwise} \end{cases}$$

where:

$$\gamma_m = \frac{\lambda_m - \lambda_{\text{short}}}{\lambda_{\text{long}} - \lambda_{\text{short}}}$$

Additionally, an attention temperature scaling factor is applied:

$$\text{scale} = \frac{1}{\sqrt{0.1 \ln(s) + 1}}$$

**Applied to dense layers only**, since banded layers with window $w = 128$ see only local positions regardless of sequence length.

### 2.5 Mixture-of-Experts (MoE) Block

#### 2.5.1 Router

A linear projection maps the residual stream to expert scores:

$$\mathbf{g}(\mathbf{x}) = \mathbf{W}_r \, \mathbf{x} \in \mathbb{R}^{N_e}$$

where $\mathbf{W}_r \in \mathbb{R}^{N_e \times d}$.

**Router parameter count per layer:**
- 120B: $128 \times 2{,}880 = 368{,}640$
- 20B: $32 \times 2{,}880 = 92{,}160$

**Top-$k$ Selection ($k = 4$):**

$$\mathcal{T}(\mathbf{x}) = \text{TopK}(\mathbf{g}(\mathbf{x}), k=4)$$

**Routing weights (softmax over selected experts only):**

$$p_i(\mathbf{x}) = \frac{\exp(g_i(\mathbf{x}))}{\sum_{j \in \mathcal{T}(\mathbf{x})} \exp(g_j(\mathbf{x}))}, \quad i \in \mathcal{T}(\mathbf{x})$$

$$p_i(\mathbf{x}) = 0, \quad i \notin \mathcal{T}(\mathbf{x})$$

#### 2.5.2 Expert Network (SwiGLU with Clamping and Residual)

Each expert $E_i$ implements a **modified SwiGLU** feedforward network:

**Standard SwiGLU:**

$$\text{SwiGLU}(\mathbf{x}) = \left[\sigma(\mathbf{W}_{\text{gate}} \, \mathbf{x}) \odot (\mathbf{W}_{\text{up}} \, \mathbf{x})\right] \mathbf{W}_{\text{down}}$$

where $\sigma(\cdot) = \text{SiLU}(\cdot) = \cdot \times \text{sigmoid}(\cdot)$.

**GPT-OSS unconventional modifications:**

1. **Clamping:** Applied to gate activations to prevent numerical overflow:

$$\text{gate}(\mathbf{x}) = \text{clamp}\!\left(\sigma(\mathbf{W}_{\text{gate}} \, \mathbf{x}),\; -C,\; C\right)$$

2. **Residual connection within the expert:**

$$E_i(\mathbf{x}) = \text{SwiGLU}_i^{\text{clamped}}(\mathbf{x}) + \mathbf{W}_{\text{res},i} \, \mathbf{x}$$

or equivalently, an identity-mapped skip:

$$E_i(\mathbf{x}) = \left[\text{clamp}\!\left(\sigma(\mathbf{W}_{\text{gate},i} \, \mathbf{x}), -C, C\right) \odot (\mathbf{W}_{\text{up},i} \, \mathbf{x})\right] \mathbf{W}_{\text{down},i} + \mathbf{x}_{\text{proj}}$$

**Expert weight dimensions:**

$$\mathbf{W}_{\text{gate},i}, \, \mathbf{W}_{\text{up},i} \in \mathbb{R}^{d_{ff} \times d} = \mathbb{R}^{2880 \times 2880}$$

$$\mathbf{W}_{\text{down},i} \in \mathbb{R}^{d \times d_{ff}} = \mathbb{R}^{2880 \times 2880}$$

**Parameters per expert:**

$$|E_i| = 3 \times d \times d_{ff} = 3 \times 2{,}880^2 = 24{,}883{,}200 \approx 24.88\text{M}$$

#### 2.5.3 MoE Output

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \mathcal{T}(\mathbf{x})} p_i(\mathbf{x}) \cdot E_i(\mathbf{x})$$

#### 2.5.4 MoE Parameter Budget Verification

**Per-layer MLP parameters (experts + router):**

| Model | Experts | Router | Total/layer |
|---|---|---|---|
| 120B | $128 \times 24.88\text{M} = 3{,}184.6\text{M}$ | $0.37\text{M}$ | $3{,}185.0\text{M}$ |
| 20B | $32 \times 24.88\text{M} = 796.2\text{M}$ | $0.09\text{M}$ | $796.3\text{M}$ |

**Total MLP parameters:**
- 120B: $36 \times 3{,}185.0\text{M} = 114.66\text{B}$ (reported: 114.71B — delta accounts for clamping/residual params and RMSNorm gains)
- 20B: $24 \times 796.3\text{M} = 19.11\text{B}$ (reported: 19.12B) ✓

#### 2.5.5 Active Parameter Analysis

Per-token active computation per layer:

$$\text{Active}^{(\ell)} = |\text{Attn}^{(\ell)}| + k \cdot |E| + |\mathbf{W}_r| + 2 \cdot |\boldsymbol{\gamma}_{\text{RMSNorm}}|$$

$$= 26.54\text{M} + 4 \times 24.88\text{M} + 0.37\text{M} + 2 \times 2{,}880$$

$$= 26.54 + 99.52 + 0.37 + 0.006 \approx 126.4\text{M}$$

**Total active parameters:**
- 120B: $36 \times 126.4\text{M} + |\mathbf{U}| = 4{,}550\text{M} + 579\text{M} = 5{,}129\text{M} \approx 5.13\text{B}$ ✓
- 20B: $24 \times 126.4\text{M} + 579\text{M} = 3{,}034\text{M} + 579\text{M} = 3{,}613\text{M} \approx 3.61\text{B}$ ✓

**Sparsity ratio:**
- 120B: $5.13 / 116.83 = 4.39\%$ active
- 20B: $3.61 / 20.91 = 17.27\%$ active

### 2.6 Full Forward Pass Pseudo-Algorithm

```
ALGORITHM: GPT-OSS-Forward
Input: Token sequence x = (x_1, ..., x_T), model parameters θ
Output: Logit sequence logits ∈ R^{T × V}

1.  h_t^(0) ← E[x_t]                              ∀t ∈ {1,...,T}
2.  FOR ℓ = 1 TO L:
3.      // --- Attention sub-layer ---
4.      z ← RMSNorm(h^(ℓ-1))                       // [T, d]
5.      Q ← z · W_Q^(ℓ)                             // [T, n_q · d_h]
6.      K ← z · W_K^(ℓ)                             // [T, n_kv · d_h]
7.      V ← z · W_V^(ℓ)                             // [T, n_kv · d_h]
8.      Q, K ← ApplyRoPE(Q, K, positions)
9.      IF ℓ is dense layer:
10.         S_t ← {1,...,t}                           // full causal
11.         Apply YaRN scaling if T > L_pretrain
12.     ELSE:
13.         S_t ← {max(1,t-w+1),...,t}               // banded, w=128
14.     FOR each head i:
15.         α_{t,s}^(i) ← exp(q_t·k_s/√d_h) / 
                           (exp(b^(i)) + Σ_{s'∈S_t} exp(q_t·k_{s'}/√d_h))
16.         o_t^(i) ← Σ_{s∈S_t} α_{t,s}^(i) · v_s^(i)
17.     attn_out ← Concat(o^(1),...,o^(n_q)) · W_O^(ℓ)
18.     h^(ℓ) ← h^(ℓ-1) + attn_out
19.
20.     // --- MoE sub-layer ---
21.     z' ← RMSNorm(h^(ℓ))                          // [T, d]
22.     FOR each token t:
23.         g ← W_r^(ℓ) · z'_t                        // [N_e]
24.         T(t) ← TopK(g, k=4)
25.         p_i ← softmax(g[T(t)])                     // over selected k
26.         moe_out_t ← Σ_{i∈T(t)} p_i · Expert_i(z'_t)
27.     h^(ℓ) ← h^(ℓ) + moe_out
28.
29. logits ← RMSNorm(h^(L)) · U^T                    // [T, V]
30. RETURN logits
```

### 2.7 Complexity Analysis

#### 2.7.1 FLOPs per Token per Layer

**Attention (linear projections):**

$$F_{\text{attn-proj}} = 2d(n_q d_h + 2 n_{kv} d_h + n_q d_h) = 2 \times 2880 \times (4096 + 1024 + 4096) = 53.1\text{M FLOPs}$$

**Attention (dot products + output, dense layer):**

$$F_{\text{attn-score}} = 2 \times n_q \times T \times d_h = 2 \times 64 \times T \times 64 = 8{,}192 \cdot T \text{ FLOPs}$$

**MoE (per token, $k=4$ experts):**

$$F_{\text{MoE}} = k \times (2 \times d \times d_{ff} + 2 \times d \times d_{ff} + 2 \times d_{ff} \times d) = 4 \times 6 \times d \times d_{ff}$$

$$= 24 \times 2880^2 = 199.1\text{M FLOPs}$$

**Total per token per layer (dense attention, ignoring router):**

$$F^{(\ell)} \approx 53.1\text{M} + 8{,}192 \cdot T + 199.1\text{M} = 252.2\text{M} + 8{,}192 \cdot T$$

For $T = 131{,}072$: $F^{(\ell)} \approx 252.2\text{M} + 1{,}073.7\text{M} = 1{,}325.9\text{M}$ per token per dense layer.

For banded layers: $8{,}192 \times 128 = 1.05\text{M}$, so $F^{(\ell)}_{\text{banded}} \approx 253.3\text{M}$.

#### 2.7.2 KV Cache Memory

Per layer (GQA with $n_{kv} = 8$, $d_h = 64$):

$$\text{KV per layer} = 2 \times n_{kv} \times d_h \times T \times b = 2 \times 8 \times 64 \times T \times b$$

where $b$ is bytes per element (2 for BF16).

At $T = 131{,}072$:

$$\text{KV per layer} = 2 \times 8 \times 64 \times 131{,}072 \times 2 = 268.4\text{MB}$$

**Total KV cache:**
- 120B: $36 \times 268.4\text{MB} = 9.66\text{GB}$
- 20B: $24 \times 268.4\text{MB} = 6.44\text{GB}$

**Note:** Banded layers only need KV cache of size $w = 128$ during inference if sliding window is used, reducing practical KV memory by approximately $L/2$.

### 2.8 Failure Modes (Architecture Level)

| Failure Mode | Mechanism | Mitigation |
|---|---|---|
| Expert collapse | Router converges to always select same experts | Load-balancing auxiliary loss during training |
| Attention sink saturation | All mass goes to learned bias $b^{(i)}$ | Bias initialization near zero; monitoring |
| RoPE frequency aliasing | High-frequency dimensions fail at long contexts | YaRN interpolation + graduated frequency ramp |
| Numerical overflow in SwiGLU | Unbounded gate activations | Clamping (explicitly implemented) |
| Token-dropping in MoE | Expert capacity overflow in training | Capacity factor / auxiliary loss |

---

## 3. Compression Pipeline (Quantization)

### 3.1 Objective

Reduce the memory footprint of MoE weights (which constitute $>90\%$ of total parameters) to enable single-GPU deployment while preserving model quality.

**Inputs:** Full-precision (BF16) model weights $\theta$.

**Outputs:** Quantized checkpoint $\hat{\theta}$ with MoE weights in MXFP4 format.

**Constraint:** Quantization is integrated into post-training (quantization-aware), not applied post-hoc.

### 3.2 MXFP4 (Microscaling FP4) Format

#### 3.2.1 Format Specification

MXFP4 groups weight elements into blocks of $B = 32$ and shares a single 8-bit scaling exponent per block:

- **Per-element:** 4 bits (1 sign bit, $e$ exponent bits, $m$ mantissa bits with $1 + e + m = 4$)
- **Per-block shared exponent:** 8 bits

**Effective bits per parameter:**

$$b_{\text{eff}} = 4 + \frac{8}{B} = 4 + \frac{8}{32} = 4.25 \text{ bits/param}$$

#### 3.2.2 Quantization Function

For a block of 32 weights $\mathbf{w} = (w_1, \ldots, w_{32})$:

1. **Compute shared exponent:**

$$e_{\text{shared}} = \left\lfloor \log_2\!\left(\max_{i} |w_i|\right) \right\rfloor$$

2. **Scale to shared exponent range:**

$$\tilde{w}_i = w_i \cdot 2^{-e_{\text{shared}}}$$

3. **Quantize each element to nearest FP4 representable value:**

$$\hat{w}_i = Q_{\text{FP4}}(\tilde{w}_i) = \arg\min_{q \in \mathcal{G}_{\text{FP4}}} |q - \tilde{w}_i|$$

where $\mathcal{G}_{\text{FP4}}$ is the set of representable FP4 values.

4. **Reconstruction:**

$$\hat{w}_i^{\text{recon}} = \hat{w}_i \cdot 2^{e_{\text{shared}}}$$

#### 3.2.3 Quantization Error Bound

The quantization error per element is bounded by:

$$|\hat{w}_i^{\text{recon}} - w_i| \leq \frac{\Delta_{\text{FP4}}}{2} \cdot 2^{e_{\text{shared}}}$$

where $\Delta_{\text{FP4}}$ is the finest quantization step in the FP4 grid. The relative error is:

$$\frac{|\hat{w}_i^{\text{recon}} - w_i|}{|w_i|} \leq \frac{1}{2^{m+1}} = \frac{1}{2^{m+1}}$$

where $m$ is the number of mantissa bits (typically $m = 2$ for FP4, giving relative error $\leq 12.5\%$).

### 3.3 Quantization-Aware Post-Training

The model was **post-trained with MXFP4 quantization applied to MoE weights**, meaning the quantization operator $Q$ is in the forward pass during RL post-training:

$$\hat{\mathbf{W}}_{\text{MoE}} = Q_{\text{MXFP4}}(\mathbf{W}_{\text{MoE}})$$

The gradient is approximated via the Straight-Through Estimator (STE):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{\text{MoE}}} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{W}}_{\text{MoE}}}$$

This ensures the model adapts its weight distribution to be robust to quantization noise, preventing quality degradation at inference time.

### 3.4 Checkpoint Size Derivation

**GPT-OSS-120B:**

| Component | Params | Format | Size |
|---|---|---|---|
| MoE weights | 114.71B | MXFP4 (4.25 bpp) | $114.71 \times 10^9 \times 4.25 / 8 = 60.94$ GB $\approx 56.77$ GiB |
| Attention weights | 0.96B | BF16 (16 bpp) | $0.96 \times 10^9 \times 2 = 1.92$ GB $\approx 1.79$ GiB |
| Embed + Unembed | 1.16B | BF16 (16 bpp) | $1.16 \times 10^9 \times 2 = 2.32$ GB $\approx 2.16$ GiB |
| RMSNorm + biases | ~0.2M | BF16 | negligible |
| **Total** | **116.83B** | | **≈ 60.8 GiB** ✓ |

**GPT-OSS-20B:**

| Component | Params | Format | Size |
|---|---|---|---|
| MoE weights | 19.12B | MXFP4 | $19.12 \times 10^9 \times 4.25 / 8 = 10.16$ GB $\approx 9.46$ GiB |
| Attention + Embed | 1.79B | BF16 | $1.79 \times 10^9 \times 2 = 3.58$ GB $\approx 3.33$ GiB |
| **Total** | **20.91B** | | **≈ 12.8 GiB** ✓ |

**Compression ratios:**
- 120B: from $116.83\text{B} \times 2 = 233.66$ GB (BF16) to 60.8 GiB → **$\approx 3.6\times$ compression**
- 20B: from $20.91\text{B} \times 2 = 41.82$ GB (BF16) to 12.8 GiB → **$\approx 3.1\times$ compression**

### 3.5 Information Preservation Analysis

**Quantization noise model:** The quantization error on the MoE output is:

$$\delta_{\text{MoE}}(\mathbf{x}) = \sum_{i \in \mathcal{T}} p_i \cdot \left[E_i(\mathbf{x}; \hat{\mathbf{W}}_i) - E_i(\mathbf{x}; \mathbf{W}_i)\right]$$

Since only $k = 4$ experts are active per token and the softmax routing weights $p_i \in [0,1]$ sum to 1, the output perturbation is bounded:

$$\|\delta_{\text{MoE}}\| \leq \max_{i \in \mathcal{T}} \|E_i(\mathbf{x}; \hat{\mathbf{W}}_i) - E_i(\mathbf{x}; \mathbf{W}_i)\|$$

Quantization-aware training minimizes $\mathbb{E}[\|\delta_{\text{MoE}}\|^2]$ throughout the RL optimization, ensuring the final checkpoint has adapted to the quantization grid.

### 3.6 Compression Pseudo-Algorithm

```
ALGORITHM: MXFP4-Quantize
Input: Weight tensor W ∈ R^{m×n} (MoE expert), block size B=32
Output: Quantized representation (Q_blocks, E_shared)

1.  Flatten W → w ∈ R^{m·n}
2.  Partition w into blocks of size B: {w_1, ..., w_{⌈mn/B⌉}}
3.  FOR each block b_j:
4.      e_j ← floor(log2(max(|b_j|)))           // shared exponent
5.      FOR each element w_i ∈ b_j:
6.          w_scaled ← w_i · 2^{-e_j}
7.          q_i ← NearestFP4(w_scaled)           // quantize to FP4 grid
8.      Q_blocks[j] ← pack({q_i}, 4 bits each)
9.      E_shared[j] ← encode(e_j, 8 bits)
10. RETURN (Q_blocks, E_shared)

ALGORITHM: MXFP4-Dequantize
Input: (Q_blocks, E_shared)
Output: Reconstructed W_hat

1.  FOR each block j:
2.      FOR each element q_i ∈ Q_blocks[j]:
3.          w_hat_i ← DecodeFP4(q_i) · 2^{Decode(E_shared[j])}
4.  RETURN Reshape(w_hat, [m, n])
```

### 3.7 Failure Modes (Compression)

| Failure Mode | Description | Mitigation |
|---|---|---|
| Outlier clipping | Large weight outliers lose precision under shared exponent | QAT adapts weight distributions; clamping in SwiGLU prevents extreme activations |
| Block boundary artifacts | Weights at block boundaries have suboptimal exponent sharing | Block size 32 is a compromise between granularity and overhead |
| Gradient bias from STE | Straight-through estimator introduces bias in gradient estimates | Sufficiently long QAT schedule allows convergence despite STE bias |
| Non-MoE precision mismatch | BF16 non-MoE + MXFP4 MoE creates mixed-precision numerical dynamics | Pre-LN + RMSNorm stabilizes residual stream magnitudes |

---

## 4. Optimization Strategy

### 4.1 Pretraining Optimization

#### 4.1.1 Hardware and Framework

- **GPUs:** NVIDIA H100 (80 GB HBM3)
- **Framework:** PyTorch with expert-optimized Triton kernels
- **Attention kernel:** Flash Attention (specifically FlashAttention-2/3 compatible)
- **Compute budget:**
  - 120B: 2.1 million H100-hours
  - 20B: ~210K H100-hours ($\approx 10\times$ fewer)

#### 4.1.2 Pretraining Loss

Standard autoregressive cross-entropy:

$$\mathcal{L}_{\text{pretrain}}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

$$= -\frac{1}{T} \sum_{t=1}^{T} \log \frac{\exp(\text{logits}_{x_t})}{\sum_{v=1}^{V} \exp(\text{logits}_v)}$$

#### 4.1.3 MoE Auxiliary Load-Balancing Loss

To prevent expert collapse, an auxiliary load-balancing loss is added:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{N_e} f_i \cdot P_i$$

where:
- $f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}[i \in \mathcal{T}(\mathbf{x}_t)]$ is the fraction of tokens routed to expert $i$
- $P_i = \frac{1}{T} \sum_{t=1}^{T} p_i(\mathbf{x}_t)$ is the mean routing probability for expert $i$
- $\alpha > 0$ is the balancing coefficient

**Desired equilibrium:** $f_i = k / N_e$ and $P_i = k / N_e$ for all $i$, giving uniform expert utilization.

**Total pretraining objective:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pretrain}} + \mathcal{L}_{\text{aux}}$$

#### 4.1.4 Distributed Training Strategy

Given model sizes and compute budgets, the training likely employs:

- **Expert Parallelism:** Each expert is placed on a subset of GPUs; all-to-all communication dispatches tokens to their assigned expert GPUs
- **Tensor Parallelism (TP):** Attention weight matrices are sharded across GPUs within a node
- **Data Parallelism (DP):** Gradient synchronization across replica groups via ZeRO or FSDP
- **Sequence Parallelism:** For long-context training (131K tokens), activation memory is distributed across the sequence dimension
- **Pipeline Parallelism:** Layer-level pipeline staging across nodes
- **Activation Checkpointing:** Recompute activations during backward pass to reduce memory

**Communication overhead for MoE:**

All-to-all token dispatch per MoE layer:

$$\text{Comm}_{\text{a2a}} = 2 \times k \times d \times T_{\text{local}} \times b_{\text{comm}}$$

where $T_{\text{local}}$ is the token count per GPU and $b_{\text{comm}}$ is the communication bandwidth utilization.

#### 4.1.5 Flash Attention Integration

Flash Attention reduces attention memory from $O(T^2)$ to $O(T)$ by computing attention in tiled blocks in SRAM without materializing the full $T \times T$ attention matrix:

**IO complexity:**

$$\text{Standard:}\; O\!\left(\frac{T^2 \cdot n_q \cdot d_h}{}\right) \text{ HBM reads/writes}$$

$$\text{FlashAttention:}\; O\!\left(\frac{T^2 \cdot n_q \cdot d_h^2}{M_{\text{SRAM}}}\right) \text{ HBM reads/writes}$$

where $M_{\text{SRAM}}$ is the on-chip SRAM size. With $d_h = 64$ and typical SRAM of 20–40 MB on H100, this yields significant IO reduction.

For banded layers with window $w = 128$: the attention computation is already $O(Tw)$, so Flash Attention primarily helps dense layers.

### 4.2 Post-Training Optimization

#### 4.2.1 Reinforcement Learning for Reasoning (CoT RL)

Post-training uses **chain-of-thought reinforcement learning**, similar to the OpenAI o3 training pipeline.

**Objective (GRPO-style / outcome-based RL):**

Given a prompt $x$ and a set of $N$ sampled completions $\{y^{(i)}\}_{i=1}^{N}$, each receiving a reward $r^{(i)}$:

$$\mathcal{L}_{\text{RL}}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{N} \sum_{i=1}^{N} A^{(i)} \cdot \log p_\theta(y^{(i)} \mid x) \right]$$

where the advantage is computed as:

$$A^{(i)} = \frac{r^{(i)} - \text{mean}(\{r^{(j)}\}_{j=1}^N)}{\text{std}(\{r^{(j)}\}_{j=1}^N) + \epsilon}$$

**KL regularization against reference policy:**

$$\mathcal{L}_{\text{post-train}} = \mathcal{L}_{\text{RL}} + \beta \cdot D_{\text{KL}}(p_\theta \| p_{\text{ref}})$$

where $p_{\text{ref}}$ is the pretrained model (or SFT checkpoint) and $\beta$ controls deviation from the reference.

**Reward sources:**
- **Verifiable rewards:** Automated verification for math (exact match), code (test execution), structured outputs
- **Reward model:** Learned preference model for open-ended tasks

#### 4.2.2 Quantization-Aware RL

MXFP4 quantization is applied to MoE weights during the RL forward pass, so the policy gradient is computed through the quantized model:

$$\nabla_\theta \mathcal{L}_{\text{RL}} \approx \nabla_\theta \mathcal{L}_{\text{RL}}\!\left(\theta; Q_{\text{MXFP4}}(\mathbf{W}_{\text{MoE}})\right)$$

with STE through the quantization boundaries.

#### 4.2.3 Safety Post-Training: Deliberative Alignment

**Mechanism:** The model is trained to explicitly reason about safety policies in its chain of thought before deciding whether to comply with or refuse a request.

$$p_\theta(\text{refuse} \mid x) = \sum_{\text{CoT } c} p_\theta(c \mid x) \cdot p_\theta(\text{refuse} \mid c, x)$$

The RL reward incorporates a safety component:

$$r_{\text{total}} = r_{\text{task}} + \lambda_{\text{safety}} \cdot r_{\text{safety}}$$

where $r_{\text{safety}}$ penalizes compliance with disallowed requests and rewards appropriate refusals, following OpenAI's content policy taxonomy.

#### 4.2.4 Instruction Hierarchy Training

The model is trained with structured examples where messages from different roles conflict, enforcing the hierarchy:

$$\text{System} \succ \text{Developer} \succ \text{User} \succ \text{Assistant} \succ \text{Tool}$$

**Training signal:** When a user instruction contradicts a system/developer instruction, the correct behavior is to follow the higher-priority role. This is enforced via supervised examples and RL penalties for hierarchy violations.

### 4.3 Training Stages Summary

```
ALGORITHM: Full Training Pipeline
Input: Raw corpus R, safety filters, tokenizer, H100 cluster
Output: Quantized model checkpoint θ_hat

STAGE 1: DATA PREPARATION
  D ← CBRN_Filter(Deduplicate(R))
  D_tok ← Tokenize(D, o200k_harmony)

STAGE 2: PRETRAINING
  θ_0 ← RandomInit(architecture_config)
  FOR step = 1 TO S_pretrain:
    batch ← Sample(D_tok, domain_weights)
    L ← L_pretrain(θ, batch) + L_aux(θ, batch)
    θ ← OptimizerStep(θ, ∇L)                    // AdamW + cosine LR

STAGE 3: SUPERVISED FINE-TUNING (implied)
  θ_sft ← SFT(θ, D_sft)                         // on demonstration data

STAGE 4: RL POST-TRAINING (with QAT)
  θ_ref ← θ_sft
  FOR step = 1 TO S_RL:
    x ← Sample(D_RL)                             // math, code, science, ...
    {y^(i)} ← SampleCompletions(θ, x, N)         // with MXFP4 in forward
    {r^(i)} ← ComputeRewards(x, {y^(i)})         // verifiable + RM + safety
    L ← L_RL(θ, {y^(i)}, {r^(i)}) + β·KL(θ||θ_ref)
    θ ← OptimizerStep(θ, ∇L)

STAGE 5: CHECKPOINT EXPORT
  θ_hat ← MXFP4_Quantize(θ.MoE) ∪ BF16(θ.non_MoE)
  RETURN θ_hat
```

---

## 5. Training Stages: Detailed Exposition

### 5.1 Stage 1: Pretraining

#### 5.1.1 Objective, Inputs, Outputs

- **Objective:** Learn general next-token prediction over web-scale text
- **Input:** Tokenized corpus, trillions of tokens
- **Output:** Pretrained base model $\theta_{\text{base}}$
- **Invariant:** Perplexity monotonically decreasing on held-out validation sets; expert utilization approximately uniform

#### 5.1.2 Compute Scaling Analysis

For the 120B model with 5.13B active params and 2.1M H100-hours:

**Estimated FLOPs (assuming $\sim$1000 TFLOPS/H100 effective on MoE):**

$$C_{\text{total}} = 2.1 \times 10^6 \times 3600 \times 1000 \times 10^{12} \approx 7.56 \times 10^{24} \text{ FLOPs}$$

**Chinchilla-style token count estimate:**

Using $C \approx 6 N_{\text{active}} D$ where $D$ is tokens trained on:

$$D \approx \frac{C}{6 N_{\text{active}}} = \frac{7.56 \times 10^{24}}{6 \times 5.13 \times 10^9} \approx 2.46 \times 10^{14} \approx 246\text{T tokens}$$

This is broadly consistent with "trillions of tokens."

For the 20B model ($\sim$10$\times$ fewer compute): $\sim$0.21M H100-hours, $\sim$25T tokens (or fewer with larger batch/shorter schedule).

### 5.2 Stage 2: Post-Training for Reasoning

#### 5.2.1 Chain-of-Thought RL

The model is taught to produce explicit reasoning traces (chain-of-thought) before final answers, using reinforcement learning with outcome-based rewards.

**Formalization:**

Let $y = (c, a)$ where $c$ is the CoT (analysis channel) and $a$ is the final answer (final channel). The policy generates:

$$y \sim p_\theta(\cdot \mid x) = p_\theta(c \mid x) \cdot p_\theta(a \mid c, x)$$

The reward $r(x, a)$ is computed on the **final answer** $a$ only (outcome-based), but the entire sequence $(c, a)$ receives credit:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x, y \sim p_\theta} \left[ r(x, a) \cdot \nabla_\theta \log p_\theta(y \mid x) \right]$$

This incentivizes the model to produce CoT traces that lead to correct answers, without direct supervision on the CoT content itself.

**Critical design choice:** No direct optimization pressure is placed on the CoT content. This preserves CoT monitorability—the reasoning traces genuinely reflect the model's internal deliberation rather than being optimized to appear benign.

#### 5.2.2 Variable Effort Reasoning

Three reasoning levels are trained: `low`, `medium`, `high`, controlled by system prompt keywords (e.g., `"Reasoning: low"`).

**Mechanism:** During RL, the reasoning budget is varied:

$$p_\theta(y \mid x, \text{effort}=e) \quad \text{for } e \in \{\text{low}, \text{medium}, \text{high}\}$$

The expected CoT length scales with effort level:

$$\mathbb{E}[|c| \mid e=\text{low}] < \mathbb{E}[|c| \mid e=\text{medium}] < \mathbb{E}[|c| \mid e=\text{high}]$$

**Test-time scaling behavior (from empirical results):**

The relationship between average CoT + answer length $\bar{L}$ and accuracy follows approximately log-linear scaling:

$$\text{Accuracy} \approx \alpha + \beta \cdot \log(\bar{L})$$

Empirical data from Figure 3:

| Benchmark | Model | Low $(\bar{L}, \text{Acc})$ | Med $(\bar{L}, \text{Acc})$ | High $(\bar{L}, \text{Acc})$ |
|---|---|---|---|---|
| AIME 2025 | 120B | (~2K, ~65%) | (~5K, ~80%) | (~12K, ~93%) |
| AIME 2025 | 20B | (~2K, ~50%) | (~5K, ~72%) | (~15K, ~92%) |
| GPQA | 120B | (~3K, ~67%) | (~6K, ~73%) | (~16K, ~80%) |
| GPQA | 20B | (~3K, ~57%) | (~7K, ~66%) | (~30K, ~72%) |

### 5.3 Stage 3: Agentic Tool Use Training

#### 5.3.1 Tool Specification

Three tool categories trained during post-training:

1. **Browsing tool:** `search(query)` and `open(url)` functions for web interaction
2. **Python tool:** Stateful Jupyter notebook execution environment
3. **Developer-defined functions:** Arbitrary function schemas specified in Developer messages

#### 5.3.2 Tool Call Mechanism

Within the harmony format, the model generates tool calls on the `commentary` channel, receives results on the `tool` channel, and can interleave:

$$\text{CoT (analysis)} \to \text{ToolCall (commentary)} \to \text{ToolResult (tool)} \to \text{CoT (analysis)} \to \text{FinalAnswer (final)}$$

The model is trained to decide whether to use tools based on system prompt specification (`"Tools: browsing, python"` or `"Tools: none"`).

#### 5.3.3 Tool-Augmented Reward

For tool-using tasks, the reward function evaluates the final outcome regardless of tool use:

$$r(x, a) = \begin{cases} 1 & \text{if } a \text{ is correct (verified)} \\ 0 & \text{otherwise} \end{cases}$$

The model learns instrumentally that tool use improves answer quality, without explicit tool-use supervision.

### 5.4 Convergence Dynamics and Failure Modes

| Aspect | Details |
|---|---|
| **Reward hacking** | Model may exploit verification loopholes (e.g., code tests that pass incorrectly); mitigated by diverse reward signals |
| **CoT degeneracy** | Model could produce very long but uninformative CoTs; variable effort training and length penalties regulate this |
| **Mode collapse** | RL may collapse to narrow output distribution; KL regularization against $p_{\text{ref}}$ prevents this |
| **Tool misuse** | Model may call tools unnecessarily or in infinite loops; episode length limits and tool call budgets |
| **Safety-capability tension** | Increasing reasoning capability may increase refusal-bypass risk; deliberative alignment and instruction hierarchy training address this |

---

## 6. Inference Path

### 6.1 Harmony Chat Format

#### 6.1.1 Message Structure

Messages are delimited by special tokens with keyword arguments:

```
<|header_start|>role=System<|header_end|>
{system_message}
<|message_end|>

<|header_start|>role=Developer<|header_end|>
{developer_message}
<|message_end|>

<|header_start|>role=User<|header_end|>
{user_message}
<|message_end|>

<|header_start|>role=Assistant channel=analysis<|header_end|>
{chain_of_thought}
<|channel_end|>

<|header_start|>role=Assistant channel=commentary<|header_end|>
{tool_call}
<|channel_end|>

<|header_start|>role=Assistant channel=final<|header_end|>
{final_answer}
<|message_end|>
```

#### 6.1.2 Multi-Turn Handling

**Critical implementation detail:** In multi-turn conversations, reasoning traces from past assistant turns **must be removed** before providing context for the next turn. This prevents:
- Context window waste from accumulated CoTs
- Cross-turn reasoning contamination
- Attention distraction from irrelevant past reasoning

### 6.2 Inference Pseudo-Algorithm

```
ALGORITHM: GPT-OSS-Inference
Input: Conversation history H, system prompt S, tools T, effort level e
Output: Final answer a, (optional) tool calls, (optional) CoT c

1.  prompt ← FormatHarmony(S, H, T, e)
2.  // Remove past CoTs from H
3.  FOR each past_turn in H:
4.      past_turn.remove(channel=analysis)
5.  
6.  // Load quantized model
7.  W_MoE ← MXFP4_Dequantize(checkpoint.MoE)      // on-the-fly or cached
8.  W_other ← checkpoint.BF16_params
9.  
10. // Autoregressive generation
11. tokens ← Tokenize(prompt)
12. kv_cache ← Initialize(L_layers)
13. 
14. WHILE not EOS:
15.     logits ← Forward(tokens[-1], kv_cache, W_MoE, W_other)  // Step 2.6
16.     // For banded layers: maintain sliding window KV of size w=128
17.     // For dense layers: append to full KV cache
18.     next_token ← Sample(logits, temperature, top_p)
19.     tokens.append(next_token)
20.     
21.     IF next_token == <channel_end> AND current_channel == "commentary":
22.         // Parse tool call
23.         tool_call ← ParseToolCall(tokens[channel_start:])
24.         tool_result ← ExecuteTool(tool_call, T)
25.         tokens.extend(FormatToolResult(tool_result))
26.     
27.     IF next_token == <message_end>:
28.         BREAK
29. 
30. c ← ExtractChannel(tokens, "analysis")
31. a ← ExtractChannel(tokens, "final")
32. RETURN a, c
```

### 6.3 KV Cache Management at Inference

#### 6.3.1 Hybrid KV Strategy

Due to alternating attention patterns:

- **Dense layers:** Full KV cache up to $T$ positions, size $2 \times n_{kv} \times d_h \times T = 1{,}024 \times T$ bytes (BF16)
- **Banded layers:** Sliding window of $w = 128$ positions, size $2 \times n_{kv} \times d_h \times w = 1{,}024 \times 128 = 131{,}072$ bytes per layer

**Total KV at sequence length $T$:**

$$\text{KV}_{\text{total}} = \frac{L}{2} \cdot 1{,}024 T + \frac{L}{2} \cdot 1{,}024 \times 128 \text{ bytes (BF16)}$$

For 120B at $T = 131{,}072$:

$$\text{KV} = 18 \times 1{,}024 \times 131{,}072 + 18 \times 1{,}024 \times 128 = 2.42\text{GB} + 2.36\text{MB} \approx 2.42\text{GB}$$

compared to full KV on all 36 layers: $36 \times 1{,}024 \times 131{,}072 = 4.83\text{GB}$.

**Savings from hybrid strategy:** ~50% KV memory reduction.

### 6.4 Single-GPU Deployment Feasibility

**GPT-OSS-120B on 80 GB H100:**

| Component | Memory |
|---|---|
| Model weights (MXFP4 checkpoint) | 60.8 GiB |
| KV cache (131K context) | ~2.4 GiB |
| Activations (single token) | ~0.1 GiB |
| Framework overhead | ~2–5 GiB |
| **Total** | **~65–68 GiB** ✓ (fits in 80 GB) |

**GPT-OSS-20B on 16 GB GPU:**

| Component | Memory |
|---|---|
| Model weights | 12.8 GiB |
| KV cache (shorter context) | ~1.0 GiB |
| Activations + overhead | ~1–2 GiB |
| **Total** | **~15 GiB** ✓ (tight fit in 16 GB) |

### 6.5 Inference Failure Modes

| Failure Mode | Mechanism | Mitigation |
|---|---|---|
| Hallucinated CoT | No optimization pressure on CoT content → CoT may contain factually incorrect or policy-violating content | Do not expose CoT to end users without filtering/moderation |
| Factual hallucination | Small model size limits parametric knowledge | Enable browsing tool; tool use significantly reduces hallucination |
| Tool call loops | Model repeatedly calls tools without convergence | Max tool call budget per turn |
| Context overflow | Long CoTs + tool results exhaust 131K window | Truncation with priority to recent context; CoT pruning between turns |
| MXFP4 dequantization latency | On-the-fly dequantization adds overhead | Cache dequantized weights in memory; fused kernels |

---

## 7. Evaluation Protocol

### 7.1 Evaluation Framework

**Protocol:** Basic pass@1 at `high` reasoning mode using default system prompt. All results are single-attempt accuracy unless otherwise specified.

**Comparison baselines:** OpenAI o3, o3-mini, o4-mini, GPT-4o.

### 7.2 Reasoning and Knowledge Benchmarks

#### 7.2.1 AIME (Competition Mathematics)

| Configuration | 120B (low/med/high) | 20B (low/med/high) |
|---|---|---|
| AIME 2024, no tools | 56.3 / 80.4 / 95.8 | 42.1 / 80.0 / 92.1 |
| AIME 2024, with tools | 75.4 / 87.9 / 96.6 | 61.2 / 86.0 / 96.0 |
| AIME 2025, no tools | 50.4 / 80.0 / 92.5 | 37.1 / 72.1 / 91.7 |
| AIME 2025, with tools | 72.9 / 91.6 / 97.9 | 57.5 / 90.4 / 98.7 |

**Key finding:** GPT-OSS-20B uses >20K CoT tokens per AIME problem on average, compensating for smaller parametric knowledge through extended reasoning. Tool use (Python execution) provides substantial uplift, especially at lower effort levels.

**Comparison:** 120B-high approaches o4-mini (98.7 AIME 2024, 99.5 AIME 2025), surpasses o3-mini (87.3 / 86.5).

#### 7.2.2 GPQA Diamond (PhD-Level Science)

| Configuration | 120B | 20B |
|---|---|---|
| No tools (low/med/high) | 67.1 / 73.1 / 80.1 | 56.8 / 66.0 / 71.5 |
| With tools (low/med/high) | 68.1 / 73.5 / 80.9 | 58.0 / 67.1 / 74.2 |

**Key finding:** Knowledge-intensive tasks show less tool uplift and greater sensitivity to model size. The 20B model lags by ~9 points due to smaller parametric knowledge.

#### 7.2.3 MMLU (College-Level)

| Model | Accuracy |
|---|---|
| 120B (low/med/high) | 85.9 / 88.0 / 90.0 |
| 20B (low/med/high) | 80.4 / 84.0 / 85.3 |
| o3 | 93.4 |

#### 7.2.4 HLE (Expert-Level)

| Configuration | 120B | 20B |
|---|---|---|
| No tools (low/med/high) | 5.2 / 8.6 / 14.9 | 4.2 / 7.0 / 10.9 |
| With tools (low/med/high) | 9.1 / 11.3 / 19.0 | 6.3 / 8.8 / 17.3 |

**Key finding:** Tool access (browsing) provides ~4 absolute points uplift at high effort for 120B, reflecting the value of external knowledge retrieval on expert-level questions.

### 7.3 Coding and Tool Use Benchmarks

#### 7.3.1 Codeforces (Elo)

| Configuration | 120B | 20B |
|---|---|---|
| No tools (low/med/high) | 1595 / 2205 / 2463 | 1366 / 1998 / 2230 |
| With tools (low/med/high) | 1653 / 2365 / 2622 | 1251 / 2064 / 2516 |

**Comparison:** 120B-high with tools (2622) approaches o3 (2706) and o4-mini (2719).

#### 7.3.2 SWE-Bench Verified

| Model | low/med/high |
|---|---|
| 120B | 47.9 / 52.6 / 62.4 |
| 20B | 37.4 / 53.2 / 60.7 |
| o3 | 69.1 |

Evaluated on fixed subset of $n = 477$ verified tasks. Primary metric: pass@1 without access to ground-truth unit tests.

#### 7.3.3 τ-Bench (Function Calling)

| Benchmark | 120B (low/med/high) | 20B (low/med/high) |
|---|---|---|
| Retail | 49.4 / 62.0 / 67.8 | 35.0 / 47.3 / 54.8 |
| Airline | 42.6 / 48.6 / 49.2 | 32.0 / 42.6 / 38.0 |

### 7.4 Health Evaluation

#### 7.4.1 HealthBench

| Metric | 120B (low/med/high) | 20B (low/med/high) |
|---|---|---|
| HealthBench (Score %) | 53.0 / 55.9 / 57.6 | 40.4 / 41.8 / 42.5 |
| HealthBench Hard | 22.8 / 26.9 / 30.0 | 9.0 / 12.9 / 10.8 |
| HealthBench Consensus | 90.6 / 90.8 / 89.9 | 84.9 / 83.0 / 82.6 |

**Key finding:** 120B-high (57.6) nearly matches o3 (59.8) on HealthBench, substantially outperforms o4-mini (50.1), o3-mini (37.8), o1 (41.8), GPT-4o (32.0). Represents a major Pareto improvement on health performance vs. cost frontier.

### 7.5 Multilingual Evaluation (MMMLU)

14-language professionally translated MMLU. Average across all languages:

| Model | low/med/high |
|---|---|
| 120B | 74.1 / 79.3 / 81.3 |
| 20B | 67.0 / 73.5 / 75.7 |
| o3-mini (high) | 80.7 |
| o4-mini (high) | 85.2 |
| o3 (high) | 88.8 |

**Weakest languages:** Yoruba (120B: 62.4), Swahili (120B: 72.3) — resource-scarce languages show largest gaps.

### 7.6 Hallucination Evaluation

Both evaluated **without browsing**:

| Eval | Metric | 120B | 20B | o4-mini |
|---|---|---|---|---|
| SimpleQA | Accuracy ↑ | 0.168 | 0.067 | 0.234 |
| SimpleQA | Hallucination rate ↓ | 0.782 | 0.914 | 0.750 |
| PersonQA | Accuracy ↑ | 0.298 | 0.155 | 0.356 |
| PersonQA | Hallucination rate ↓ | 0.491 | 0.532 | 0.361 |

**Analysis:** Smaller models hallucinate more due to reduced parametric knowledge. The 20B model's 91.4% SimpleQA hallucination rate is a direct consequence of its 20.91B total parameter budget limiting world knowledge storage. Browsing tool access is expected to substantially mitigate this.

### 7.7 Fairness (BBQ)

| Metric | 120B | 20B | o4-mini |
|---|---|---|---|
| Ambiguous accuracy ↑ | 0.87 | 0.79 | 0.82 |
| Disambiguated accuracy ↑ | 0.90 | 0.89 | 0.95 |

---

## 8. Safety Evaluation and Deployment Constraints

### 8.1 Disallowed Content Performance

**Standard evaluation (metric: `not_unsafe` ↑):**

| Category | 120B | 20B | o4-mini | GPT-4o |
|---|---|---|---|---|
| hate (aggregate) | 0.996 | 0.996 | 0.996 | 0.996 |
| self-harm/intent + instructions | 0.995 | 0.984 | 1.000 | 1.000 |
| personal-data/semi-restrictive | 0.967 | 0.947 | 0.975 | 0.967 |
| sexual/exploitative | 1.000 | 0.980 | 0.990 | 1.000 |
| sexual/minors | 1.000 | 0.971 | 0.990 | 1.000 |
| illicit/non-violent | 1.000 | 0.983 | 0.991 | 0.983 |
| illicit/violent | 1.000 | 1.000 | 1.000 | 1.000 |
| personal-data/restricted | 0.996 | 0.978 | 0.955 | 0.978 |

**Production benchmarks (more challenging, multi-turn):**

GPT-OSS-120B **outperforms** o4-mini on most categories (e.g., harassment/threatening: 0.832 vs 0.695; self-harm/instructions: 0.910 vs 0.901). The 20B model underperforms o4-mini on illicit/violent (0.744 vs 0.845).

### 8.2 Jailbreak Robustness (StrongReject)

| Category | 120B | 20B | o4-mini |
|---|---|---|---|
| illicit/non-violent-crime | 0.979 | 0.960 | 0.980 |
| violence | 0.983 | 0.979 | 0.991 |
| abuse/disinfo/hate | 0.993 | 0.982 | 0.982 |
| sexual-content | 0.989 | 0.970 | 0.974 |

**Finding:** Approximately at parity with o4-mini on known jailbreak techniques.

### 8.3 Instruction Hierarchy

| Evaluation | 120B | 20B | o4-mini |
|---|---|---|---|
| System prompt extraction | 0.832 | 0.881 | 0.993 |
| Prompt injection hijacking | 0.780 | 0.639 | 0.917 |
| Phrase protection (sys/user) | 0.912 | 0.793 | 0.937 |
| Password protection (sys/user) | 0.965 | 0.947 | 0.982 |
| Phrase protection (dev/user) | 0.909 | 0.661 | 0.912 |
| Password protection (dev/user) | 1.000 | 0.946 | 0.947 |

**Finding:** Both GPT-OSS models underperform o4-mini on instruction hierarchy, especially system prompt extraction (120B: 0.832 vs 0.993) and prompt injection hijacking (120B: 0.780 vs 0.917). This is a known weakness of the open-weight release. Developers can fine-tune for improved robustness.

### 8.4 Preparedness Framework Assessment

#### 8.4.1 Tracked Categories

Three categories evaluated: **Biological & Chemical**, **Cybersecurity**, **AI Self-Improvement**.

**Result:** Default GPT-OSS-120B does not reach High capability threshold in any tracked category.

#### 8.4.2 Adversarial Fine-Tuning Methodology

**Threat model:** Technical adversary with strong post-training infrastructure, ML expertise, domain-specific data, and large compute budget.

**Approach:**
1. **Helpful-only RL:** Additional RL stage rewarding compliance with unsafe prompts, removing safety refusals
2. **Domain-specific capability maximization:**
   - Bio: end-to-end browsing training + in-domain human expert biorisk data
   - Cyber: CTF challenge environment training data

**Result:** Even adversarially fine-tuned 120B does not reach High capability in Bio/Chemical or Cyber categories per SAG assessment.

**Open-source baseline comparison (Bio):**

| Evaluation | Adversarial 120B | DeepSeek R1-0528 | Qwen 3 Thinking | Kimi K2 |
|---|---|---|---|---|
| Long-form biorisk | Competitive | Competitive | Competitive | Competitive |
| Multimodal virology troubleshooting | Below o3 | Comparable | Comparable | Comparable |
| ProtocolQA Open-Ended | Below expert consensus (54%) | Below expert | Below expert | Below expert |
| Tacit Knowledge MCQ | Below expert consensus (80%) | Above 80th pctile PhD (63%) | Above 80th pctile | Above 80th pctile |
| TroubleshootingBench | Below 80th pctile human (36.4%) | Below threshold | Below threshold | Below threshold |

**Conclusion:** No single open model consistently outperforms others; GPT-OSS-120B does not significantly advance the frontier of hazardous biological capabilities in open-weight models.

#### 8.4.3 Cybersecurity Evaluation

**CTF Performance (pass@12, 16 rollouts per challenge):**

| Level | 120B (default) | Cybermax (adversarial) | o3 |
|---|---|---|---|
| Collegiate | ~45% | N/A (in training data) | ~55% |
| Professional | ~35% | ~40% (3pp below o3) | ~43% |

**Cyber Range (pass@12, 30 trials, no hints):**

All models score **0%** on all 5 scenarios without hints or solver code. This includes o3, o4-mini, and all GPT-OSS variants. The scenarios require complex multi-step exploitation chains beyond current model capabilities.

#### 8.4.4 AI Self-Improvement

| Benchmark | 120B-high | 20B-high | o3 | o4-mini |
|---|---|---|---|---|
| SWE-Bench Verified | 62.4% | 60.7% | 69.1% | 68.1% |
| OpenAI PRs | ~35% | ~33% | ~37% | ~37% |
| PaperBench | Low | Low | Higher | Higher |

GPT-OSS models do not demonstrate improved performance over proprietary models on self-improvement-relevant benchmarks.

### 8.5 Deployment Constraints and Recommendations

#### 8.5.1 Open-Weight Risk Profile

Open-weight models present a fundamentally different risk profile from API-served models:

- **No revocation:** Once released, OpenAI cannot revoke access or implement additional mitigations
- **Fine-tuning vulnerability:** Adversaries can fine-tune to bypass safety refusals
- **System-level responsibility shift:** Developers deploying GPT-OSS must implement their own system-level safeguards

#### 8.5.2 Chain-of-Thought Monitorability

**Design decision:** No direct optimization pressure on CoT content.

**Rationale:** Preserves CoT as a genuine signal for monitoring misbehavior. Direct pressure on CoT could teach the model to hide reasoning while still misbehaving (steganographic CoT).

**Deployment implication:** CoTs may contain hallucinated or policy-violating content. Developers **must not** expose raw CoTs to end users without:
- Content filtering/moderation
- Summarization
- Safety classification

#### 8.5.3 Inference Provider Requirements

1. **Harmony format compliance:** Proper chat formatting is critical for achieving best capabilities
2. **Multi-turn CoT pruning:** Remove past reasoning traces between conversation turns
3. **Tool harness implementation:** Reference harnesses provided; custom implementations must match tool specification
4. **System message protection:** Instruction hierarchy provides system > developer > user prioritization, but is weaker than o4-mini's implementation—additional fine-tuning may be required
5. **Browsing domain blocklists:** When deploying with browsing for sensitive domains, maintain blocklists to prevent eval contamination and dangerous information retrieval

#### 8.5.4 Memory and Compute Requirements

| Deployment Target | 120B | 20B |
|---|---|---|
| Minimum GPU memory | 80 GB (single H100) | 16 GB |
| Checkpoint size | 60.8 GiB | 12.8 GiB |
| KV cache (full context) | ~2.4 GB | ~1.6 GB |
| Recommended GPU | H100 80GB | RTX 4090 / A6000 |
| Tokens/second (est.) | Limited by memory bandwidth | Higher throughput |

#### 8.5.5 Serving Topology Considerations

- **MoE token routing:** All-to-all dispatch pattern if using expert parallelism across GPUs; single-GPU deployment avoids this overhead but requires full model in memory
- **Continuous batching:** Compatible with standard vLLM/TensorRT-LLM serving; MoE routing creates variable per-token compute depending on expert selection
- **Tail latency:** Expert load imbalance can cause tail latency spikes; request-level expert affinity caching may help
- **Fault tolerance:** Checkpoint at 60.8 GiB enables rapid model reload; stateless serving with KV cache reconstruction on failure

#### 8.5.6 License and Usage

- **License:** Apache 2.0 + GPT-OSS usage policy
- **API compatibility:** Responses API compatible
- **Structured outputs:** Supported
- **Full CoT visibility:** Complete chain-of-thought is accessible (unlike proprietary models)

### 8.6 Deployment Pseudo-Algorithm

```
ALGORITHM: GPT-OSS-Serving
Input: Model checkpoint θ_hat, serving config C
Output: Running inference endpoint

1.  // Model Loading
2.  W ← LoadCheckpoint(θ_hat)
3.  W_MoE ← MXFP4_Dequantize(W.MoE)            // or lazy dequant
4.  W_other ← W.BF16_params
5.  
6.  // Serving Loop
7.  WHILE serving:
8.      request ← ReceiveRequest()
9.      prompt ← FormatHarmony(request, C.system_prompt)
10.     
11.     // Validate harmony format compliance
12.     ASSERT ValidHarmonyFormat(prompt)
13.     
14.     // Multi-turn: strip past CoTs
15.     prompt ← StripPastAnalysisChannels(prompt)
16.     
17.     // Generate
18.     tokens ← []
19.     kv_cache ← Init()
20.     WHILE not EOS and len(tokens) < C.max_tokens:
21.         logits ← Forward(tokens[-1], kv_cache, W)
22.         next ← Sample(logits, C.sampling_params)
23.         tokens.append(next)
24.         
25.         // Handle tool calls
26.         IF IsToolCall(next, tokens):
27.             tool_call ← Parse(tokens)
28.             IF C.tools_enabled AND tool_call.type ∈ C.allowed_tools:
29.                 result ← Execute(tool_call, C.tool_harness)
30.                 tokens.extend(FormatResult(result))
31.     
32.     // Post-processing
33.     response ← ExtractFinalChannel(tokens)
34.     cot ← ExtractAnalysisChannel(tokens)
35.     
36.     // Safety filtering on CoT before any exposure
37.     IF C.expose_cot:
38.         cot ← SafetyFilter(cot)
39.     
40.     SendResponse(response, cot if C.expose_cot)
```

---

## 9. Comprehensive Parameter and Architectural Reference

### 9.1 Complete Parameter Decomposition

**Per-layer parameter breakdown (both models share identical per-layer structure except expert count):**

| Component | Formula | Per-layer (120B) | Per-layer (20B) |
|---|---|---|---|
| $\mathbf{W}_Q$ | $d \times n_q d_h$ | 11.80M | 11.80M |
| $\mathbf{W}_K$ | $d \times n_{kv} d_h$ | 1.47M | 1.47M |
| $\mathbf{W}_V$ | $d \times n_{kv} d_h$ | 1.47M | 1.47M |
| $\mathbf{W}_O$ | $n_q d_h \times d$ | 11.80M | 11.80M |
| Attn biases $b^{(i)}$ | $n_q$ | 64 | 64 |
| Router $\mathbf{W}_r$ | $N_e \times d$ | 0.37M | 0.09M |
| Expert gate $\mathbf{W}_{\text{gate}}$ | $N_e \times d_{ff} \times d$ | $128 \times 8.29\text{M}$ | $32 \times 8.29\text{M}$ |
| Expert up $\mathbf{W}_{\text{up}}$ | $N_e \times d_{ff} \times d$ | $128 \times 8.29\text{M}$ | $32 \times 8.29\text{M}$ |
| Expert down $\mathbf{W}_{\text{down}}$ | $N_e \times d \times d_{ff}$ | $128 \times 8.29\text{M}$ | $32 \times 8.29\text{M}$ |
| RMSNorm gains | $2 \times d$ | 5,760 | 5,760 |

### 9.2 Tensor Flow Diagram (per layer)

$$\mathbf{x} \in \mathbb{R}^{T \times d} \xrightarrow{\text{RMSNorm}} \mathbb{R}^{T \times d} \xrightarrow{\mathbf{W}_{Q/K/V}} \begin{cases} \mathbf{Q} \in \mathbb{R}^{T \times n_q \times d_h} \\ \mathbf{K} \in \mathbb{R}^{T \times n_{kv} \times d_h} \\ \mathbf{V} \in \mathbb{R}^{T \times n_{kv} \times d_h} \end{cases}$$

$$\xrightarrow{\text{RoPE}} \begin{cases} \tilde{\mathbf{Q}} \\ \tilde{\mathbf{K}} \end{cases} \xrightarrow{\text{Attn}(\text{banded/dense})} \mathbf{A} \in \mathbb{R}^{T \times n_q \times d_h} \xrightarrow{\mathbf{W}_O} \mathbb{R}^{T \times d} \xrightarrow{+\text{residual}} \mathbf{h}'$$

$$\mathbf{h}' \xrightarrow{\text{RMSNorm}} \mathbb{R}^{T \times d} \xrightarrow{\mathbf{W}_r} \mathbb{R}^{T \times N_e} \xrightarrow{\text{TopK}} \mathcal{T} \xrightarrow{\text{Expert}_{i \in \mathcal{T}}} \mathbb{R}^{T \times d} \xrightarrow{+\text{residual}} \mathbf{h}^{(\ell)}$$

### 9.3 Critical Implementation Notes

1. **SwiGLU clamping range $C$:** Not specified; likely $C \in [8, 50]$ to prevent overflow in FP4 while maintaining expressivity
2. **SwiGLU residual:** Expert output includes a residual path, possibly $\mathbf{W}_{\text{res}} \in \mathbb{R}^{d \times d}$ or identity mapping — this is "unconventional" per the model card
3. **Attention pattern assignment:** Dense vs. banded alternation starting index not specified; both orderings are valid
4. **YaRN parameters:** $\lambda_{\text{long}}, \lambda_{\text{short}}$, base context $L_{\text{pretrain}}$ not specified; standard YaRN uses $L_{\text{pretrain}} = 4{,}096$ or $8{,}192$ with extension to $L_{\max} = 131{,}072$
5. **Training infrastructure:** Triton kernels from `triton-lang/triton/tree/main/python/triton_kernels` — custom fused kernels for MoE dispatch, attention, and SwiGLU
6. **No weight tying:** Embedding and unembedding matrices are separate (evidenced by counting unembedding toward active but not embedding)