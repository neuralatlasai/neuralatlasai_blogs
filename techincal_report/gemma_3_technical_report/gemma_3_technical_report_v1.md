

# Gemma 3: End-to-End Technical Report

---

## 1. Data Pipeline

<img src="assets/gemma3_architecture_teardown-03.png" alt="Gemma 3 data pipeline showing raw multilingual text and image-text corpora flowing through safety filtering, PII removal, decontamination, near-duplicate removal, quality scoring, SentencePiece tokenization, and offline vision embedding storage" width="100%" />

*Figure. End-to-end data curation flow for Gemma 3, showing how filtering, quality reweighting, multilingual resampling, and offline vision preprocessing combine before multimodal sequence construction.*

### 1.1. Objective

Construct a curated, deduplicated, decontaminated, multilingual, multimodal corpus of text and image tokens sufficient to train dense decoder-only transformers at scales $\{1\text{B}, 4\text{B}, 12\text{B}, 27\text{B}\}$ with token budgets $\{2\text{T}, 4\text{T}, 12\text{T}, 14\text{T}\}$ respectively.

### 1.2. Tokenizer Specification

<img src="assets/gemma3_technical_architecture-04.png" alt="Gemma 3 tokenizer semantics and instruction protocol with raw unicode text, SentencePiece 256K tokenizer, control tokens, and BOS handling warning" width="100%" />

*Figure. Tokenizer and instruction-format semantics, including the SentencePiece 256K tokenizer, preserved whitespace and byte fallback behavior, and the requirement that `[BOS]` be injected structurally rather than tokenized literally.*

- **Type:** SentencePiece (Kudo & Richardson, 2018), shared with Gemini 2.0
- **Vocabulary size:** $|\mathcal{V}| = 262{,}144$ (256K entries)
- **Properties:**
  - Split digits: each digit is an independent token
  - Preserved whitespace: whitespace characters are not normalized
  - Byte-level encodings: fallback to byte-level representation for out-of-vocabulary characters
- **Multilingual balance:** vocabulary construction explicitly balances non-English languages, yielding lower fertility (tokens-per-word) for underrepresented scripts compared to prior Gemma tokenizers

**Formal tokenization mapping:**

$$
\text{Tokenize}: \mathcal{S} \rightarrow \{t_1, t_2, \ldots, t_L\}, \quad t_i \in \{1, 2, \ldots, |\mathcal{V}|\}
$$

where $\mathcal{S}$ is a raw Unicode string and $L$ is the resulting sequence length.

**Control tokens for instruction-tuned (IT) formatting:**

| Token | Semantics |
|---|---|
| `[BOS]` | Beginning of sequence (explicitly added, not from text "[BOS]") |
| `<start_of_turn>user` | Start of user turn |
| `<start_of_turn>model` | Start of model turn |
| `<end_of_turn>` | End of any turn |
| `<eos>` | End of generation (PT models only) |

**Invariant:** `[BOS]` must be prepended after tokenization via `add_bos=True`; tokenizing the string "[BOS]" does **not** yield the `[BOS]` token.

### 1.3. Data Composition and Curation

**Token budgets per model:**

| Model | Token Budget |
|---|---|
| 1B | 2T |
| 4B | 4T |
| 12B | 12T |
| 27B | 14T |

Token counts include both text tokens and image-derived soft tokens. The increase over Gemma 2 budgets accounts for the multimodal mixture.

**Multilingual data augmentation:**

- Both monolingual and parallel corpora are added
- Language imbalance is addressed via a temperature-based resampling strategy inspired by Chung et al. (2023):

$$
p_\ell \propto \left(\frac{n_\ell}{\sum_{\ell'} n_{\ell'}}\right)^{1/T}
$$

where $n_\ell$ is the number of tokens for language $\ell$, and $T$ is a temperature parameter controlling upsampling of low-resource languages ($T > 1$ flattens the distribution).

### 1.4. Filtering Pipeline

**Stages:**

1. **Safety filtering:** Remove content with unwanted or unsafe utterances
2. **PII removal:** Remove certain personal information and other sensitive data
3. **Decontamination:** Remove near-duplicates of evaluation set examples from the pre-training corpus to prevent benchmark leakage
4. **Recitation reduction:** Minimize proliferation of sensitive outputs through deduplication heuristics
5. **Quality reweighting:** Inspired by Sachdeva et al. (2024), assign quality scores to documents and reweight sampling probabilities:

$$
w_d = f_{\text{quality}}(d), \quad p(d) \propto w_d
$$

where $f_{\text{quality}}$ is a learned or heuristic quality scoring function operating at document level $d$.

**Failure modes:**

- Residual contamination despite decontamination (acknowledged explicitly: "there is always a risk of contamination of these probes")
- False negatives in PII detection
- Quality scoring bias toward English or formal text

### 1.5. Image Data Pipeline

<img src="assets/gemma3_technical_architecture-06.png" alt="Offline vision precomputation and spatial compression path from 896 by 896 images through frozen SigLIP, 4 by 4 average pooling, saved compressed tokens, and decoder-only language model training" width="100%" />

*Figure. Offline vision embedding precomputation for Gemma 3, illustrating how SigLIP features are compressed to 256 tokens and removed from training-step compute cost.*

- Images are paired with text in the multimodal corpus
- Each image is processed by the frozen SigLIP encoder during a **pre-computation step**: vision embeddings are extracted offline and stored
- This decouples vision encoder compute from language model training, adding **zero additional cost** to training step time

**Pseudo-Algorithm: Data Preprocessing Pipeline**

```
ALGORITHM: DataPreprocessingPipeline
INPUT: Raw multilingual text corpus C_text, image-text corpus C_img
OUTPUT: Tokenized training shards S

1.  For each document d in C_text ∪ C_img:
2.      Apply safety filter → reject if flagged
3.      Apply PII detector → redact or reject
4.      Compute quality score w_d = f_quality(d)
5.  Deduplicate against evaluation benchmarks (decontamination)
6.  Deduplicate within corpus (near-duplicate removal)
7.  Compute language-specific sampling weights:
        p_ℓ ∝ (n_ℓ / Σ n_ℓ')^(1/T)
8.  For each image in C_img:
9.      Resize to 896×896
10.     Pass through frozen SigLIP encoder
11.     Apply 4×4 average pooling → 256 tokens
12.     Store embedding vectors e_img ∈ R^{256 × d_v}
13. Tokenize all text using SentencePiece tokenizer
14. Interleave image embeddings at appropriate positions
15. Shard into training batches with sequence packing
16. Return S
```

---

## 2. Model Architecture

<img src="assets/gemma3_technical_architecture-02.png" alt="Gemma 3 family topology showing the 1B text-only model and the 4B 12B and 27B multimodal models sharing a frozen SigLIP vision encoder plus token budgets and context lengths" width="100%" />

*Figure. Gemma 3 family overview: a text-only 1B model and three multimodal dense decoders that share a frozen SigLIP encoder, along with their training-token budgets and context behavior.*

### 2.1. Overall Architecture Class

<img src="assets/gemma3_technical_architecture-05.png" alt="Core Gemma 3 decoder and stability stack with text and image embeddings entering a decoder-only transformer and a zoomed decoder block showing pre-RMSNorm, QK-Norm, attention, post-RMSNorm, and FFN" width="100%" />

*Figure. High-level decoder stack and the internal stability structure of a Gemma 3 layer, highlighting pre-norm, QK-norm, post-norm, and the decoder-only multimodal token path.*

**Definition:** Gemma 3 is a family of **decoder-only causal Transformers** with Grouped-Query Attention (GQA), interleaved local/global attention layers, RMSNorm-based normalization, and an optional frozen vision encoder for multimodal variants.

### 2.2. Parameter Counts

| Model | Vision Encoder | Embedding Parameters | Non-Embedding Parameters | Total |
|---|---|---|---|---|
| 1B | 0 | 302M | 698M | 1.0B |
| 4B | 417M | 675M | 3,209M | 4.3B |
| 12B | 417M | 1,012M | 10,759M | 12.2B |
| 27B | 417M | 1,416M | 25,600M | 27.4B |

The 1B model is **text-only** (no vision encoder). The 4B, 12B, and 27B models **share the same 400M SigLIP vision encoder**, which remains frozen throughout training.

### 2.3. Embedding Layer

**Token embedding matrix:**

$$
\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}
$$

where $|\mathcal{V}| = 262{,}144$ and $d_{\text{model}}$ varies by model size.

Input token indices $\{t_1, \ldots, t_L\}$ are mapped to continuous representations:

$$
\mathbf{x}_i = \mathbf{E}[t_i] \in \mathbb{R}^{d_{\text{model}}}
$$

For multimodal inputs, image soft tokens $\mathbf{e}_j^{\text{img}} \in \mathbb{R}^{d_v}$ from SigLIP are projected into the language model's embedding space via a linear projection:

$$
\mathbf{x}_j^{\text{img}} = \mathbf{W}_{\text{proj}} \mathbf{e}_j^{\text{img}} + \mathbf{b}_{\text{proj}}, \quad \mathbf{W}_{\text{proj}} \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

### 2.4. Attention Mechanism

#### 2.4.1. Grouped-Query Attention (GQA)

<img src="assets/gemma3_technical_architecture-07.png" alt="Grouped-query attention architecture with many query heads mapped into fewer shared key value groups and a note about KV cache memory reduction by the n_h over n_g ratio" width="100%" />

*Figure. Grouped-Query Attention in Gemma 3, where multiple query heads share fewer KV groups to reduce cache memory without switching to a sparse-expert architecture.*

**Definition:** GQA partitions $n_h$ query heads into $n_g$ groups, each sharing a single key-value head. This reduces KV-cache memory by factor $n_h / n_g$ while maintaining quality close to full Multi-Head Attention (MHA).

For a given layer, let:
- $\mathbf{Q} \in \mathbb{R}^{L \times n_h \times d_k}$: query projections
- $\mathbf{K} \in \mathbb{R}^{L \times n_g \times d_k}$: key projections (shared across groups)
- $\mathbf{V} \in \mathbb{R}^{L \times n_g \times d_v}$: value projections (shared across groups)

Each query head $h$ in group $g(h) = \lfloor h \cdot n_g / n_h \rfloor$ attends using:

$$
\text{Attn}_h(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}_h \mathbf{K}_{g(h)}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}_{g(h)}
$$

where $\mathbf{M}$ is the attention mask (causal or sliding-window, defined below).

#### 2.4.2. QK-Normalization

Gemma 3 replaces Gemma 2's **soft-capping** mechanism with **QK-norm** (Dehghani et al., 2023; Wortsman et al., 2023; Chameleon Team, 2024):

$$
\hat{\mathbf{Q}}_h = \text{RMSNorm}(\mathbf{Q}_h), \quad \hat{\mathbf{K}}_{g(h)} = \text{RMSNorm}(\mathbf{K}_{g(h)})
$$

$$
\text{Attn}_h = \text{softmax}\!\left(\frac{\hat{\mathbf{Q}}_h \hat{\mathbf{K}}_{g(h)}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}_{g(h)}
$$

**Rationale:** QK-norm stabilizes attention logit magnitudes without the non-differentiable saturation behavior of tanh-based soft-capping, improving training stability and enabling standard kernel optimizations (e.g., FlashAttention compatibility).

#### 2.4.3. Normalization Scheme

Gemma 3 uses both **pre-norm** and **post-norm** with RMSNorm (Zhang & Sennrich, 2019):

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}
$$

where $\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learnable scale parameter and $\epsilon$ is a small constant for numerical stability.

The combination of pre-norm (applied before attention and FFN sub-layers) and post-norm (applied after the residual addition) provides gradient flow stability at initialization while maintaining representation quality at depth.

### 2.5. Local-Global Attention Interleaving

<img src="assets/gemma3_technical_architecture-08.png" alt="Gemma 3 local-global attention interleaving diagram showing a repeating 5 local 1 global pattern and a KV cache memory comparison at 128K context" width="100%" />

*Figure. The 5:1 local-global attention schedule and its KV-cache impact, making the hybrid attention topology and the long-context memory reduction visible at a glance.*

#### 2.5.1. Definition

Gemma 3 introduces a **5:1 local-to-global** attention layer interleaving pattern. Starting from layer $l = 1$:

$$
\text{LayerType}(l) = \begin{cases} \text{Global} & \text{if } l \bmod 6 = 0 \\ \text{Local} & \text{otherwise} \end{cases}
$$

The first layer is **always local**. This means that in a stack of $N$ layers, approximately $\frac{N}{6}$ layers are global and $\frac{5N}{6}$ are local.

#### 2.5.2. Attention Masks

**Global attention mask** (causal):

$$
M_{ij}^{\text{global}} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{otherwise} \end{cases}
$$

**Local sliding window attention mask:**

$$
M_{ij}^{\text{local}} = \begin{cases} 0 & \text{if } i - w < j \leq i \\ -\infty & \text{otherwise} \end{cases}
$$

where $w = 1024$ is the sliding window size for Gemma 3.

#### 2.5.3. KV-Cache Memory Analysis

For a sequence of length $L$, with $n_g$ KV heads, head dimension $d_k$, and precision $b$ bytes per element:

**Global-only KV-cache (standard transformer):**

$$
\text{KV}_{\text{global-only}} = 2 \cdot N \cdot n_g \cdot d_k \cdot L \cdot b
$$

**Gemma 3 hybrid KV-cache:**

$$
\text{KV}_{\text{Gemma3}} = 2 \cdot n_g \cdot d_k \cdot b \cdot \left(\frac{N}{6} \cdot L + \frac{5N}{6} \cdot \min(w, L)\right)
$$

For $L \gg w$:

$$
\text{KV}_{\text{Gemma3}} \approx 2 \cdot n_g \cdot d_k \cdot b \cdot N \cdot \left(\frac{L}{6} + \frac{5w}{6}\right)
$$

**Compression ratio:**

$$
\rho = \frac{\text{KV}_{\text{Gemma3}}}{\text{KV}_{\text{global-only}}} = \frac{1}{6} + \frac{5w}{6L}
$$

For $L = 128\text{K}$ and $w = 1024$:

$$
\rho = \frac{1}{6} + \frac{5 \times 1024}{6 \times 131072} \approx 0.1667 + 0.0065 \approx 0.173
$$

This yields approximately **5.8× reduction** in KV-cache memory versus a global-only transformer.

**Empirical validation (from ablation, Figure 5):** At 32K context, global-only incurs ~60% memory overhead from KV-cache alone, while 1:3 local:global with $w=1024$ reduces this to <15%.

#### 2.5.4. Ablation Results

**Local:Global ratio impact on perplexity (Figure 3):**

| Ratio | 2B Δ Perplexity | 9B Δ Perplexity |
|---|---|---|
| 1:1 (Gemma 2 default) | baseline | baseline |
| 3:1 | ≈ 0.0 | ≈ 0.0 |
| 5:1 (Gemma 3 default) | ≈ 0.0 | ≈ 0.0 |
| 7:1 | ≈ +0.05 | ≈ +0.03 |

**Sliding window size impact (Figure 4):**

| Window Size | 2B (L:G=1:1) Δ PPL | 2B (L:G=3:1) Δ PPL |
|---|---|---|
| 512 | ≈ +0.01 | ≈ +0.02 |
| 1024 | ≈ 0.0 | ≈ 0.0 |
| 2048 | ≈ −0.005 | ≈ −0.005 |
| 4096 | baseline | baseline |

**Key finding:** Sliding window can be reduced to 1024 with negligible perplexity degradation, while yielding substantial KV-cache savings.

### 2.6. Positional Encoding: RoPE

<img src="assets/gemma3_technical_architecture-09.png" alt="Dual RoPE configuration for Gemma 3 showing local layers with base 10K and global layers with base 1M plus rescaling by s equals 8 for long-context extension to 128K" width="100%" />

*Figure. Dual-frequency RoPE configuration in Gemma 3, showing how local and global layers use different bases and how only the global branch is rescaled for 128K extension.*

#### 2.6.1. Standard RoPE

Rotary Position Embedding (RoPE) applies position-dependent rotation to query and key vectors:

$$
\text{RoPE}(\mathbf{x}, m) = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \end{pmatrix} + \begin{pmatrix} -x_2 \\ x_1 \\ -x_4 \\ x_3 \\ \vdots \end{pmatrix} \odot \begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \end{pmatrix}
$$

where:

$$
\theta_k = \frac{1}{\beta^{2k/d_k}}, \quad k = 1, \ldots, d_k/2
$$

and $\beta$ is the RoPE base frequency, $m$ is the token position.

#### 2.6.2. Gemma 3 RoPE Configuration

| Layer Type | RoPE Base Frequency $\beta$ |
|---|---|
| Global self-attention | $1{,}000{,}000$ (1M) |
| Local self-attention | $10{,}000$ (10K) |

**Rationale:** 
- Global layers must represent relative positions across the full 128K context; high $\beta$ ensures that rotation angles $m\theta_k$ vary slowly, preventing aliasing at large position differences
- Local layers only attend within a 1024-token window; standard $\beta = 10\text{K}$ provides sufficient resolution within this span

#### 2.6.3. Long-Context Extension via RoPE Rescaling

Models are pre-trained with 32K context, then extended to 128K via positional interpolation (Chen et al., 2023):

$$
\theta_k^{\text{extended}} = \frac{\theta_k}{s}, \quad s = \frac{L_{\text{target}}}{L_{\text{train}}} = \frac{128\text{K}}{32\text{K}} = 4
$$

However, empirically a **scaling factor of $s=8$** is found to work best. This is applied only to global self-attention layers.

**Result (Figure 7):** After RoPE rescaling, perplexity remains stable up to 128K tokens but degrades rapidly beyond this length, indicating that the effective context window is bounded by the interpolation range.

### 2.7. Vision Encoder

#### 2.7.1. Architecture

- **Base model:** SigLIP (Sigmoid Loss for Image-Language Pre-training), 400M parameter variant
- **Backbone:** Vision Transformer (ViT) (Dosovitskiy, 2020)
- **Training objective:** Contrastive loss variant of CLIP with sigmoid activation instead of softmax:

$$
\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log\sigma(y_i \cdot z_i) \right]
$$

where $y_i \in \{-1, +1\}$ indicates whether image-text pair $i$ is matched, $z_i = \tau \cdot \mathbf{v}_i^\top \mathbf{t}_i$ is the scaled cosine similarity, and $\sigma$ is the sigmoid function.

- **Input resolution:** $896 \times 896$ pixels (square)
- **Patch size:** $p \times p$ (standard ViT patching)
- **Raw output token count:** Depends on $p$; for 896 resolution with patch size 14: $(896/14)^2 = 4096$ tokens

#### 2.7.2. Output Compression via Average Pooling

The vision encoder output is compressed to a **fixed 256 tokens** regardless of resolution:

$$
\mathbf{E}_{\text{img}} \in \mathbb{R}^{G^2 \times d_v} \xrightarrow{\text{AvgPool}_{k \times k}} \mathbf{E}_{\text{compressed}} \in \mathbb{R}^{256 \times d_v}
$$

For 896 resolution with 4096 raw tokens:

$$
k = \sqrt{4096 / 256} = 4, \quad \text{i.e., } 4 \times 4 \text{ average pooling}
$$

**Information preservation analysis:** Average pooling is a linear operator:

$$
\mathbf{e}_j^{\text{compressed}} = \frac{1}{k^2} \sum_{(u,v) \in \mathcal{P}_j} \mathbf{e}_{u,v}
$$

where $\mathcal{P}_j$ is the $k \times k$ spatial patch assigned to output position $j$. This preserves the mean signal within each patch but discards spatial high-frequency information within the pooling window.

#### 2.7.3. Impact of Resolution (Ablation, Table 7)

| Resolution | DocVQA | InfoVQA | TextVQA |
|---|---|---|---|
| 256 | 31.9 | 23.1 | 44.1 |
| 448 | 45.4 | 31.6 | 53.5 |
| 896 | 59.8 | 33.7 | 58.0 |

Higher resolution is strictly beneficial, especially for document and text-recognition tasks.

### 2.8. Pan & Scan (P&S)

<img src="assets/gemma3_architecting-07.png" alt="Dynamic spatial inference Pan and Scan with adaptive crop grid, SigLIP encoding per crop, and concatenation into a flat sequence of image tokens" width="100%" />

*Figure. Pan and Scan at inference time: non-square images are tiled into aspect-ratio-aware crops, each resized to 896 by 896 and encoded into a bounded visual token sequence.*

#### 2.8.1. Definition

An **inference-time adaptive windowing algorithm** that segments non-square or high-resolution images into multiple non-overlapping crops of equal size, each resized to $896 \times 896$ for encoder processing.

#### 2.8.2. Algorithm

```
ALGORITHM: PanAndScan
INPUT: Image I of size (H, W), max_crops N_max
OUTPUT: Set of encoded image token sequences {E_1, ..., E_C}

1.  Compute aspect ratio r = W / H
2.  IF r ≈ 1.0 AND max(H,W) ≤ 896:
3.      Resize I to 896×896 → single crop
4.      C = 1
5.  ELSE:
6.      Determine grid (n_h, n_w) such that:
          n_h × n_w ≤ N_max
          n_w / n_h ≈ r (preserve aspect ratio)
7.      Partition I into n_h × n_w non-overlapping tiles
8.      For each tile t_k:
9.          Resize t_k to 896×896
10.     C = n_h × n_w
11. For each crop c_k, k = 1..C:
12.     E_k = SigLIP(c_k) ∈ R^{256 × d_v}
13. Return {E_1, ..., E_C}
```

**Total image tokens per image:**

$$
L_{\text{img}} = C \times 256
$$

where $C \leq N_{\text{max}}$.

#### 2.8.3. Impact (Table 8)

| Model | DocVQA | DocVQA (P&S) | Δ | InfoVQA | InfoVQA (P&S) | Δ |
|---|---|---|---|---|---|---|
| 4B | 72.8 | 81.0 | +8.2 | 44.1 | 57.0 | +12.9 |
| 27B | 85.6 | 90.4 | +4.8 | 59.4 | 76.4 | +17.0 |

P&S is critical for tasks requiring text reading on images and variable aspect ratio handling.

**Properties:**
- **Inference-time only:** Not used during training; can be disabled for faster inference
- **No retraining required:** The frozen SigLIP encoder generalizes to crops
- **Controllable compute:** Maximum number of crops bounds inference cost

---

## 3. Optimization Strategy

<img src="assets/gemma3_technical_architecture-10.png" alt="Top 256 sparse knowledge distillation for Gemma 3 with teacher full vocabulary, top-k support, renormalization, and student loss over sampled logits" width="100%" />

*Figure. Gemma 3 pre-training distillation compresses teacher supervision to the top 256 logits per position, preserving most teacher probability mass while drastically reducing target storage.*

### 3.1. Pre-Training Objective: Knowledge Distillation

#### 3.1.1. Formal Objective

Let $\theta_S$ denote student parameters, $\theta_T$ denote (frozen) teacher parameters. For each token position $i$ in a training sequence:

1. **Teacher probability computation:** The teacher produces a full distribution $p_T(\cdot | \mathbf{x}_{<i})$ over $|\mathcal{V}|$ tokens
2. **Top-$K$ sampling:** Select the top $K = 256$ tokens by teacher probability:

$$
\mathcal{T}_i = \text{top-}K\left(p_T(\cdot | \mathbf{x}_{<i}), K=256\right)
$$

3. **Renormalization:** Construct a truncated teacher distribution:

$$
\tilde{p}_T(v | \mathbf{x}_{<i}) = \begin{cases} \frac{p_T(v | \mathbf{x}_{<i})}{\sum_{v' \in \mathcal{T}_i} p_T(v' | \mathbf{x}_{<i})} & \text{if } v \in \mathcal{T}_i \\ 0 & \text{otherwise} \end{cases}
$$

4. **Distillation loss:** Cross-entropy between the student's distribution (restricted to sampled logits) and the renormalized teacher distribution:

$$
\mathcal{L}_{\text{distill}} = -\sum_{i=1}^{L} \sum_{v \in \mathcal{T}_i} \tilde{p}_T(v | \mathbf{x}_{<i}) \log p_S(v | \mathbf{x}_{<i}; \theta_S)
$$

#### 3.1.2. Properties

- **Memory efficiency:** Only 256 logits per token are materialized, versus $262{,}144$ for full cross-entropy, yielding a $\sim 1024\times$ reduction in target storage
- **Approximation quality:** Teacher probability mass on the top-256 tokens is typically $> 0.99$ for well-trained teachers on natural language, so the truncation error is negligible:

$$
\sum_{v \in \mathcal{T}_i} p_T(v | \mathbf{x}_{<i}) \approx 1
$$

- **Gradient flow:** The student receives gradients only on the 256 sampled logits per position, but the teacher's distribution within this support is preserved exactly

#### 3.1.3. Small vs. Large Teacher (Ablation, Figure 8)

| Training Horizon | Better Teacher |
|---|---|
| Short ($< 10$B tokens) | Smaller teacher |
| Long ($> 100$B tokens) | Larger teacher |

**Explanation:** At short horizons, the regularization effect of a weaker teacher (less peaked distribution, more entropy) acts as beneficial noise injection. At long horizons, the superior signal from a better teacher dominates, and the student has sufficient capacity and data to exploit it.

### 3.2. Optimizer and Sharding

#### 3.2.1. Optimizer State Sharding

ZeRO-3 (Ren et al., 2021) is used to shard optimizer state, gradients, and parameters across data-parallel ranks:

- **Stage 1:** Optimizer state partitioned across $D$ ranks → memory per rank: $\mathcal{O}(P/D)$ for optimizer states
- **Stage 2:** Gradients additionally sharded
- **Stage 3:** Parameters additionally sharded

Total memory per device:

$$
M_{\text{per\_device}} = \frac{|\theta| + |\nabla\theta| + |\text{opt\_state}|}{D} + M_{\text{activations}}
$$

where $D$ is the number of data-parallel ranks.

#### 3.2.2. Multi-Pod Training

For multi-pod configurations, **data replica reduction** is performed over the data center network using the Pathways system (Barham et al., 2022):

- All-reduce of gradients is performed hierarchically: intra-pod via high-bandwidth interconnect, inter-pod via data center network
- The GSPMD partitioner (Xu et al., 2021) handles tensor sharding annotations
- MegaScale XLA compiler (XLA, 2019) generates optimized HLO programs

### 3.3. Compute Infrastructure

<img src="assets/gemma3_technical_architecture-11.png" alt="Distributed training infrastructure and topology for Gemma 3 with TPU counts by model size, logical sharding axes, ZeRO-3 partitioning, and intra-pod and inter-pod reduction" width="100%" />

*Figure. Training topology for Gemma 3 across TPU generations, showing how model scale maps onto hardware size, logical sharding, ZeRO-3 partitioning, and hierarchical reduction.*

| Model | Accelerator | #Chips | Data Shards | Seq. Shards | Replica |
|---|---|---|---|---|---|
| 1B | TPUv5e | 512 | 16 | 16 | 2 |
| 4B | TPUv5e | 2,048 | 16 | 16 | 8 |
| 12B | TPUv4 | 6,144 | 16 | 16 | 24 |
| 27B | TPUv5p | 6,144 | 24 | 8 | 32 |

**Sharding dimensions:**
- **Data shards:** Batch dimension partitioned
- **Sequence shards:** Sequence length dimension partitioned (sequence parallelism)
- **Replica:** Parameter replicas for fault tolerance and throughput

**Vision encoder compute:** Pre-computed embeddings eliminate vision encoder FLOPs from training step time entirely.

---

## 4. Training Stages

<img src="assets/gemma3_technical_architecture-12.png" alt="End-to-end Gemma 3 training lifecycle from short-context pre-training through long-context extension, alignment, and deployment preparation with checkpoint outputs" width="100%" />

*Figure. The Gemma 3 training lifecycle from 32K pre-training through long-context extension, instruction tuning, RL fine-tuning, and quantization-aware deployment preparation.*

### 4.1. Stage 1: Pre-Training

#### 4.1.1. Phase 1: Short-Context Pre-Training

- **Context length:** 32K tokens
- **RoPE base frequency:** 1M (global), 10K (local)
- **Data:** Mixed text + pre-computed image embeddings
- **Objective:** $\mathcal{L}_{\text{distill}}$ (Section 3.1)
- **Duration:** Bulk of the token budget (majority of 2T–14T tokens)

#### 4.1.2. Phase 2: Long-Context Extension

- **Context length:** Extended to 128K tokens (for 4B, 12B, 27B models)
- **RoPE rescaling:** Frequencies divided by scaling factor $s = 8$:

$$
\theta_k^{\text{ext}} = \frac{\theta_k}{8}
$$

- **Applicable only to global attention layers**
- **1B model:** Remains at 32K context (no extension)
- **Continuation training** on long-context data to adapt to the new positional encoding

**Convergence dynamics (Figure 7):**
- Pre-rescaling: perplexity degrades sharply beyond 32K
- Post-rescaling: perplexity stable up to 128K
- Beyond 128K: rapid degradation (extrapolation failure)

### 4.2. Stage 2: Instruction Tuning (Post-Training)

#### 4.2.1. Supervised Fine-Tuning (SFT) via Knowledge Distillation

- **Teacher:** Large instruction-tuned (IT) teacher model
- **Method:** Improved knowledge distillation (Agarwal et al., 2024; Anil et al., 2018; Hinton et al., 2015)
- **Data curation for SFT:**
  - Filter examples containing personal information
  - Remove unsafe/toxic model outputs
  - Remove mistaken self-identification data
  - Deduplicate examples
  - Include subsets encouraging in-context attribution, hedging, and appropriate refusals

**Objective:** Same truncated distillation loss as pre-training, but on IT-formatted data:

$$
\mathcal{L}_{\text{SFT}} = -\sum_{i \in \text{model\_turn}} \sum_{v \in \mathcal{T}_i} \tilde{p}_{T_{\text{IT}}}(v | \mathbf{x}_{<i}) \log p_S(v | \mathbf{x}_{<i}; \theta_S)
$$

Loss is computed **only on model turn tokens** (masked for user turns and control tokens).

#### 4.2.2. Reinforcement Learning Fine-Tuning

<img src="assets/gemma3_technical_architecture-13.png" alt="Gemma 3 RL alignment stack showing BOND best-of-n candidate selection, WARM reward model averaging, and WARP policy averaging across RL checkpoints" width="100%" />

*Figure. RL alignment stack for Gemma 3, showing how BOND, WARM, and WARP cooperate to select strong responses, regularize reward models, and stabilize policy updates.*

**Methods used:**

1. **BOND** (Sessa et al., 2024) — Best-of-N Distillation: Generate $N$ responses, select the best according to a reward model, distill from the selected response

2. **WARM** (Ramé et al., 2024b) — Weight Averaged Reward Models: Ensemble of reward models via weight averaging in parameter space to reduce reward hacking

3. **WARP** (Ramé et al., 2024a) — Weight Averaged Rewarded Policies: Average policy weights across RL checkpoints to stabilize training

**Formal BOND Objective:**

Given prompt $\mathbf{x}$, generate $N$ completions $\{\mathbf{y}^{(1)}, \ldots, \mathbf{y}^{(N)}\}$ from the current policy $\pi_\theta$:

$$
\mathbf{y}^* = \arg\max_{\mathbf{y}^{(k)}} R(\mathbf{x}, \mathbf{y}^{(k)})
$$

Then update:

$$
\mathcal{L}_{\text{BOND}} = -\sum_i \log \pi_\theta(y^*_i | \mathbf{x}, \mathbf{y}^*_{<i})
$$

**WARM Reward Aggregation:**

$$
R_{\text{WARM}}(\mathbf{x}, \mathbf{y}) = R_{\bar{\phi}}(\mathbf{x}, \mathbf{y}), \quad \bar{\phi} = \frac{1}{M}\sum_{m=1}^{M} \phi_m
$$

where $\{\phi_m\}_{m=1}^M$ are independently trained reward model parameters. Weight averaging in parameter space rather than output space provides implicit regularization against reward model overoptimization.

**WARP Policy Aggregation:**

$$
\theta_{\text{WARP}} = \frac{1}{J}\sum_{j=1}^{J} \theta_j^{\text{RL}}
$$

where $\theta_j^{\text{RL}}$ are policy parameters at different RL checkpoints or from different RL runs. This reduces variance and mitigates policy collapse.

#### 4.2.3. Reward Functions

| Reward Type | Signal Source | Target Ability |
|---|---|---|
| Human preference RM | Human feedback data | Helpfulness, chat quality |
| Code execution feedback | Execution environment (Gehring et al., 2024) | Coding |
| Ground-truth math rewards | Verifiable solutions (DeepSeek-AI, 2025; Lambert et al., 2024) | Mathematics |
| Safety RM | Safety annotations | Harm reduction |
| Weight-averaged RM (WARM) | Ensemble of above | Robustness against reward hacking |

**Multi-objective reward formulation:**

$$
R_{\text{total}}(\mathbf{x}, \mathbf{y}) = \sum_{c} \alpha_c R_c(\mathbf{x}, \mathbf{y})
$$

where $c$ indexes capability domains (helpfulness, math, coding, safety, instruction-following, multilingual) and $\alpha_c$ are domain-specific weights.

#### 4.2.4. Post-Training Data Filtering

```
ALGORITHM: PostTrainingDataFiltering
INPUT: Candidate SFT/RL dataset D
OUTPUT: Filtered dataset D_filtered

1.  For each example (x, y) in D:
2.      IF contains_personal_information(y): REJECT
3.      IF is_unsafe_or_toxic(y): REJECT
4.      IF is_mistaken_self_identification(y): REJECT
5.      IF is_near_duplicate(y, D_seen): REJECT
6.      D_seen ← D_seen ∪ {y}
7.  Add examples encouraging:
        - In-context attribution
        - Hedging on uncertain claims
        - Appropriate refusals
8.  Return D_filtered
```

**Effect:** Including hedging/attribution data improves factuality metrics without degrading other capabilities.

### 4.3. Stage 3: Quantization-Aware Training (QAT)

<img src="assets/gemma3_technical_architecture-14.png" alt="Gemma 3 deployment formats and quantization trade-offs showing simulated quantize-dequantize QAT branching to per-channel int4, per-block int4, and SFP8" width="100%" />

*Figure. Quantization-aware training and deployment-format trade-offs in Gemma 3, contrasting per-channel Int4, per-block Int4, and SFP8 against the frozen full-precision teacher target.*

#### 4.3.1. Objective

Fine-tune full-precision checkpoints for a small number of steps ($\sim 5000$) with simulated quantization in the forward pass, using the non-quantized checkpoint's output probabilities as targets.

#### 4.3.2. Quantization Schemes

| Format | Description |
|---|---|
| Per-channel Int4 | 4-bit integer quantization with per-output-channel scale/zero-point |
| Per-block Int4 (block=32) | 4-bit integer quantization with scale/zero-point per block of 32 elements |
| Switched FP8 (SFP8) | 8-bit floating point with dynamic exponent selection |

#### 4.3.3. QAT Forward Pass

For a weight tensor $\mathbf{W}$, the quantized version during training is:

$$
\hat{\mathbf{W}} = \text{Dequant}(\text{Quant}(\mathbf{W}))
$$

**Per-channel Int4:**

$$
\text{Quant}(\mathbf{W}_{c,:}) = \text{clamp}\!\left(\left\lfloor \frac{\mathbf{W}_{c,:}}{s_c} \right\rceil + z_c, \; 0, \; 15\right)
$$

$$
s_c = \frac{\max(\mathbf{W}_{c,:}) - \min(\mathbf{W}_{c,:})}{2^4 - 1}, \quad z_c = -\left\lfloor \frac{\min(\mathbf{W}_{c,:})}{s_c} \right\rceil
$$

**Per-block Int4 (block size $B=32$):**

$$
\text{Quant}(\mathbf{W}_{c, bB:(b+1)B}) = \text{clamp}\!\left(\left\lfloor \frac{\mathbf{W}_{c, bB:(b+1)B}}{s_{c,b}} \right\rceil + z_{c,b}, \; 0, \; 15\right)
$$

Per-block quantization introduces more scale parameters (one per block of 32 elements) but provides finer granularity.

**Straight-Through Estimator (STE) for gradient propagation:**

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{W}}}
$$

The quantization function is non-differentiable; STE passes gradients through as if quantization were the identity function.

#### 4.3.4. QAT Loss

$$
\mathcal{L}_{\text{QAT}} = -\sum_{i=1}^{L} \sum_{v=1}^{|\mathcal{V}|} p_{\text{FP}}(v | \mathbf{x}_{<i}; \theta_{\text{FP}}) \log p_{\text{Q}}(v | \mathbf{x}_{<i}; \hat{\theta})
$$

where $p_{\text{FP}}$ is the full-precision checkpoint's output distribution (frozen target) and $p_{\text{Q}}$ is the quantized model's output. Data distribution matches both pre-training and post-training distributions.

#### 4.3.5. Memory Footprints (Table 3)

| Model | BF16 (GB) | Int4 (GB) | Int4-B32 (GB) | SFP8 (GB) | BF16+KV@32K (GB) | Int4+KV@32K (GB) |
|---|---|---|---|---|---|---|
| 1B | 2.0 | 0.5 | 0.7 | 1.0 | 2.9 | 1.4 |
| 4B | 8.0 | 2.6 | 2.9 | 4.4 | 12.7 | 7.3 |
| 12B | 24.0 | 6.6 | 7.1 | 12.4 | 38.9 | 21.5 |
| 27B | 54.0 | 14.1 | 15.3 | 27.4 | 72.7 | 32.8 |

KV-cache is quantized to 8 bits in the "+KV" configurations.

**Weight compression ratio (Int4 vs BF16):**

$$
\rho_{\text{weight}} = \frac{4}{16} = 0.25 \quad (\text{4× compression})
$$

**Including metadata overhead (per-block Int4, block=32):**

$$
\rho_{\text{weight+meta}} = \frac{4 + 16/32}{16} = \frac{4.5}{16} = 0.28125
$$

---

## 5. Compression Pipeline

### 5.1. Vision Embedding Compression

#### 5.1.1. Raw-to-Compressed Information Flow

$$
\text{Image} \in \mathbb{R}^{896 \times 896 \times 3} \xrightarrow{\text{SigLIP}} \mathbf{E}_{\text{raw}} \in \mathbb{R}^{G^2 \times d_v} \xrightarrow{\text{AvgPool}_{k \times k}} \mathbf{E}_{\text{comp}} \in \mathbb{R}^{256 \times d_v} \xrightarrow{\mathbf{W}_{\text{proj}}} \mathbf{X}_{\text{img}} \in \mathbb{R}^{256 \times d_{\text{model}}}
$$

**Compression factor (spatial):**

$$
\text{CF}_{\text{spatial}} = \frac{G^2}{256} = \frac{4096}{256} = 16 \quad \text{(for 896 resolution)}
$$

**Total compression from raw pixels:**

$$
\text{CF}_{\text{total}} = \frac{896 \times 896 \times 3}{256 \times d_v}
$$

### 5.2. KV-Cache Compression

As derived in Section 2.5.3:

$$
\text{CF}_{\text{KV}} = \frac{1}{\rho} = \frac{6L}{L + 5w} \approx 5.8 \text{ at } L = 128\text{K}, w = 1024
$$

### 5.3. Weight Quantization Compression

See Section 4.3.5.

### 5.4. Information Preservation Invariants

<img src="assets/gemma3_architecture-15.png" alt="Gemma 3 information compression invariants showing spatial compression, temporal KV cache compression, precision compression, and their combined effect on parameter efficiency" width="100%" />

*Figure. Gemma 3 compression synthesis, tying together visual-token pooling, hybrid KV-cache reduction, and quantized deployment as one coordinated information-preservation strategy.*

1. **Vision compression:** Average pooling preserves first moment of patch activations; higher spatial frequencies are lost. For tasks requiring fine-grained spatial detail, P&S mitigates this by providing multiple crops at full resolution.

2. **KV-cache compression (local/global):** No information is discarded from global layers; local layers retain full causal attention within their window. Tokens beyond the window boundary in local layers have **zero attention** — information from those positions is available only through global layers.

3. **Weight quantization:** QAT ensures that the quantized model's output distribution approximates the full-precision model's distribution via distillation:

$$
D_{\text{KL}}\!\left(p_{\text{FP}} \| p_{\text{Q}}\right) \leq \epsilon_{\text{QAT}}
$$

where $\epsilon_{\text{QAT}}$ is minimized over $\sim 5000$ fine-tuning steps.

---

## 6. Inference Path

### 6.1. End-to-End Inference Pipeline

```
ALGORITHM: Gemma3Inference
INPUT: Text tokens T = [t_1, ..., t_L_text], Images I = [I_1, ..., I_M]
OUTPUT: Generated token sequence Y

// Phase 1: Vision Encoding (if multimodal)
1.  For each image I_m:
2.      Apply P&S → crops {c_1, ..., c_C}
3.      For each crop c_k:
4.          E_k = SigLIP(Resize(c_k, 896×896))  // R^{256 × d_v}
5.      E_m = Concat(E_1, ..., E_C)  // R^{(256·C) × d_v}
6.      X_m^img = W_proj · E_m  // R^{(256·C) × d_model}

// Phase 2: Construct Input Sequence
7.  Interleave text embeddings X^text and image embeddings X^img
8.  Prepend [BOS] token embedding
9.  Full input X = [x_BOS, x_1, ..., x_L] ∈ R^{L_total × d_model}

// Phase 3: Prefill (Parallel)
10. Initialize KV-cache:
        For global layers: allocate full-length KV storage
        For local layers: allocate sliding window buffer of size w=1024
11. Forward pass through all N layers:
        For layer l = 1 to N:
            Apply RMSNorm (pre-norm)
            IF LayerType(l) == Global:
                Compute GQA with causal mask over full sequence
                Apply RoPE with β = 1M
                Store K, V in global KV-cache
            ELSE:  // Local
                Compute GQA with sliding window mask (w=1024)
                Apply RoPE with β = 10K
                Store K, V in circular buffer of size w
            Apply QK-norm before softmax
            Apply RMSNorm (post-norm)
            Apply FFN
            Residual connection

// Phase 4: Autoregressive Decoding
12. REPEAT:
13.     Compute next-token logits from last layer output
14.     Sample or greedy-select: y_t = argmax p(· | x_{<t})
15.     Update KV-cache:
            Global layers: append new K, V
            Local layers: overwrite oldest entry in circular buffer
16.     IF y_t == <end_of_turn> or y_t == <eos>: BREAK
17. Return Y = [y_1, ..., y_T]
```

### 6.2. KV-Cache Memory During Inference

For the 27B model in BF16 at 128K context:

**Global layers:**
- Number: $N_g = N/6$
- KV per layer: $2 \times n_g \times d_k \times L \times 2$ bytes (BF16)

**Local layers:**
- Number: $N_l = 5N/6$
- KV per layer: $2 \times n_g \times d_k \times w \times 2$ bytes (BF16)

**Total:**

$$
\text{KV}_{\text{total}} = 2 \cdot n_g \cdot d_k \cdot 2 \cdot \left(\frac{N}{6} \cdot L + \frac{5N}{6} \cdot w\right) \; \text{bytes}
$$

From Table 3, at 32K context the 27B model has 18.7 GB KV-cache overhead ($72.7 - 54.0$) in BF16.

### 6.3. Decoding Complexity

**Per-token cost in autoregressive decoding:**

- **Global attention layer:** $\mathcal{O}(n_g \cdot d_k \cdot L)$ — scales linearly with sequence length
- **Local attention layer:** $\mathcal{O}(n_g \cdot d_k \cdot w)$ — constant with respect to sequence length

**Aggregate per-token cost:**

$$
\text{FLOPs}_{\text{per\_token}} = \underbrace{\frac{N}{6} \cdot \mathcal{O}(n_g \cdot d_k \cdot L)}_{\text{global layers}} + \underbrace{\frac{5N}{6} \cdot \mathcal{O}(n_g \cdot d_k \cdot w)}_{\text{local layers}} + \underbrace{N \cdot \mathcal{O}(d_{\text{model}}^2)}_{\text{FFN}}
$$

For $L \gg w$, the global layers dominate attention cost, but since only $1/6$ of layers are global, the effective attention FLOPs are reduced by $\sim 6\times$ compared to all-global transformers.

---

## 7. Evaluation Protocol

### 7.1. Pre-Training Probes

Monitored during pre-training across six categories:
- **Science** (STEM knowledge benchmarks)
- **Code** (code generation/completion)
- **Factuality** (factual accuracy)
- **Multilinguality** (non-English language understanding)
- **Reasoning** (logical/mathematical reasoning)
- **Vision** (visual question answering, new to Gemma 3)

**Decontamination caveat:** Despite decontamination, residual probe contamination risk exists (Mirzadeh et al., 2024), limiting definitive conclusions from pre-training probes.

### 7.2. Instruction-Tuned Model Benchmarks (Table 6)

| Benchmark | Measures | Metric Type |
|---|---|---|
| MMLU-Pro | Broad knowledge, reasoning | Accuracy |
| LiveCodeBench | Live coding tasks | Pass rate |
| Bird-SQL (dev) | SQL generation | Execution accuracy |
| GPQA Diamond | Graduate-level QA | Accuracy |
| SimpleQA | Simple factual QA | Accuracy |
| FACTS Grounding | Factual grounding | Score |
| Global MMLU-Lite | Multilingual knowledge | Accuracy |
| MATH | Mathematical problem solving | Accuracy |
| HiddenMath | Hidden/novel math problems | Accuracy |
| MMMU (val) | Multimodal multitask understanding | Accuracy |

### 7.3. Key Performance Comparisons

<img src="assets/gemma3_architecture-03.png" alt="Capability and performance diagnostics comparing Gemma 3 27B IT against Gemma 2 27B IT and Gemini 1.5 models across MMLU-Pro, MATH, HiddenMath, MMMU, and LMSYS Elo" width="100%" />

*Figure. High-signal performance comparison for Gemma 3 27B IT across reasoning, multimodal understanding, and live preference benchmarks.*

**Gemma 3 27B IT vs. predecessors and peers:**

| Benchmark | Gemma 2 27B IT | Gemma 3 27B IT | Gemini 1.5 Flash | Gemini 1.5 Pro |
|---|---|---|---|---|
| MMLU-Pro | 56.9 | 67.5 | 67.3 | 75.8 |
| MATH | 55.6 | 89.0 | 77.9 | 86.5 |
| HiddenMath | 14.8 | 60.3 | 47.2 | 52.0 |
| MMMU (val) | — | 64.9 | 62.3 | 65.9 |
| FACTS Grounding | 62.4 | 74.9 | 82.9 | 80.0 |

**Critical observation:** Gemma 3 27B IT achieves MATH score of 89.0, surpassing both Gemini 1.5 Flash (77.9) and Gemini 1.5 Pro (86.5). HiddenMath improvement from 14.8 → 60.3 represents a **4× gain**, directly attributable to the RL fine-tuning with ground-truth math rewards.

### 7.4. LMSYS Chatbot Arena (Table 5)

- **Gemma 3 27B IT Elo:** 1338 (preliminary, March 8, 2025)
- **Rank:** Top 10 overall
- **Comparison with open models:**

| Model | Elo | Size | Type |
|---|---|---|---|
| DeepSeek-R1 | 1363 | 671B/37B activated | MoE |
| **Gemma 3 27B IT** | **1338** | **27B** | **Dense** |
| DeepSeek-V3 | 1318 | 671B/37B activated | MoE |
| LLaMA 3.1 405B | 1269 | 405B | Dense |
| Qwen2.5 72B | 1257 | 72B | Dense |
| Gemma 2 27B IT | 1220 | 27B | Dense |

Gemma 3 27B IT outperforms models $2.7\times$–$15\times$ larger in parameter count.

### 7.5. Memorization Evaluation

**Protocol (Gemma Team, 2024b methodology):**

1. Subsample training data uniformly across corpora
2. For each sample, use prefix of length 50 tokens
3. Generate continuation of length 50 tokens
4. **Exact memorization:** All 50 continuation tokens match source suffix
5. **Approximate memorization:** Match up to edit distance ≤ 10% (≤ 5 token edits)

**Results (Figure 9):**
- Gemma 3 memorization rates are **orders of magnitude lower** than all prior Gemma and Gemini models (log-scale y-axis)
- Marginal difference between 4B, 12B, 27B; 1B memorizes least
- Approximate memorization exceeds exact memorization by $\sim 24\times$ on average
- **Zero personal information** detected in memorized outputs across all Gemma 3 models (using Google Cloud SDP with high-recall detection)

### 7.6. Safety Evaluation

**Baseline assurance:** Synthetic adversarial queries → human rater labeling → violation rate measurement. Gemma 3 violation rate is "significantly low overall."

**CBRN evaluation:** Multiple-choice knowledge-based questions on biological, radiological, nuclear, and chemical hazards. Gemma 3 model knowledge in these domains is assessed as "low."

---

## 8. Deployment Constraints

<img src="assets/gemma3_technical_architecture-15.png" alt="Inference architecture and deployment dashboard for Gemma 3 with Pan and Scan inference, benchmark capability summary, safety and memorization audit, and device memory budgets" width="100%" />

*Figure. Deployment-oriented summary of Gemma 3: multimodal inference flow, benchmark capabilities, low memorization and PII risk, and hardware memory tiers for quantized serving.*

### 8.1. Target Hardware

- **Phones, laptops, consumer GPUs** — models are designed for standard consumer-grade hardware
- 1B model: fits in <3 GB including KV-cache at 32K context (BF16)
- 4B Int4: fits in ~7.3 GB including KV-cache at 32K context
- 27B Int4: fits in ~32.8 GB including KV-cache at 32K context (single high-end GPU)

### 8.2. Memory Budget Analysis

For deployment at context length $L_{\text{ctx}}$, the total memory is:

$$
M_{\text{total}} = M_{\text{weights}}(q) + M_{\text{KV}}(L_{\text{ctx}}, q_{\text{KV}}) + M_{\text{activations}}
$$

where $q$ denotes weight quantization format, $q_{\text{KV}}$ denotes KV-cache quantization format.

**Critical scaling behavior:**

For global-only transformers: $M_{\text{KV}} \propto L_{\text{ctx}}$

For Gemma 3 (5:1 local:global): $M_{\text{KV}} \propto \frac{L_{\text{ctx}}}{6} + \frac{5w}{6}$

At $L_{\text{ctx}} = 128\text{K}$, the KV-cache for Gemma 3 is $\sim 5.8\times$ smaller than a global-only equivalent, making 128K context feasible on hardware where it would otherwise be impossible.

### 8.3. Quantization Deployment Formats

Targeted at popular open-source inference engines (e.g., llama.cpp):

| Format | Bits | Granularity | Use Case |
|---|---|---|---|
| Per-channel Int4 | 4 | Per output channel | Maximum compression, slight quality loss |
| Per-block Int4 (B=32) | 4 + metadata | Per 32-element block | Better quality than per-channel, slightly larger |
| SFP8 | 8 | Switched floating point | Best quality among quantized, 2× size of Int4 |

### 8.4. Vision Encoder Deployment

- **Frozen during training → frozen during inference:** No gradient computation needed
- **Shared across 4B, 12B, 27B:** Single 417M encoder serves all model sizes
- **Pre-computation possible:** For batch inference on known image sets, SigLIP embeddings can be cached
- **P&S overhead:** Each additional crop adds 256 tokens to the sequence, linearly increasing prefill cost

### 8.5. Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| KV-cache OOM | Context exceeds memory at 128K | Local/global interleaving reduces by 5.8×; quantize KV to 8-bit |
| Vision hallucination | Model generates descriptions not grounded in image content | P&S for fine-grained detail; safety filtering in post-training |
| Memorization/recitation | Model reproduces training data verbatim | Deduplication, quality filtering; memorization rates verified to be very low |
| Long-context degradation | Performance degrades beyond 128K tokens | RoPE rescaling bounds effective context; do not exceed 128K |
| Quantization quality loss | Int4 quantization introduces errors | QAT with 5000 steps of distillation-based fine-tuning; use SFP8 for quality-sensitive tasks |
| Reward hacking in RL | Policy exploits reward model artifacts | WARM (weight-averaged reward models) provides implicit regularization |
| Contamination | Benchmark data leaks into training | Decontamination pipeline; acknowledged residual risk |

### 8.6. Serving Topology Considerations

- **Single-device serving:** 1B and 4B (quantized) fit on mobile/laptop
- **Single-GPU serving:** 12B (Int4) and 27B (Int4) fit on 40–80 GB GPUs
- **Multi-device serving:** 27B BF16 at 128K context requires model parallelism or offloading
- **Batch inference:** Pre-computed SigLIP embeddings enable batched multimodal inference without redundant vision encoding
- **P&S as optional optimization:** Can be disabled for latency-sensitive serving at the cost of quality on non-square/high-resolution images

---

## 9. Convergence Dynamics and Training Stability

### 9.1. Normalization and Stability

The combination of **pre-norm + post-norm + RMSNorm + QK-norm** provides:

1. **Pre-norm:** Stabilizes gradient magnitudes entering attention and FFN sub-layers:

$$
\mathbf{h}_l = \mathbf{x}_l + \text{SubLayer}(\text{RMSNorm}(\mathbf{x}_l))
$$

2. **Post-norm:** Additional normalization after residual addition prevents representation drift at depth:

$$
\mathbf{x}_{l+1} = \text{RMSNorm}(\mathbf{h}_l)
$$

3. **QK-norm:** Prevents attention logit explosion that causes softmax saturation:

$$
\text{logit}_{ij} = \frac{\text{RMSNorm}(\mathbf{q}_i)^\top \text{RMSNorm}(\mathbf{k}_j)}{\sqrt{d_k}}
$$

This bounds logit magnitudes: $|\text{logit}_{ij}| \leq \frac{1}{\sqrt{d_k}} \|\boldsymbol{\gamma}_Q\| \|\boldsymbol{\gamma}_K\|$, where $\boldsymbol{\gamma}_Q, \boldsymbol{\gamma}_K$ are learnable scale parameters.

### 9.2. Distillation Convergence

The distillation loss converges as the student distribution $p_S$ approaches the truncated teacher distribution $\tilde{p}_T$:

$$
\mathcal{L}_{\text{distill}} \geq H(\tilde{p}_T)
$$

with equality when $p_S = \tilde{p}_T$. The gap:

$$
\mathcal{L}_{\text{distill}} - H(\tilde{p}_T) = D_{\text{KL}}(\tilde{p}_T \| p_S) \geq 0
$$

drives gradient updates toward matching the teacher's distribution.

### 9.3. RL Fine-Tuning Stability

**WARM** stabilizes RL training by averaging reward model parameters:

$$
\text{Var}[R_{\text{WARM}}] \leq \frac{1}{M} \text{Var}[R_{\text{single}}] + \text{bias}^2
$$

The bias term arises from weight averaging being applied in parameter space (not output space), but empirically this bias is small while variance reduction is substantial.

**WARP** stabilizes policy updates by averaging across RL trajectories:

$$
\text{Var}[\theta_{\text{WARP}}] = \frac{1}{J^2} \sum_{j=1}^{J} \text{Var}[\theta_j^{\text{RL}}]
$$

reducing policy variance by $1/J$.

---

## 10. Complete Pseudo-Algorithm: End-to-End Training

```
ALGORITHM: Gemma3Training
INPUT: Raw corpus C, Teacher model T, Reward models {R_c}
OUTPUT: Final IT checkpoint θ_IT, Quantized checkpoints {θ_Q}

// ===== STAGE 1: DATA PREPARATION =====
1.  D_train = DataPreprocessingPipeline(C)  // Section 1.5
2.  For each image in D_train:
3.      E_img = SigLIP(image)  // Pre-compute, store on disk
4.      Replace image with E_img in D_train

// ===== STAGE 2: PRE-TRAINING (SHORT CONTEXT) =====
5.  Initialize θ_S randomly
6.  Set context_length = 32K
7.  Set RoPE_base_global = 1M, RoPE_base_local = 10K
8.  For step = 1 to T_pretrain:
9.      Sample batch B from D_train
10.     For each token position i in B:
11.         Compute teacher logits: p_T(· | x_{<i})
12.         Select top-256 tokens: T_i = top-256(p_T)
13.         Renormalize: p̃_T over T_i
14.     Compute L_distill = -Σ_i Σ_{v∈T_i} p̃_T(v) log p_S(v; θ_S)
15.     Update θ_S via optimizer (ZeRO-3 sharded)
16.     Monitor pre-training probes (science, code, etc.)

// ===== STAGE 3: LONG-CONTEXT EXTENSION =====
17. For models ∈ {4B, 12B, 27B}:
18.     Rescale RoPE: θ_k ← θ_k / 8 (global layers only)
19.     Set context_length = 128K
20.     Continue training on long-context data for T_ext steps
21.     Validate: check perplexity stability up to 128K

// ===== STAGE 4: INSTRUCTION TUNING (SFT) =====
22. D_sft = PostTrainingDataFiltering(D_raw_sft)  // Section 4.2.4
23. θ_SFT ← θ_S (initialize from pre-trained)
24. For step = 1 to T_sft:
25.     Sample (prompt, response) from D_sft
26.     Format with IT control tokens
27.     Compute L_SFT (distillation from IT teacher, model turn only)
28.     Update θ_SFT

// ===== STAGE 5: RL FINE-TUNING =====
29. θ_RL ← θ_SFT
30. Construct WARM reward: R_WARM using weight-averaged RMs
31. For step = 1 to T_rl:
32.     Sample prompt x
33.     Generate N responses {y^(1), ..., y^(N)} from π_{θ_RL}
34.     Score: r_k = R_WARM(x, y^(k)) for each k
35.     Select y* = argmax_k r_k  (BOND)
36.     Compute L_BOND = -Σ_i log π_{θ_RL}(y*_i | x, y*_{<i})
37.     Update θ_RL
38.     Periodically: save checkpoint θ_j^RL
39. Apply WARP: θ_IT = (1/J) Σ_j θ_j^RL

// ===== STAGE 6: QUANTIZATION-AWARE TRAINING =====
40. For each quantization format q ∈ {Int4-channel, Int4-block, SFP8}:
41.     θ_Q ← θ_IT
42.     For step = 1 to 5000:
43.         Forward pass with simulated quantization (STE)
44.         Compute L_QAT using θ_IT probabilities as targets
45.         Update θ_Q
46.     Save θ_Q

// ===== STAGE 7: EVALUATION =====
47. Run benchmark suite (Table 6) on θ_IT
48. Run Chatbot Arena evaluation
49. Run memorization audit (prefix-50, suffix-50)
50. Run safety evaluations (baseline assurance, CBRN)
51. Run SDP personal information detection on memorized outputs

52. Return θ_IT, {θ_Q}
```

---

## 11. Architectural Complexity Summary

### 11.1. FLOPs per Training Step

For a sequence of length $L$ and model with $N$ layers, $d_{\text{model}}$ hidden dimension, $d_{\text{ff}}$ FFN dimension:

**Attention FLOPs per layer per token:**

$$
\text{FLOPs}_{\text{attn}}^{\text{global}} = 4 \cdot d_{\text{model}} \cdot d_k \cdot n_h + 2 \cdot n_g \cdot d_k \cdot L
$$

$$
\text{FLOPs}_{\text{attn}}^{\text{local}} = 4 \cdot d_{\text{model}} \cdot d_k \cdot n_h + 2 \cdot n_g \cdot d_k \cdot w
$$

**FFN FLOPs per layer per token:**

$$
\text{FLOPs}_{\text{FFN}} = 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} \cdot 2 = 4 \cdot d_{\text{model}} \cdot d_{\text{ff}}
$$

(factor of 2 for up-projection and down-projection, another factor of 2 for gating if using SwiGLU-style FFN)

**Total training FLOPs (forward pass):**

$$
\text{FLOPs}_{\text{total}} \approx L \cdot \left[\frac{N}{6}\left(\text{FLOPs}_{\text{attn}}^{\text{global}} + \text{FLOPs}_{\text{FFN}}\right) + \frac{5N}{6}\left(\text{FLOPs}_{\text{attn}}^{\text{local}} + \text{FLOPs}_{\text{FFN}}\right)\right]
$$

Backward pass ≈ $2\times$ forward pass FLOPs.

### 11.2. Space Complexity

| Component | Memory |
|---|---|
| Parameters | $|\theta| \cdot b_{\text{precision}}$ |
| Optimizer state (Adam) | $2|\theta| \cdot 4$ bytes (FP32 first/second moments) |
| Gradients | $|\theta| \cdot b_{\text{precision}}$ |
| Activations (with checkpointing) | $\mathcal{O}(\sqrt{N} \cdot L \cdot d_{\text{model}})$ |
| KV-cache (inference) | See Section 2.5.3 |

---

## 12. Tensor Transformation Summary

**Layer-by-layer tensor flow for a single Gemma 3 decoder layer (local):**

$$
\mathbf{x} \in \mathbb{R}^{L \times d_{\text{model}}} \xrightarrow{\text{RMSNorm}} \hat{\mathbf{x}} \in \mathbb{R}^{L \times d_{\text{model}}}
$$

$$
\hat{\mathbf{x}} \xrightarrow{\mathbf{W}_Q} \mathbf{Q} \in \mathbb{R}^{L \times n_h \times d_k} \xrightarrow{\text{RMSNorm (QK-norm)}} \hat{\mathbf{Q}}
$$

$$
\hat{\mathbf{x}} \xrightarrow{\mathbf{W}_K} \mathbf{K} \in \mathbb{R}^{L \times n_g \times d_k} \xrightarrow{\text{RMSNorm (QK-norm)}} \hat{\mathbf{K}}
$$

$$
\hat{\mathbf{x}} \xrightarrow{\mathbf{W}_V} \mathbf{V} \in \mathbb{R}^{L \times n_g \times d_v}
$$

$$
\text{RoPE}(\hat{\mathbf{Q}}, \hat{\mathbf{K}}, \beta=10\text{K}) \rightarrow \hat{\mathbf{Q}}', \hat{\mathbf{K}}'
$$

$$
\mathbf{A} = \text{softmax}\!\left(\frac{\hat{\mathbf{Q}}' {\hat{\mathbf{K}}'}^\top}{\sqrt{d_k}} + \mathbf{M}^{\text{local}}\right) \in \mathbb{R}^{L \times n_h \times L}
$$

$$
\mathbf{O} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{L \times n_h \times d_v} \xrightarrow{\text{reshape, } \mathbf{W}_O} \mathbf{o} \in \mathbb{R}^{L \times d_{\text{model}}}
$$

$$
\mathbf{h} = \mathbf{x} + \mathbf{o} \xrightarrow{\text{RMSNorm (post-norm)}} \hat{\mathbf{h}}
$$

$$
\hat{\mathbf{h}} \xrightarrow{\text{RMSNorm (pre-norm)}} \hat{\mathbf{h}}'
$$

$$
\hat{\mathbf{h}}' \xrightarrow{\text{FFN}} \mathbf{f} \in \mathbb{R}^{L \times d_{\text{model}}}
$$

$$
\mathbf{x}_{\text{out}} = \hat{\mathbf{h}} + \mathbf{f} \xrightarrow{\text{RMSNorm (post-norm)}} \text{next layer input}
$$

For **global layers**, the only differences are:
- $\mathbf{M}^{\text{local}} \rightarrow \mathbf{M}^{\text{global}}$ (full causal mask)
- $\beta = 10\text{K} \rightarrow \beta = 1\text{M}$
- KV-cache stores full sequence length instead of sliding window buffer
