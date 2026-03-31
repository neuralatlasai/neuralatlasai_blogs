# Llama 3: End-to-End Technical Report

---

## 1. Data Pipeline

### 1.1 Definition and Objectives

The data pipeline for Llama 3 is a multi-stage system that transforms raw heterogeneous web-scale corpora into a curated, deduplicated, quality-filtered token stream suitable for autoregressive language model pre-training.

**Formal Objective:**

Given raw corpus $\mathcal{D}_{\text{raw}} = \{d_1, d_2, \ldots, d_N\}$ where each $d_i$ is a raw HTML document, produce a filtered token corpus $\mathcal{D}_{\text{clean}} = \{t_1, t_2, \ldots, t_T\}$ where $T \approx 15.6 \times 10^{12}$ tokens, such that:

$$\mathcal{D}_{\text{clean}} = \Phi_{\text{anneal}} \circ \Phi_{\text{mix}} \circ \Phi_{\text{model-filter}} \circ \Phi_{\text{heuristic}} \circ \Phi_{\text{dedup}} \circ \Phi_{\text{extract}} \circ \Phi_{\text{safety}}(\mathcal{D}_{\text{raw}})$$

**Inputs:** Raw HTML crawl data with temporal coverage through end of 2023.

**Outputs:** 15.6T tokenized sequences with controlled domain distribution.

**Invariants:**
- No benchmark training sets included in pre-training data (contamination control)
- PII and adult content removal maintained across all stages
- Mathematical and code structure preserved through extraction

---

### 1.2 Stage-Wise Pipeline

#### 1.2.1 PII and Safety Filtering ($\Phi_{\text{safety}}$)

- **Mechanism:** Domain-level blocklisting using Meta safety rankings; URL-pattern filters for PII-dense domains; adult content domain removal
- **Failure Modes:** False negatives on novel PII patterns; over-filtering of legitimate educational/medical content containing sensitive terminology

#### 1.2.2 Text Extraction and Cleaning ($\Phi_{\text{extract}}$)

- **Mechanism:** Custom HTML parser optimized for precision in boilerplate removal and content recall
- **Key Design Decisions:**
  - Preserves `alt` attribute text from images (critical for mathematical content rendered as images)
  - Preserves code and math structural formatting
  - Removes all markdown markers (empirically shown to degrade performance for web-data-dominant training)
- **Evaluation:** Human evaluations against third-party article-focused parsers; Meta's parser demonstrated superior recall on heterogeneous web content

#### 1.2.3 De-duplication ($\Phi_{\text{dedup}}$)

Three-level hierarchical deduplication:

**URL-level:**
- Global URL deduplication across entire dataset
- Retention policy: keep most recent version per URL

**Document-level:**
- Global MinHash (Broder, 1997) near-duplicate detection
- MinHash signature: document $d$ mapped to signature $\text{sig}(d) = [\min_{t \in S(d)} h_1(t), \ldots, \min_{t \in S(d)} h_k(t)]$ where $S(d)$ is the shingle set and $h_i$ are independent hash functions
- Jaccard similarity threshold for near-duplicate pair identification:

$$J(d_i, d_j) = \frac{|S(d_i) \cap S(d_j)|}{|S(d_i) \cup S(d_j)|} \approx \frac{1}{k}\sum_{l=1}^{k} \mathbb{1}[\text{sig}_l(d_i) = \text{sig}_l(d_j)]$$

**Line-level:**
- Aggressive line-level deduplication following ccNet (Wenzek et al., 2019)
- Removal threshold: lines appearing $> 6$ times per bucket of 30M documents
- Removes boilerplate (navigation menus, cookie warnings) and some high-quality repeated text
- **Trade-off:** Empirical evaluations showed net positive despite removing some quality content

#### 1.2.4 Heuristic Filtering ($\Phi_{\text{heuristic}}$)

- **Duplicated n-gram coverage ratio** (Rae et al., 2021): removes lines with excessive repeated content (logging, error messages) that escape line-dedup due to length/uniqueness
- **Dirty word counting** (Raffel et al., 2020): filters adult websites not caught by domain blocklists
- **Token-distribution KL divergence filter:** For document $d$ with token distribution $p_d$ and corpus-wide distribution $q$:

$$D_{\text{KL}}(p_d \| q) = \sum_{v \in \mathcal{V}} p_d(v) \log \frac{p_d(v)}{q(v)}$$

Documents with $D_{\text{KL}}(p_d \| q) > \tau_{\text{KL}}$ are removed as outliers with excessive rare token concentrations.

#### 1.2.5 Model-Based Quality Filtering ($\Phi_{\text{model-filter}}$)

**Two-tier classifier system:**

**Tier 1 — fastText classifier:**
- Trained to classify whether text would be referenced by Wikipedia (Touvron et al., 2023a)
- Low computational cost, high throughput

**Tier 2 — RoBERTa-based classifier:**
- Training procedure: (1) Curate cleaned web documents, (2) Define quality requirements in natural language, (3) Use Llama 2 chat model as annotator to judge whether documents meet requirements, (4) Train DistilRoBERTa (Sanh et al., 2019) on Llama 2 annotations for efficiency
- Each document $d$ receives quality score $s(d) = f_{\text{DistilRoBERTa}}(d) \in [0, 1]$
- Threshold $\tau_q$ applied for filtering

**Domain-specific pipelines (Code & Reasoning):**
- Separate DistilRoBERTa classifiers trained on Llama 2-annotated web data
- Prompt-tuned to target: mathematical deduction, STEM reasoning, code interleaved with natural language
- Domain-specific HTML extraction, customized text features, domain-specific heuristics
- Separate token distribution handling (code/math distributions differ substantially from natural language)

#### 1.2.6 Multilingual Processing

- **Language identification:** fastText-based classifier categorizing documents into 176 languages
- **Per-language processing:** Document-level and line-level deduplication within each language
- **Quality ranking:** Multilingual Llama 2-based classifier for quality prioritization
- **Multilingual token budget:** Determined experimentally, balancing English vs. multilingual benchmark performance

---

### 1.3 Data Mix Determination

#### 1.3.1 Knowledge Classification

A classifier categorizes web data by information type (e.g., arts/entertainment, science, technology). Over-represented categories are downsampled.

#### 1.3.2 Scaling Law Experiments for Data Mix

- Train small models on candidate data mixes
- Use scaling laws (Section 3.2.1) to predict large-model performance on each mix
- Iterate: select new candidate mix → train larger model → evaluate on key benchmarks

#### 1.3.3 Final Data Mix

| Domain | Proportion |
|---|---|
| General knowledge | ~50% |
| Mathematical and reasoning | ~25% |
| Code | ~17% |
| Multilingual | ~8% |

---

### 1.4 Annealing Data

- **Mechanism:** During final 40M tokens of pre-training, upsample high-quality data in select domains (code, mathematics)
- **Contamination control:** No benchmark training sets included
- **Empirical results:**
  - 8B model: GSM8K +24.0%, MATH +6.4% from annealing
  - 405B model: negligible improvement (strong in-context learning already present)

**Annealing as data quality assessment:**
- Anneal learning rate of 50%-trained 8B model linearly to 0 on 40B tokens
- Assign 30% weight to candidate new dataset, 70% to default mix
- More efficient than full scaling law experiments for small domain-specific datasets

<img src="assets/llama_3_405b_technical_blueprint_p05.png" alt="Staged context extension and high-quality data annealing schedule for Llama 3" width="100%" />

*Figure. Annealing and staged context-extension schedule, corresponding to this section's late-phase data-quality emphasis and the different gains observed at 8B versus 405B scale.*

---

### 1.5 Pseudo-Algorithm: Data Pipeline

```
ALGORITHM: DataPipeline
INPUT: Raw HTML corpus D_raw
OUTPUT: Tokenized corpus D_clean

1. D ← SafetyFilter(D_raw)                          // PII, adult, unsafe domain removal
2. D ← HTMLExtract(D, preserve_alt=True,
                      preserve_code_math=True,
                      remove_markdown=True)
3. D ← URLDedup(D, policy="keep_most_recent")
4. D ← MinHashDedup(D, num_hashes=k,
                       jaccard_threshold=τ_J)
5. FOR each bucket B of 30M documents in D:
       D ← LineDedup(B, max_occurrences=6)
6. D ← HeuristicFilter(D,
         ngram_dup_ratio, dirty_word_count,
         kl_divergence_threshold=τ_KL)
7. D_general ← QualityClassifier(D, fasttext, roberta,
                                    threshold=τ_q)
8. D_code ← CodeClassifier(D, domain_roberta)
9. D_math ← MathClassifier(D, domain_roberta)
10. D_multilingual ← MultilingualPipeline(D,
         lang_id=fasttext_176,
         per_lang_dedup=True,
         quality_rank=llama2_multilingual)
11. D_clean ← MixData(D_general, D_code, D_math,
                        D_multilingual,
                        proportions=[0.50, 0.17, 0.25, 0.08])
12. D_clean ← Tokenize(D_clean, vocab_size=128000)
13. RETURN D_clean
```

---

### 1.6 Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| Benchmark contamination | Training set overlap with evaluation benchmarks | Explicit exclusion of benchmark data |
| Over-filtering | Quality classifiers remove niche but valuable content | Multi-tier filtering with empirical validation |
| Language misclassification | fastText errors on code-switched or low-resource text | Per-language deduplication + quality ranking |
| KL divergence false positives | Legitimate technical documents with unusual token distributions removed | Domain-specific pipelines for code/math |
| Temporal staleness | Older crawl data dominates | Upsampling recent web data in later pre-training stages |

---

## 2. Compression Pipeline (Tokenization)

### 2.1 Definition

The tokenizer maps raw text $x = (c_1, c_2, \ldots, c_L)$ (character sequence of length $L$) to a token sequence $t = (t_1, t_2, \ldots, t_T)$ where each $t_i \in \mathcal{V}$ with vocabulary size $|\mathcal{V}| = 128{,}000$.

### 2.2 Tokenizer Design

**Base vocabulary:** 100K tokens from tiktoken (OpenAI's BPE tokenizer)

**Extension:** 28K additional tokens for non-English language support

**Compression ratios:**

$$\text{CR}_{\text{Llama 3}} = \frac{L}{T} = 3.94 \text{ characters/token (English)}$$

$$\text{CR}_{\text{Llama 2}} = 3.17 \text{ characters/token (English)}$$

**Improvement:**

$$\Delta\text{CR} = \frac{3.94 - 3.17}{3.17} \approx +24.3\%$$

This means for fixed compute budget (fixed number of tokens processed), Llama 3 reads $\sim 24\%$ more raw text than Llama 2.

### 2.3 Information Preservation Analysis

For a BPE tokenizer with vocabulary $\mathcal{V}$, the encoding is a lossless bijection:

$$\text{Encode}: \Sigma^* \rightarrow \mathcal{V}^*, \quad \text{Decode}: \mathcal{V}^* \rightarrow \Sigma^*$$

$$\text{Decode}(\text{Encode}(x)) = x, \quad \forall x \in \Sigma^*$$

**No information degradation** occurs — tokenization is invertible. The compression is purely representational: it reduces sequence length without any lossy compression.

### 2.4 Impact on Effective Compute

Given training budget $C$ FLOPs processing $T$ tokens, the effective text coverage is:

$$L_{\text{effective}} = T \times \text{CR}$$

For Llama 3 flagship: $T = 15.6 \times 10^{12}$ tokens $\times$ 3.94 chars/token $= 61.5 \times 10^{12}$ characters

### 2.5 Multilingual Compression

The 28K additional tokens improve compression ratios for non-English languages without degrading English tokenization. This is achieved by including high-frequency subword units from target languages in the BPE merge table.

### 2.6 Pseudo-Algorithm: Tokenization

```
ALGORITHM: BPETokenize
INPUT: Text string x, merge table M (128K entries)
OUTPUT: Token sequence t

1. Initialize t ← character-level split of x (UTF-8 bytes)
2. WHILE any adjacent pair (t_i, t_{i+1}) exists in M:
       pair* ← argmin_{(t_i,t_{i+1}) ∈ M} rank(t_i, t_{i+1})
       Merge all occurrences of pair* in t
3. Map merged symbols to integer IDs via vocabulary V
4. RETURN t
```

---

## 3. Model Architecture

### 3.1 Definition

Llama 3 is a decoder-only autoregressive Transformer (Vaswani et al., 2017) that models the conditional probability:

$$p(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_1, \ldots, x_{t-1}; \theta)$$

where $\theta$ denotes the model parameters and each $x_t \in \mathcal{V}$ with $|\mathcal{V}| = 128{,}000$.

### 3.2 Architecture Specification

| Hyperparameter | 8B | 70B | 405B |
|---|---|---|---|
| Layers $L$ | 32 | 80 | 126 |
| Model dimension $d_{\text{model}}$ | 4,096 | 8,192 | 16,384 |
| FFN dimension $d_{\text{ff}}$ | 14,336 | 28,672 | 53,248 |
| Attention heads $n_h$ | 32 | 64 | 128 |
| Key/Value heads $n_{\text{kv}}$ | 8 | 8 | 8 |
| Head dimension $d_h$ | 128 | 128 | 128 |
| Vocabulary size $|\mathcal{V}|$ | 128,000 | 128,000 | 128,000 |
| Peak learning rate | $3 \times 10^{-4}$ | $1.5 \times 10^{-4}$ | $8 \times 10^{-5}$ |
| Activation function | SwiGLU | SwiGLU | SwiGLU |
| Positional encoding | RoPE ($\theta = 500{,}000$) | RoPE ($\theta = 500{,}000$) | RoPE ($\theta = 500{,}000$) |

<img src="assets/llama_3_405b_technical_blueprint_p01.png" alt="Llama 3 405B flagship dense architecture with parameter scale, training scale, and compute budget" width="100%" />

*Figure. Flagship dense-architecture summary, corresponding to the 405B decoder-only design and system-scale specification in Section 3.*

### 3.3 Architectural Blocks

#### 3.3.1 Input Embedding

$$\mathbf{h}_0 = \mathbf{E}[x] \in \mathbb{R}^{T \times d_{\text{model}}}$$

where $\mathbf{E} \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$ is the token embedding matrix.

#### 3.3.2 Transformer Block (repeated $L$ times)

Each layer $l \in \{1, \ldots, L\}$ computes:

**Pre-normalization (RMSNorm):**

$$\hat{\mathbf{h}}_{l-1} = \text{RMSNorm}(\mathbf{h}_{l-1})$$

$$\text{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \epsilon}} \cdot \gamma_i$$

where $\gamma \in \mathbb{R}^{d_{\text{model}}}$ is a learnable scale parameter and $\epsilon$ is a small constant for numerical stability.

**Grouped Query Attention (GQA):**

$$\mathbf{Q} = \hat{\mathbf{h}}_{l-1} \mathbf{W}_Q \in \mathbb{R}^{T \times n_h \times d_h}$$

$$\mathbf{K} = \hat{\mathbf{h}}_{l-1} \mathbf{W}_K \in \mathbb{R}^{T \times n_{\text{kv}} \times d_h}$$

$$\mathbf{V} = \hat{\mathbf{h}}_{l-1} \mathbf{W}_V \in \mathbb{R}^{T \times n_{\text{kv}} \times d_h}$$

where $\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times n_h d_h}$, $\mathbf{W}_K \in \mathbb{R}^{d_{\text{model}} \times n_{\text{kv}} d_h}$, $\mathbf{W}_V \in \mathbb{R}^{d_{\text{model}} \times n_{\text{kv}} d_h}$.

**GQA mechanism:** Each group of $g = n_h / n_{\text{kv}}$ query heads shares one key-value head. For 405B: $g = 128/8 = 16$ query heads per KV head.

**RoPE positional encoding applied to Q and K:**

For position $m$ and dimension pair $(2i, 2i+1)$:

$$\text{RoPE}(\mathbf{q}, m)_{2i} = q_{2i} \cos(m\theta_i) - q_{2i+1} \sin(m\theta_i)$$

$$\text{RoPE}(\mathbf{q}, m)_{2i+1} = q_{2i} \sin(m\theta_i) + q_{2i+1} \cos(m\theta_i)$$

where:

$$\theta_i = \theta_{\text{base}}^{-2i/d_h}, \quad \theta_{\text{base}} = 500{,}000$$

The elevated base frequency $\theta_{\text{base}} = 500{,}000$ (vs. 10,000 in standard RoPE) extends effective context length support; Xiong et al. (2023) demonstrated efficacy up to 32,768 tokens, and Llama 3 extends this further to 128K through continued pre-training.

**Scaled dot-product attention:**

$$\text{Attn}(\mathbf{Q}_{\text{rope}}, \mathbf{K}_{\text{rope}}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}_{\text{rope}} \mathbf{K}_{\text{rope}}^\top}{\sqrt{d_h}} + \mathbf{M}\right) \mathbf{V}$$

where $\mathbf{M}$ is the attention mask combining:
1. **Causal mask:** $M_{ij} = -\infty$ for $j > i$
2. **Document mask:** $M_{ij} = -\infty$ when tokens $i$ and $j$ belong to different documents within the same sequence

The document mask prevents cross-document attention within packed sequences, which is critical for long-context continued pre-training.

**Output projection:**

$$\mathbf{a}_l = \text{Attn}(\mathbf{Q}_{\text{rope}}, \mathbf{K}_{\text{rope}}, \mathbf{V}) \mathbf{W}_O$$

where $\mathbf{W}_O \in \mathbb{R}^{n_h d_h \times d_{\text{model}}}$.

**Residual connection:**

$$\mathbf{h}_l' = \mathbf{h}_{l-1} + \mathbf{a}_l$$

**SwiGLU Feed-Forward Network:**

$$\hat{\mathbf{h}}_l' = \text{RMSNorm}(\mathbf{h}_l')$$

$$\text{SwiGLU}(\mathbf{x}) = (\mathbf{x}\mathbf{W}_1 \odot \text{SiLU}(\mathbf{x}\mathbf{W}_{\text{gate}})) \mathbf{W}_2$$

where $\mathbf{W}_1, \mathbf{W}_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

**Second residual connection:**

$$\mathbf{h}_l = \mathbf{h}_l' + \text{SwiGLU}(\hat{\mathbf{h}}_l')$$

#### 3.3.3 Output Head

$$\mathbf{h}_{\text{final}} = \text{RMSNorm}(\mathbf{h}_L)$$

$$\text{logits} = \mathbf{h}_{\text{final}} \mathbf{W}_{\text{out}} \in \mathbb{R}^{T \times |\mathcal{V}|}$$

$$p(x_{t+1} \mid x_{\leq t}) = \text{softmax}(\text{logits}_t)$$

### 3.4 Parameter Count Analysis

For the 405B model:

**Per-layer parameters:**
- Attention: $(d_{\text{model}} \times n_h d_h) + (d_{\text{model}} \times n_{\text{kv}} d_h) \times 2 + (n_h d_h \times d_{\text{model}})$
  - $= 16384 \times 16384 + 2 \times (16384 \times 1024) + 16384 \times 16384$
  - $= 268{,}435{,}456 + 33{,}554{,}432 + 268{,}435{,}456 = 570{,}425{,}344$
- FFN: $3 \times d_{\text{model}} \times d_{\text{ff}} = 3 \times 16384 \times 53248 = 2{,}617{,}245{,}696$
- RMSNorm: $2 \times d_{\text{model}} = 32{,}768$ (negligible)
- **Per-layer total:** $\approx 3.19 \times 10^9$

**Total across 126 layers:** $\approx 126 \times 3.19 \times 10^9 \approx 401.6 \times 10^9$

**Embedding + output:** $128000 \times 16384 \approx 2.1 \times 10^9$

**Grand total:** $\approx 405 \times 10^9$ parameters

### 3.5 Complexity Analysis

**Attention complexity per layer (full causal):**

$$\mathcal{O}(T^2 \cdot d_h \cdot n_h) = \mathcal{O}(T^2 \cdot d_{\text{model}})$$

**FFN complexity per layer:**

$$\mathcal{O}(T \cdot d_{\text{model}} \cdot d_{\text{ff}}) = \mathcal{O}(T \cdot d_{\text{model}} \cdot d_{\text{ff}})$$

**Total FLOPs per forward pass (approximation):**

$$C_{\text{forward}} \approx 2 \times P \times T$$

where $P$ is total parameter count. For training (forward + backward):

$$C_{\text{train}} \approx 6 \times P \times T$$

**Training budget verification:**

$$C_{\text{total}} = 6 \times 405 \times 10^9 \times 15.6 \times 10^{12} \approx 3.79 \times 10^{25} \text{ FLOPs}$$

This matches the stated $3.8 \times 10^{25}$ FLOPs budget.

### 3.6 KV-Cache Memory Analysis

During inference, KV cache per token per layer:

$$\text{KV per token per layer} = 2 \times n_{\text{kv}} \times d_h \times \text{sizeof(dtype)}$$

For 405B in BF16:

$$= 2 \times 8 \times 128 \times 2 = 4{,}096 \text{ bytes/token/layer}$$

$$\text{Total KV for 128K context} = 128000 \times 126 \times 4096 = 66.1 \text{ GB}$$

GQA reduces this by factor $n_h / n_{\text{kv}} = 16\times$ compared to MHA:

$$\text{MHA equivalent} = 66.1 \times 16 = 1{,}057 \text{ GB}$$

---

## 4. Scaling Laws

### 4.1 Definition

Scaling laws establish functional relationships between compute budget $C$, model size $N$ (parameters), data size $D$ (tokens), and loss $\mathcal{L}$, enabling prediction of optimal resource allocation.

### 4.2 Compute-Optimal Model Size

**Power-law relation:**

$$N^{\star}(C) = A \cdot C^{\alpha}$$

**Fitted parameters:** $(\alpha, A) = (0.53, 0.29)$

This gives the optimal number of training tokens as a function of compute. Since $C \approx 6ND$ (for dense Transformers), the optimal token count:

$$D^{\star}(C) = \frac{C}{6 N^{\star}(C)}$$

**Extrapolation to flagship budget:**

$$C = 3.8 \times 10^{25} \text{ FLOPs} \implies N^{\star} \approx 402\text{B parameters}, \quad D^{\star} \approx 16.55\text{T tokens}$$

### 4.3 IsoFLOPs Methodology

- Pre-train models with compute budgets between $6 \times 10^{18}$ and $10^{22}$ FLOPs
- At each budget, sweep model sizes from 40M to 16B parameters
- Measure validation loss (negative log-likelihood on held-out set)
- Fit second-degree polynomial to loss vs. model-size at each compute budget
- Parabola minimum = compute-optimal model for that budget

**Key observation:** IsoFLOPs curves become flatter around the minimum at higher compute budgets, implying the flagship model's performance is robust to small perturbations in the size-vs-tokens trade-off.

### 4.4 Downstream Task Performance Prediction

**Two-step methodology:**

**Step 1:** Linear correlation between normalized negative log-likelihood on downstream task and training FLOPs (using scaling law models up to $10^{22}$ FLOPs):

$$\text{NLL}_{\text{task}}(C) = a \cdot \log C + b$$

**Step 2:** Sigmoidal relation between log-likelihood and accuracy (using both scaling law models and Llama 2 models):

$$\text{Accuracy}(\text{NLL}) = \frac{1}{1 + \exp(-(\alpha \cdot \text{NLL} + \beta))}$$

This two-step extrapolation spans four orders of magnitude in compute and was validated to only slightly underestimate the final 405B performance on ARC Challenge.

### 4.5 Over-Training Smaller Models

Smaller models (8B, 70B) are intentionally trained beyond compute-optimal token counts. Rationale: at fixed inference budget, an over-trained smaller model outperforms a compute-optimal model of the same size. The flagship 405B model further improves smaller models during post-training via knowledge distillation effects (rejection sampling, synthetic data generation).

---

## 5. Optimization Strategy

### 5.1 Optimizer

**AdamW** with decoupled weight decay:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda_t \theta_{t-1}\right)$$

**Weight decay schedule:** $\lambda_t = 0.1 \times \eta_t$ (weight decay proportional to learning rate at each step)

### 5.2 Learning Rate Schedule

**Three-phase schedule:**

**Phase 1 — Linear Warmup:**

$$\eta_t = \eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \in [0, T_{\text{warmup}}]$$

where $T_{\text{warmup}} = 8{,}000$ steps.

**Phase 2 — Cosine Decay:**

$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{\pi (t - T_{\text{warmup}})}{T_{\text{total}} - T_{\text{warmup}}}\right)\right)$$

where $\eta_{\text{peak}} = 8 \times 10^{-5}$, $\eta_{\text{min}} = 8 \times 10^{-7}$, $T_{\text{total}} = 1{,}200{,}000$ steps for the 405B model.

**Phase 3 — Annealing (final 40M tokens):**

$$\eta_t = \eta_{\text{anneal\_start}} \cdot \left(1 - \frac{t - T_{\text{anneal\_start}}}{T_{\text{anneal\_end}} - T_{\text{anneal\_start}}}\right)$$

Linear decay to 0.

### 5.3 Batch Size Schedule

Progressive batch size increase for training stability:

| Tokens Processed | Batch Size (tokens) | Sequence Length |
|---|---|---|
| $[0, 252\text{M}]$ | 4M | 4,096 |
| $[252\text{M}, 2.87\text{T}]$ | 8M | 8,192 |
| $[2.87\text{T}, 15.6\text{T}]$ | 16M | 8,192 |

**Rationale:** Lower batch size early in training reduces gradient noise impact and improves stability; larger batch size later improves compute efficiency (better GPU utilization).

### 5.4 Numerical Stability Measures

- **FP32 gradient accumulation** across micro-batches during backward pass
- **FP32 reduce-scatter** of gradients across FSDP data-parallel workers
- **FP32 accumulation** for intermediate tensors used multiple times in forward (e.g., vision encoder outputs reused across layers)
- **BF16** for forward computation and parameter storage (mixed precision)

### 5.5 Scaling Law Experiment Optimization Settings

For scaling law runs ($6 \times 10^{18}$ to $10^{22}$ FLOPs):
- Cosine LR schedule with 2,000 warmup steps
- Peak LR: $2 \times 10^{-4}$ to $4 \times 10^{-4}$ (model-size dependent)
- Cosine decay to 0.1× peak
- Weight decay: $0.1 \times \eta_t$
- Fixed batch size per compute scale: 250K to 4M tokens

---

## 6. Training Stages

### 6.1 Stage 1: Initial Pre-Training

**Objective:** Learn language structure, world knowledge, reasoning patterns from next-token prediction.

$$\mathcal{L}_{\text{PT}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

**Configuration for 405B:**
- Parameters: 405B
- Tokens: ~15.6T (across all stages including long-context and annealing)
- Initial context: 8K tokens
- Compute: $3.8 \times 10^{25}$ FLOPs
- Hardware: up to 16K H100 GPUs

**Data mix adjustments during training:**
- Increased non-English data percentage during training for multilingual performance
- Upsampled mathematical data for reasoning
- Added more recent web data in later stages for knowledge recency
- Downsampled subsets identified as lower quality

**Stability:** Very stable training; few loss spikes observed; no manual interventions required for divergence correction.

### 6.2 Stage 2: Long-Context Pre-Training

**Objective:** Extend supported context length from 8K to 128K tokens.

**Methodology:**
- Gradual context length increase in 6 stages: $8\text{K} \rightarrow \ldots \rightarrow 128\text{K}$
- At each stage, continue pre-training until:
  1. Short-context evaluation performance fully recovered
  2. Perfect accuracy on needle-in-a-haystack tasks up to current context length

**Compute:** ~800B training tokens for the long-context stage

**Design rationale:** Self-attention compute scales as $\mathcal{O}(T^2)$; training on long sequences from the start would be prohibitively expensive. Gradual extension amortizes the quadratic cost.

**Success criteria:**

$$\text{Short-context perf.}(\text{stage } k) \geq \text{Short-context perf.}(\text{stage } k-1)$$

$$\text{Needle accuracy}(\text{length} \leq L_k) = 100\%$$

### 6.3 Stage 3: Annealing

**Objective:** Final quality refinement through learning rate annealing and high-quality data upsampling.

- Final 40M tokens
- Linear LR decay to 0
- Context length: 128K maintained
- Data mix: upsampled high-quality code and mathematical data
- **Polyak averaging:** Final pre-trained model = average of checkpoints during annealing:

$$\theta_{\text{final}} = \frac{1}{K} \sum_{k=1}^{K} \theta_{t_k}$$

where $\theta_{t_k}$ are checkpoint parameters at annealing step $t_k$.

### 6.4 Stage 4: Post-Training (Alignment)

**Objective:** Align pre-trained model to follow instructions, match human preferences, and enhance specific capabilities.

**Multiple rounds of iterative alignment:**

<img src="assets/llama_3_405b_technical_blueprint_p06.png" alt="Iterative six-round post-training protocol with data collection, reward modeling, rejection sampling, supervised finetuning, direct preference optimization, and model averaging" width="100%" />

*Figure. Iterative post-training alignment loop, corresponding to the SFT, rejection sampling, and DPO stages described in Section 6.4.*

#### 6.4.1 Supervised Fine-Tuning (SFT)

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{D}_{\text{SFT}}|} \sum_{(x,y) \in \mathcal{D}_{\text{SFT}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

where $x$ is instruction/prompt, $y$ is target response.

#### 6.4.2 Rejection Sampling (RS)

For each prompt $x$, generate $K$ candidate responses $\{y_1, \ldots, y_K\}$ from the current policy $\pi_\theta$. Select best response using reward model $R$:

$$y^* = \arg\max_{y_k} R(x, y_k)$$

Use selected responses as SFT training data for next iteration.

#### 6.4.3 Direct Preference Optimization (DPO)

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

where:
- $y_w$: preferred (winning) response
- $y_l$: dispreferred (losing) response
- $\pi_{\text{ref}}$: reference policy (typically SFT checkpoint)
- $\beta$: temperature parameter controlling deviation from reference
- $\sigma$: sigmoid function

**Implicit reward under DPO:**

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

**Iterative procedure:**

```
ALGORITHM: PostTraining
INPUT: Pre-trained model θ_0
OUTPUT: Aligned model θ_final

FOR round r = 1, 2, ..., R:
    1. θ_SFT ← SFT(θ_{r-1}, D_SFT^{(r)})
    2. Generate K responses per prompt from π_{θ_SFT}
    3. Score with reward model R(x, y)
    4. Select best responses → D_RS^{(r)}
    5. θ_RS ← SFT(θ_SFT, D_RS^{(r)})
    6. Collect preference pairs → D_pref^{(r)}
    7. θ_r ← DPO(θ_RS, D_pref^{(r)}, π_ref = θ_RS)
RETURN θ_R
```

**Capabilities integrated during post-training:**
- Tool use (function calling)
- Coding enhancement
- Reasoning improvement
- Multilingual instruction following
- Safety mitigations (Llama Guard 3)

---

## 7. Infrastructure and Distributed Training

### 7.1 Hardware Configuration

- **GPUs:** Up to 16K NVIDIA H100 (80GB HBM3, 700W TDP)
- **Server:** 8 GPUs + 2 CPUs per server (Meta Grand Teton platform)
- **Intra-server interconnect:** NVLink
- **Inter-server interconnect:** 400 Gbps RoCE (RDMA over Converged Ethernet) via Arista 7800 and Minipack2 OCP rack switches
- **Storage:** Tectonic distributed filesystem — 240 PB across 7,500 SSD servers, 2 TB/s sustained throughput, 7 TB/s peak
- **Scheduler:** MAST (Meta's global-scale training scheduler)

### 7.2 Network Topology

Three-layer Clos network (24K GPU cluster):

| Layer | Description | Oversubscription |
|---|---|---|
| Bottom (ToR) | 16 GPUs (2 servers) per Minipack2 ToR switch | 1:1 |
| Middle (Cluster) | 192 racks = 3,072 GPUs per pod, full bisection bandwidth | 1:1 |
| Top (Aggregation) | 8 pods = 24K GPUs per cluster | 1:7 |

### 7.3 4D Parallelism

**Parallelism order:** [TP, CP, PP, DP] — innermost to outermost, matching bandwidth requirements.

| Parallelism | Dimension | Mechanism | Communication Pattern |
|---|---|---|---|
| **Tensor Parallelism (TP)** | Innermost | Splits weight tensors across GPUs within server | All-reduce (intra-node NVLink) |
| **Context Parallelism (CP)** | Second | Partitions input sequence across GPUs | All-gather K,V tensors |
| **Pipeline Parallelism (PP)** | Third | Partitions model layers across GPU groups | Point-to-point (inter-node) |
| **Data Parallelism (DP/FSDP)** | Outermost | Shards optimizer states + gradients, replicates forward | All-gather params, reduce-scatter grads |

<img src="assets/llama_3_405b_technical_blueprint_p04.png" alt="4D parallelism topology across 16K H100 GPUs for Llama 3 training" width="100%" />

*Figure. 4D parallelism topology, directly corresponding to the TP, CP, PP, and DP arrangement explained in Section 7.3.*

#### 7.3.1 Tensor Parallelism Details

For attention projection $\mathbf{W}_Q \in \mathbb{R}^{d_{\text{model}} \times n_h d_h}$ split across $|TP|$ devices:

$$\mathbf{W}_Q = [\mathbf{W}_Q^{(1)}, \mathbf{W}_Q^{(2)}, \ldots, \mathbf{W}_Q^{(|TP|)}]$$

Each device $k$ computes $\mathbf{Q}^{(k)} = \mathbf{x} \mathbf{W}_Q^{(k)}$, followed by all-reduce for output projection.

#### 7.3.2 Context Parallelism Details

Sequence of length $S$ partitioned into $2 \times |CP|$ chunks. Rank $i$ receives chunks $i$ and $(2|CP| - 1 - i)$ for load balancing.

**Implementation choice:** All-gather based (not ring-based):
1. All-gather K and V tensors across CP group
2. Compute attention on local Q chunk against full K, V

**Rationale:**
- Easier support for heterogeneous attention masks (document mask)
- Exposed all-gather latency is negligible because GQA makes K, V much smaller than Q
- Attention compute is $\mathcal{O}(S^2)$ vs. all-gather $\mathcal{O}(S)$, making communication overhead negligible

#### 7.3.3 Pipeline Parallelism Improvements

**Modified schedule:** $N$ (number of contiguous micro-batches per stage) is made **tunable** (not fixed to PP or $M$).

**Pipeline bubble ratio:**

$$\text{Bubble ratio} = \frac{PP - 1}{V \times M}$$

where $V$ = number of virtual pipeline stages per rank, $M$ = total micro-batches.

**Load balancing:** Remove one Transformer layer from first and last pipeline stages:
- First stage first chunk: embedding only
- Last stage last chunk: output projection + loss only

**Optimizations:**
- Asynchronous point-to-point communication
- `TORCH_NCCL_AVOID_RECORD_STREAMS` for memory reduction
- Proactive tensor deallocation (input/output tensors of each pipeline stage)
- No activation checkpointing needed for 8K sequence pre-training

#### 7.3.4 FSDP Configuration

- Shards optimizer states and gradients across DP group
- Model shards: **not resharded after forward computation** to avoid extra all-gather during backward
- Asynchronous weight prefetching for latency hiding

### 7.4 Training Configurations and MFU

| Stage | GPUs | TP | CP | PP | DP | Seq. Len. | Batch/DP | Tokens/Batch | TFLOPs/GPU | BF16 MFU |
|---|---|---|---|---|---|---|---|---|---|---|
| Initial (8K) | 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| Scale-up | 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| Long-context (128K) | 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

**MFU definition:**

$$\text{MFU} = \frac{\text{Observed TFLOPs/GPU}}{\text{Peak BF16 TFLOPs/GPU (H100)}}$$

H100 peak BF16: ~989 TFLOPs → MFU = 430/989 ≈ 43%

### 7.5 Network Optimizations

**Load balancing:**
- 16 network flows created between any two GPUs (instead of 1)
- Enhanced-ECMP (E-ECMP) hashing on additional RoCE header fields

**Congestion control:**
- Deep-buffer spine switches for transient burst absorption
- No DCQCN needed due to effective E-ECMP load balancing
- High-priority routing for small control messages

### 7.6 Collective Communication (NCCLX)

Custom fork of NCCL with optimizations:
- Tuned chunking and data transfer for high-latency multi-hop networks (tens of microseconds)
- Prioritized small control messages to avoid head-of-line blocking in deep-buffer switches
- Ongoing deeper changes for future versions

### 7.7 Reliability

**Effective training time:** >90% despite daily automated maintenance interruptions

**54-day interruption analysis:**
- Total: 466 interruptions
- Planned: 47 (firmware upgrades, config updates)
- Unplanned: 419
  - GPU-related: 58.7% (faulty GPU: 30.1%, HBM3: 17.2%, SRAM: 4.5%, system processor: 4.1%, thermal: 1.4%, silent data corruption: 1.4%)
  - Software bugs: 12.9%
  - Network: 8.4%
  - Host maintenance: 7.6%
  - Manual intervention required: only 3 times

**Diagnostic tools:**
- PyTorch NCCL flight recorder: ring buffer capturing collective metadata + stack traces
- Online configuration changes for selective intensive tracing (no restart needed)
- Straggler detection: prioritize suspicious communications from selected process groups

**Environmental factors:**
- Diurnal 1-2% throughput variation from temperature-dependent GPU DVFS
- Synchronized power consumption fluctuations (tens of MW) from collective checkpointing/startup

---

## 8. Inference Path

### 8.1 Autoregressive Generation

At inference time, tokens are generated sequentially:

$$x_{t+1} \sim p_\theta(x_{t+1} \mid x_1, \ldots, x_t)$$

**KV Cache:** Key and value tensors from previous positions are cached and reused.

**Per-step complexity:**

$$\mathcal{O}(L \cdot (d_{\text{model}}^2 + T \cdot n_{\text{kv}} \cdot d_h))$$

where $T$ is current sequence length.

### 8.2 GQA Inference Benefits

KV cache memory per token:

$$\text{KV}_{\text{GQA}} = 2 \times L \times n_{\text{kv}} \times d_h \times \text{sizeof(BF16)}$$

$$= 2 \times 126 \times 8 \times 128 \times 2 = 516{,}096 \text{ bytes/token}$$

For 128K context: $\approx 63$ GB

**Comparison to MHA:** $16\times$ reduction (MHA would require $\approx 1{,}008$ GB)

### 8.3 Memory Bandwidth Bound Analysis

For single-token generation (decode phase), the operation is memory-bandwidth bound:

$$\text{Time per token} \approx \frac{\text{Model parameters} \times \text{sizeof(dtype)}}{\text{Memory bandwidth}}$$

For 405B in BF16 across multiple GPUs with aggregate HBM bandwidth:

$$\text{Model memory} = 405 \times 10^9 \times 2 = 810 \text{ GB}$$

H100 HBM3 bandwidth: 3.35 TB/s per GPU. With 8 GPUs (TP=8):

$$\text{Aggregate bandwidth} = 8 \times 3.35 = 26.8 \text{ TB/s}$$

$$\text{Theoretical min time/token} \approx \frac{810}{26800} \approx 30 \text{ ms}$$

### 8.4 RoPE at Inference

At inference, RoPE is applied to Q and K at each position without modification. The elevated $\theta_{\text{base}} = 500{,}000$ ensures wavelengths are long enough for 128K contexts:

Maximum effective wavelength:

$$\lambda_{\max} = 2\pi \cdot \theta_{\text{base}}^{d_h/d_h} = 2\pi \cdot 500000 \approx 3.14 \times 10^6$$

This far exceeds 128K, ensuring minimal positional encoding degradation at maximum context.

### 8.5 Document Mask at Inference

During inference for single-turn generation, the document mask reduces to a standard causal mask. The document mask is primarily relevant during training with packed sequences containing multiple documents.

### 8.6 Pseudo-Algorithm: Inference

```
ALGORITHM: AutoregressiveInference
INPUT: Prompt tokens x = [x_1, ..., x_P], model θ, max_len T
OUTPUT: Generated tokens [x_{P+1}, ..., x_T]

1. // Prefill phase (parallel)
   h ← Embedding(x)                           // [P, d_model]
   FOR l = 1 to L:
       Q, K, V ← Attention_QKV(h, l)          // Apply RoPE to Q, K
       KV_cache[l] ← (K, V)                   // Cache K, V
       h ← TransformerBlock(h, Q, K, V, l)
   logits ← OutputHead(h[-1])                  // Last position
   x_{P+1} ← Sample(softmax(logits))

2. // Decode phase (sequential, one token at a time)
   FOR t = P+1 to T-1:
       h ← Embedding(x_t)                     // [1, d_model]
       FOR l = 1 to L:
           q, k, v ← Attention_QKV(h, l)      // Apply RoPE at position t
           KV_cache[l] ← Append(KV_cache[l], (k, v))
           h ← TransformerBlock(h, q, KV_cache[l], l)
       logits ← OutputHead(h)
       x_{t+1} ← Sample(softmax(logits / temperature))
       IF x_{t+1} == EOS: BREAK

3. RETURN [x_{P+1}, ..., x_{t+1}]
```

---

## 9. Multimodal Extensions (Compositional Approach)

### 9.1 Architecture Overview

Llama 3 multimodal capabilities are added via a compositional approach with three stages:

1. **Multimodal encoder pre-training** (image encoder, speech encoder — trained independently)
2. **Vision adapter training** (cross-attention layers integrating image encoder into LLM)
3. **Speech adapter training** (adapter converting speech encodings to LLM token space)

**Critical design constraint:** Language model parameters are **frozen** during both adapter training stages. Only adapters and encoders are updated.

<img src="assets/llama_3_405b_technical_blueprint_p08.png" alt="Compositional multimodal architecture with a frozen Llama 3 core and trainable image, video, and speech modules" width="100%" />

*Figure. Compositional multimodal architecture, matching the frozen-LLM invariant and modality-specific adapter training described in Section 9.*

### 9.2 Image Encoder

- Pre-trained on large-scale image-text pairs
- Learns visual-semantic alignment
- Output: visual feature representations

### 9.3 Vision Adapter

- Architecture: series of cross-attention layers
- Input: image encoder representations as keys/values, language model hidden states as queries
- During training: adapter parameters + image encoder parameters updated; LLM parameters **frozen**

**Cross-attention formulation:**

$$\text{CrossAttn}(\mathbf{Q}_{\text{text}}, \mathbf{K}_{\text{image}}, \mathbf{V}_{\text{image}}) = \text{softmax}\left(\frac{\mathbf{Q}_{\text{text}} \mathbf{K}_{\text{image}}^\top}{\sqrt{d_h}}\right) \mathbf{V}_{\text{image}}$$

where $\mathbf{Q}_{\text{text}} = \mathbf{h}_{\text{LLM}} \mathbf{W}_Q^{\text{cross}}$ and $\mathbf{K}_{\text{image}}, \mathbf{V}_{\text{image}}$ are projections of image encoder outputs.

### 9.4 Video Adapter

- Built on top of image adapter
- Trained on paired video-text data
- Enables temporal aggregation across frames

### 9.5 Speech Encoder

- Pre-trained with self-supervised learning: mask-and-reconstruct using discrete token targets
- Learns speech signal structure without labeled data

### 9.6 Speech Adapter

- Converts speech encoder representations into token-space representations compatible with the finetuned LLM
- Joint update of adapter + speech encoder parameters during supervised finetuning
- LLM parameters **frozen**
- Integrated with text-to-speech system for bidirectional speech interaction

---

## 10. Evaluation Protocol

### 10.1 Benchmark Suite

| Category | Benchmarks | Metrics |
|---|---|---|
| General | MMLU (5-shot, 0-shot CoT), MMLU-Pro (5-shot CoT), IFEval | Accuracy |
| Code | HumanEval (0-shot), MBPP EvalPlus (0-shot) | pass@1 |
| Math | GSM8K (8-shot CoT), MATH (0-shot CoT) | Accuracy |
| Reasoning | ARC Challenge (0-shot), GPQA (0-shot CoT) | Accuracy |
| Tool Use | BFCL, Nexus | Accuracy |
| Long Context | ZeroSCROLLS/QuALITY, InfiniteBench/En.MC, NIH/Multi-needle | Accuracy |
| Multilingual | MGSM (0-shot CoT) | Accuracy |

### 10.2 Key Results (405B Instruct)

| Benchmark | Llama 3 405B | GPT-4 (0125) | GPT-4o | Claude 3.5 Sonnet |
|---|---|---|---|---|
| MMLU (5-shot) | 87.3 | 85.1 | 89.1 | 89.9 |
| HumanEval | 89.0 | 86.6 | 90.2 | 92.0 |
| GSM8K | 96.8 | 94.2 | 96.1 | 96.4 |
| MATH | 73.8 | 64.5 | 76.6 | 71.1 |
| ARC Challenge | 96.9 | 96.4 | 96.7 | 96.7 |
| BFCL | 88.5 | 88.3 | 80.5 | 90.2 |
| MGSM | 91.6 | 85.9 | 90.5 | 91.6 |

<img src="assets/llama_3_technical_synthesis_p14.png" alt="State-of-the-art benchmark synthesis comparing Llama 3 scales with leading frontier models" width="100%" />

*Figure. Benchmark synthesis table, corresponding to the evaluation section's summary of how the 405B instruct model compares with frontier systems.*

### 10.3 Scaling Law Validation

The two-step downstream prediction methodology was validated against actual 405B performance. On ARC Challenge, the scaling law prediction (extrapolating $4$ orders of magnitude from $10^{22}$ FLOPs models) slightly underestimated actual performance, confirming the methodology's utility for pre-training planning.

### 10.4 Long-Context Evaluation

- Needle-in-a-haystack: 98.1% at 128K (405B Instruct)
- ZeroSCROLLS/QuALITY: 95.2
- InfiniteBench/En.MC: 83.4

### 10.5 Evaluation of Annealing

| Model | GSM8K improvement | MATH improvement |
|---|---|---|
| 8B (with annealing) | +24.0% | +6.4% |
| 405B (with annealing) | Negligible | Negligible |

The negligible improvement for 405B confirms that the flagship model has sufficient in-context learning and reasoning capability without domain-specific annealing data.

### 10.6 Human Evaluations

Extensive human evaluations compare Llama 3 with competing models on helpfulness and harmlessness. Results indicate Llama 3 delivers better balance between these objectives than Llama 2.

---

## 11. Deployment Constraints

### 11.1 Memory Requirements

**405B model in BF16:**

| Component | Memory |
|---|---|
| Model parameters | $405 \times 10^9 \times 2 = 810$ GB |
| KV cache (128K context) | ~63 GB (GQA) |
| Activation memory | Depends on batch size |
| **Minimum GPU count (BF16)** | $\lceil 810 / 80 \rceil = 11$ GPUs minimum (H100 80GB) |

<img src="assets/llama_3_405b_technical_blueprint_p13.png" alt="Deployment optimization and inference scaling for Llama 3 405B including pipeline parallelism, FP8 quantization, and KV-cache management" width="100%" />

*Figure. Deployment optimization and inference scaling, corresponding to the memory and topology constraints summarized in Section 11.*

Practical deployment requires TP=8 minimum (single node), with PP=2 for full 405B model.

### 11.2 Serving Topology

- **Tensor parallelism:** TP=8 within single server (NVLink)
- **Pipeline parallelism:** PP≥2 across servers (RoCE/InfiniBand)
- **KV cache management:** GQA with $n_{\text{kv}}=8$ reduces cache by $16\times$ vs MHA

### 11.3 Checkpointing and Recovery

- Checkpoint saves per-GPU model state (1 MB to 4 GB per GPU)
- Minimized GPU pause time during checkpointing
- High checkpoint frequency to minimize lost work on failure
- Tectonic storage fabric handles bursty writes (peak 7 TB/s)

### 11.4 Fault Tolerance

- >90% effective training time achieved
- Automated recovery from most hardware failures
- Only 3 manual interventions in 54-day window
- Flight recorder-based diagnostics for rapid root-cause identification

### 11.5 Power and Thermal Constraints

- Synchronized power fluctuations of tens of MW across datacenter during collective operations
- Diurnal 1-2% throughput variation from temperature-dependent DVFS
- Ongoing challenge for future larger-scale training

---

## 12. Loss Formulations Summary

### 12.1 Pre-Training Loss

$$\mathcal{L}_{\text{PT}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})$$

### 12.2 SFT Loss

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[\frac{1}{|y|}\sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})\right]$$

### 12.3 DPO Loss

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

### 12.4 Scaling Law

$$N^{\star}(C) = A \cdot C^{\alpha}, \quad (\alpha, A) = (0.53, 0.29)$$

$$C_{\text{total}} \approx 6 \cdot N \cdot D$$

---

## 13. Convergence Dynamics

### 13.1 Training Stability

- Progressive batch size schedule avoids early-training instability
- Standard dense Transformer (no MoE) chosen explicitly for training stability
- Weight decay coupled to learning rate: $\lambda_t = 0.1 \cdot \eta_t$
- Few loss spikes observed; no interventions for divergence

### 13.2 Post-Training Convergence

- Multiple rounds of SFT → RS → DPO
- Each round uses the previous round's model as initialization
- DPO uses current SFT checkpoint as reference policy $\pi_{\text{ref}}$
- Rejection sampling provides on-policy data for progressive improvement

### 13.3 Polyak Averaging

Final pre-trained model is the average of checkpoints during annealing:

$$\theta_{\text{final}} = \frac{1}{K}\sum_{k=1}^{K} \theta_{t_k}$$

This smooths parameter estimates and typically reduces variance in final model quality.

---

## 14. Critical Design Decisions and Trade-offs

| Decision | Choice | Alternative Considered | Rationale |
|---|---|---|---|
| Architecture | Dense Transformer | Mixture-of-Experts | Training stability, scaling simplicity |
| Attention heads | GQA ($n_{\text{kv}}=8$) | MHA, MQA | Balance between quality (near MHA) and inference speed (near MQA) |
| Post-training | SFT + RS + DPO | PPO/RLHF | Stability, scalability, simplicity |
| Positional encoding | RoPE ($\theta=500\text{K}$) | ALiBi, learned | Long-context support, proven effectiveness |
| Tokenizer | BPE 128K vocab | Smaller vocab | Better compression, multilingual support |
| Context extension | Gradual 6-stage | Train from scratch at 128K | Compute efficiency ($\mathcal{O}(T^2)$ cost management) |
| Multimodal integration | Compositional (frozen LLM) | End-to-end joint training | Preserves language model quality; modular development |
| Parallelism | 4D (TP+CP+PP+DP) | 2D or 3D parallelism | Memory + compute efficiency at 16K GPU scale |
| CP implementation | All-gather based | Ring-based | Flexibility for document masks; GQA makes K,V small |

---

## 15. Failure Modes

| Failure Mode | Manifestation | Mitigation in Llama 3 |
|---|---|---|
| Training divergence | Loss spikes, NaN gradients | Progressive batch size, stable architecture, FP32 grad accumulation |
| Hardware failure | GPU/HBM/NIC faults causing job restart | Automated recovery, flight recorder diagnostics, frequent checkpointing |
| Straggler GPUs | Single slow GPU bottlenecking thousands | Communication prioritization tools, straggler detection |
| Silent data corruption | Incorrect computation without error signal | Monitoring, automated detection, GPU validation |
| NVLink failure | CUDA kernel stalls on load/store | NCCLX watchdog timeout, automatic failure localization |
| Pipeline imbalance | First/last stage memory/compute bottleneck | Remove one layer from first/last stages; balanced pipeline schedule |
| Data contamination | Benchmark data leaking into training | Explicit exclusion of benchmark training sets |
| Quality regression during long-context training | Short-context performance degradation | Staged extension with recovery verification |
| Over-optimization in DPO | Model diverges from reference, reward hacking | KL penalty via $\beta$ parameter, multiple rounds with fresh reference |
| Markdown artifacts in web data | Performance degradation from markdown tokens | Complete markdown marker removal |
