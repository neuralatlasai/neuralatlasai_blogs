

# Qwen3 Technical Report: End-to-End System Analysis

---

## 1. Data Pipeline

### 1.1 Definition & Objectives

The data pipeline for Qwen3 constitutes a multi-source, multi-stage curation system that ingests, transforms, annotates, filters, and mixes heterogeneous corpora into a unified pre-training dataset of **36 trillion tokens** spanning **119 languages and dialects**.

**Formal Objective:**

$$\mathcal{D}^* = \arg\max_{\mathcal{D} \subseteq \mathcal{U}} \; \mathbb{E}_{x \sim \mathcal{D}}\left[\text{Quality}(x)\right] \quad \text{s.t.} \quad |\mathcal{D}| = 36 \times 10^{12}, \; \text{LangCov}(\mathcal{D}) \geq 119, \; \text{DomainDiv}(\mathcal{D}) \geq \tau_{\text{div}}$$

where $\mathcal{U}$ is the universal crawled/collected corpus, $\text{Quality}(x)$ is a multi-dimensional quality score per instance $x$, $\text{LangCov}$ counts distinct language/dialect coverage, and $\tau_{\text{div}}$ is a domain diversity threshold.

**Inputs:** Raw web crawls, books, code repositories, PDF documents, multilingual corpora, existing Qwen2.5 family model outputs.

**Outputs:** Tokenized, annotated, deduplicated, mixed training shards with per-instance metadata (educational value, field, domain, safety).

**Invariants:**
- No contamination with downstream evaluation sets.
- Language distribution respects target coverage proportions.
- Instance-level quality annotations are consistent across annotation rounds.

### 1.2 Source Acquisition & Expansion

**1.2.1 PDF Text Extraction Pipeline**

A two-step OCR-then-refinement pipeline:

1. **Step 1 — Visual Text Recognition:** Qwen2.5-VL (vision-language model) is fine-tuned to perform text recognition on large-volume PDF-like documents.

$$\hat{t} = f_{\text{VL}}(\mathbf{I}_{\text{page}}; \theta_{\text{VL}})$$

where $\mathbf{I}_{\text{page}} \in \mathbb{R}^{H \times W \times 3}$ is a rendered page image and $\hat{t}$ is the recognized text string.

2. **Step 2 — Text Refinement:** Qwen2.5 (text LLM) refines the OCR output to correct recognition errors, formatting artifacts, and structural inconsistencies.

$$t^* = g_{\text{LLM}}(\hat{t}; \theta_{\text{LLM}})$$

**Yield:** Trillions of additional high-quality text tokens from previously inaccessible document sources.

**Pseudo-Algorithm: PDF Text Extraction**
```
PROCEDURE ExtractPDFText(pdf_corpus):
    FOR EACH document d IN pdf_corpus:
        pages ← RenderPages(d)
        FOR EACH page p IN pages:
            raw_text ← Qwen2.5_VL_Recognize(p)
            refined_text ← Qwen2.5_Refine(raw_text)
            EMIT (refined_text, metadata(d, p))
    END FOR
END PROCEDURE
```

**1.2.2 Synthetic Data Generation**

Three specialized generators produce synthetic tokens in structured formats:

| Generator | Domain | Formats |
|-----------|--------|---------|
| Qwen2.5 | General | Textbooks, QA, Instructions |
| Qwen2.5-Math | Mathematics | Problem-solution pairs, proofs |
| Qwen2.5-Coder | Code | Code snippets, docstrings, test cases |

**Scale:** Trillions of synthetic tokens across dozens of domains.

**1.2.3 Multilingual Expansion**

Language coverage expanded from 29 (Qwen2.5) to 119 languages/dialects by incorporating additional multilingual web data, parallel corpora, and monolingual resources.

### 1.3 Multi-Dimensional Annotation System

**Annotation Dimensions:**
- Educational value $e(x) \in [0, 1]$
- Field classification $f(x) \in \mathcal{F}$ (taxonomy of academic/professional fields)
- Domain classification $d(x) \in \mathcal{D}$ (content domain)
- Safety score $s(x) \in [0, 1]$

**Scale:** Over 30 trillion tokens annotated across all dimensions.

**Annotation Model:** Qwen2.5-72B-Instruct serves as the annotation backbone, providing instance-level labels.

$$\mathbf{a}(x) = [e(x), f(x), d(x), s(x)] = h_{\text{annotator}}(x; \theta_{\text{ann}})$$

### 1.4 Instance-Level Data Mixing

**Key Innovation:** Unlike prior work optimizing data mixture at source/domain level (Xie et al., 2023; Fan et al., 2023; Liu et al., 2024b), Qwen3 optimizes at the **instance level** using fine-grained labels.

**Optimization Procedure:**

1. Define candidate mixture distributions $\pi(\mathbf{a})$ over annotation vectors.
2. Train small proxy models $M_{\text{proxy}}$ under candidate mixtures.
3. Evaluate proxy models on held-out validation.
4. Select optimal mixture via extensive ablation:

$$\pi^* = \arg\max_{\pi} \; \text{ValPerf}\left(M_{\text{proxy}}(\pi)\right)$$

**Pseudo-Algorithm: Instance-Level Mixing**
```
PROCEDURE OptimizeMixture(annotated_corpus, proxy_arch, val_set):
    candidates ← GenerateMixtureCandidates(annotated_corpus)
    best_perf ← -∞
    FOR EACH π IN candidates:
        D_train ← SampleFromCorpus(annotated_corpus, π)
        M ← TrainProxyModel(proxy_arch, D_train)
        perf ← Evaluate(M, val_set)
        IF perf > best_perf:
            best_perf ← perf
            π* ← π
    RETURN π*
END PROCEDURE
```

### 1.5 Tokenization

**Method:** Byte-level Byte-Pair Encoding (BBPE)

**Vocabulary Size:** $|\mathcal{V}| = 151{,}669$

**Properties:**
- Byte-level fallback ensures lossless encoding of any UTF-8 input.
- Shared tokenizer across all Qwen3 model sizes (0.6B–235B).
- Vocabulary inherited from Qwen tokenizer family (Bai et al., 2023).

**Compression Equation:** For input byte sequence $\mathbf{b} = (b_1, \ldots, b_N)$, the tokenizer produces token sequence $\mathbf{t} = (t_1, \ldots, t_L)$ where $L \leq N$ and:

$$\text{CompressionRatio} = \frac{N}{L}$$

**Information Preservation Invariant:**

$$\text{Decode}(\text{Encode}(\mathbf{b})) = \mathbf{b} \quad \forall \; \mathbf{b} \in \{0,\ldots,255\}^*$$

This is guaranteed by the byte-level fallback mechanism.

### 1.6 Failure Modes — Data Pipeline

| Failure Mode | Description | Mitigation |
|---|---|---|
| OCR hallucination | VL model generates text not present in PDF | Two-step refinement with LLM |
| Synthetic data collapse | Generator produces repetitive/low-diversity samples | Domain-specific generators, diversity filters |
| Annotation inconsistency | Annotator assigns conflicting labels | Calibration, majority voting on subsamples |
| Contamination | Evaluation data leaks into training | Decontamination via n-gram matching against eval sets |
| Language imbalance | Over-representation of high-resource languages | Instance-level mixing optimization |

---

## 2. Model Architecture

### 2.1 Dense Architecture

#### 2.1.1 Formal Specification

Each Qwen3 dense model is a decoder-only Transformer with the following components:

**Core Blocks:**
- **Grouped Query Attention (GQA)** with QK-Norm
- **SwiGLU** feed-forward network
- **Rotary Positional Embeddings (RoPE)**
- **RMSNorm** with pre-normalization
- **No QKV-bias** (removed from Qwen2)

**Architecture Table (Dense):**

| Model | Layers $L$ | Q Heads $n_q$ | KV Heads $n_{kv}$ | Tie Embedding | Context |
|-------|-----------|--------------|-----------------|---------------|---------|
| 0.6B | 28 | 16 | 8 | Yes | 32K |
| 1.7B | 28 | 16 | 8 | Yes | 32K |
| 4B | 36 | 32 | 8 | No | 128K |
| 8B | 36 | 32 | 8 | No | 128K |
| 14B | 40 | 40 | 8 | No | 128K |
| 32B | 64 | 64 | 8 | No | 128K |

#### 2.1.2 Mathematical Formulation — Single Transformer Block

Let $\mathbf{X} \in \mathbb{R}^{T \times d}$ be the input hidden states for sequence length $T$ and hidden dimension $d$.

**Pre-Normalization:**

$$\hat{\mathbf{X}} = \text{RMSNorm}(\mathbf{X})$$

where:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

$\boldsymbol{\gamma} \in \mathbb{R}^d$ is a learnable scale parameter, $\epsilon$ is a small constant for numerical stability.

**Grouped Query Attention (GQA):**

Query, Key, Value projections with $n_q$ query heads and $n_{kv}$ KV heads:

$$\mathbf{Q} = \hat{\mathbf{X}} \mathbf{W}_Q \in \mathbb{R}^{T \times n_q \times d_h}$$
$$\mathbf{K} = \hat{\mathbf{X}} \mathbf{W}_K \in \mathbb{R}^{T \times n_{kv} \times d_h}$$
$$\mathbf{V} = \hat{\mathbf{X}} \mathbf{W}_V \in \mathbb{R}^{T \times n_{kv} \times d_h}$$

where $d_h = d / n_q$ is the per-head dimension. Each group of $G = n_q / n_{kv}$ query heads shares one KV head.

**QK-Norm (New in Qwen3):**

Before computing attention scores, queries and keys are normalized:

$$\tilde{\mathbf{Q}}_{h} = \text{RMSNorm}(\mathbf{Q}_h), \quad \tilde{\mathbf{K}}_{h} = \text{RMSNorm}(\mathbf{K}_h)$$

This stabilizes training by bounding the dot product magnitude, preventing attention logit explosion at large model scales.

**Stability Condition:** Without QK-Norm, the variance of attention logits scales as:

$$\text{Var}\left(\mathbf{q}^\top \mathbf{k}\right) \propto d_h$$

With QK-Norm applied:

$$\text{Var}\left(\tilde{\mathbf{q}}^\top \tilde{\mathbf{k}}\right) = O(1)$$

independent of $d_h$, providing controlled gradient flow.

**RoPE Application:**

For position $m$ and dimension pair $(2i, 2i+1)$:

$$\text{RoPE}(\mathbf{x}, m)_{2i} = x_{2i}\cos(m\theta_i) - x_{2i+1}\sin(m\theta_i)$$
$$\text{RoPE}(\mathbf{x}, m)_{2i+1} = x_{2i}\sin(m\theta_i) + x_{2i+1}\cos(m\theta_i)$$

where $\theta_i = b^{-2i/d_h}$ with base frequency $b$. During pre-training Stage 1–2: $b = 10{,}000$. During Stage 3 (long-context): $b = 1{,}000{,}000$ via ABF (Adjusted Base Frequency).

**Causal Attention Computation:**

$$\mathbf{A}_{h} = \text{softmax}\left(\frac{\tilde{\mathbf{Q}}_h \tilde{\mathbf{K}}_h^\top}{\sqrt{d_h}} + \mathbf{M}\right) \mathbf{V}_h$$

where $\mathbf{M}$ is the causal mask with $M_{ij} = -\infty$ for $j > i$.

**KV Head Broadcasting:** For GQA with group size $G$:

$$\mathbf{K}_{\text{broadcast}} = \text{repeat\_interleave}(\mathbf{K}, G, \text{dim}=\text{head})$$

This reduces KV cache memory by factor $G$ at inference.

**Output Projection and Residual:**

$$\mathbf{Y}_{\text{attn}} = \mathbf{X} + \text{Concat}(\mathbf{A}_1, \ldots, \mathbf{A}_{n_q}) \mathbf{W}_O$$

**SwiGLU Feed-Forward Network:**

$$\hat{\mathbf{Y}} = \text{RMSNorm}(\mathbf{Y}_{\text{attn}})$$
$$\mathbf{Y}_{\text{FFN}} = \mathbf{Y}_{\text{attn}} + \left[\text{SiLU}(\hat{\mathbf{Y}} \mathbf{W}_{\text{gate}}) \odot (\hat{\mathbf{Y}} \mathbf{W}_{\text{up}})\right] \mathbf{W}_{\text{down}}$$

where $\mathbf{W}_{\text{gate}}, \mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times d_{\text{ff}}}$, $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d}$, and $\text{SiLU}(x) = x \cdot \sigma(x)$.

**Tie Embedding:** For models with $\leq 1.7$B parameters, the output projection matrix $\mathbf{W}_{\text{lm\_head}}$ shares parameters with the input embedding matrix $\mathbf{W}_{\text{emb}}$:

$$\mathbf{W}_{\text{lm\_head}} = \mathbf{W}_{\text{emb}}^\top$$

This reduces parameter count by $|\mathcal{V}| \times d$ parameters.

#### 2.1.3 Complexity Analysis (Dense)

For sequence length $T$, hidden dimension $d$, and $L$ layers:

$$\text{Attention FLOPs per layer} = O(T^2 \cdot d)$$
$$\text{FFN FLOPs per layer} = O(T \cdot d \cdot d_{\text{ff}})$$
$$\text{Total FLOPs} = O(L \cdot T \cdot (T \cdot d + d \cdot d_{\text{ff}}))$$

**KV Cache Memory (GQA):**

$$\text{KV Cache} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot T \cdot \text{sizeof(dtype)}$$

Compared to MHA ($n_{kv} = n_q$), GQA reduces KV cache by factor $n_q / n_{kv}$.

### 2.2 MoE Architecture

#### 2.2.1 Formal Specification

**Architecture Table (MoE):**

| Model | Layers $L$ | Q / KV Heads | Total Experts $E$ | Active Experts $k$ | Context |
|-------|-----------|-------------|-------------------|--------------------|---------| 
| 30B-A3B | 48 | 32 / 4 | 128 | 8 | 128K |
| 235B-A22B | 94 | 64 / 4 | 128 | 8 | 128K |

**Key Design Differences from Qwen2.5-MoE:**
1. **No shared experts** — all 128 experts are routed; no dedicated shared FFN.
2. **Fine-grained expert segmentation** following Dai et al. (2024) — each expert is smaller, enabling more precise routing.
3. **Global-batch load balancing loss** (Qiu et al., 2025) — replaces per-sample auxiliary losses.

#### 2.2.2 MoE Router and Expert Selection

For input token representation $\mathbf{x} \in \mathbb{R}^d$, the router computes:

$$\mathbf{g} = \text{softmax}(\mathbf{x} \mathbf{W}_r) \in \mathbb{R}^{E}$$

where $\mathbf{W}_r \in \mathbb{R}^{d \times E}$ is the router weight matrix.

**Top-k Selection:**

$$\mathcal{S}(\mathbf{x}) = \text{TopK}(\mathbf{g}, k) \quad \text{where } k = 8$$

**Expert Computation:**

$$\text{MoE-FFN}(\mathbf{x}) = \sum_{e \in \mathcal{S}(\mathbf{x})} \frac{g_e}{\sum_{e' \in \mathcal{S}} g_{e'}} \cdot \text{FFN}_e(\mathbf{x})$$

where each $\text{FFN}_e$ is a SwiGLU block with reduced dimensions (fine-grained segmentation).

#### 2.2.3 Global-Batch Load Balancing Loss

Instead of per-sample auxiliary loss, the load balancing operates over the global batch $\mathcal{B}$:

$$\mathcal{L}_{\text{bal}} = \alpha \cdot E \sum_{e=1}^{E} f_e \cdot \bar{g}_e$$

where:

$$f_e = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} \mathbb{1}[e \in \mathcal{S}(x)]$$

$$\bar{g}_e = \frac{1}{|\mathcal{B}|} \sum_{x \in \mathcal{B}} g_e(x)$$

$\alpha$ is the balancing coefficient. Global-batch computation encourages **expert specialization** by allowing temporary imbalance within individual sequences while enforcing statistical balance across the entire batch.

**Advantage over per-sample balancing:** Prevents the router from distributing tokens uniformly within each sample (which destroys specialization), instead enforcing balance only at the aggregate level.

#### 2.2.4 Complexity Analysis (MoE)

**FLOPs per token:**

$$\text{FLOPs}_{\text{MoE}} = O(L \cdot (T \cdot d + k \cdot d \cdot d_{\text{ff}}^{(e)}))$$

where $d_{\text{ff}}^{(e)} = d_{\text{ff}} / (E/k)$ is the per-expert FFN dimension under fine-grained segmentation.

**Activated Parameters:**

$$P_{\text{active}} = P_{\text{attn}} + L \cdot k \cdot P_{\text{expert}}$$

For Qwen3-235B-A22B: $P_{\text{total}} = 235$B, $P_{\text{active}} = 22$B, yielding activation ratio $\approx 9.4\%$.

**Memory:** The full 235B parameters must be loaded; only 22B execute per token. KV cache is determined by attention parameters (not MoE), so:

$$\text{KV Cache}_{\text{MoE}} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot T \cdot \text{sizeof(dtype)}$$

With $n_{kv} = 4$ and $L = 94$, this is heavily compressed.

### 2.3 Architectural Comparison: Dense vs. MoE

| Property | Dense-32B | MoE-30B-A3B | MoE-235B-A22B |
|----------|-----------|-------------|---------------|
| Total Params | 32B | 30B | 235B |
| Active Params | 32B | 3B | 22B |
| KV Heads | 8 | 4 | 4 |
| Experts | 1 | 128/8 | 128/8 |
| Shared Experts | — | 0 | 0 |
| Inference Cost | High | Very Low | Moderate |

**Key Finding from Evaluation:** Qwen3 MoE base models achieve similar performance to dense models with **1/5 activated parameters**, and comparable performance to Qwen2.5 dense models with **1/10 activated parameters**.

---

## 3. Pre-Training Pipeline

### 3.1 Three-Stage Pre-Training Strategy

#### Stage 1: General Stage (S1)

**Objective:** Build broad language proficiency and general world knowledge.

$$\mathcal{L}_{\text{S1}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

- **Data:** >30 trillion tokens, 119 languages/dialects
- **Sequence Length:** 4,096 tokens
- **RoPE Base:** $b = 10{,}000$

**Batch Size & Learning Rate:** Predicted via scaling laws (see §3.4).

#### Stage 2: Reasoning Stage (S2)

**Objective:** Enhance reasoning in STEM, coding, mathematics.

$$\mathcal{L}_{\text{S2}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

- **Data:** ~5 trillion higher-quality tokens with increased STEM/code/reasoning/synthetic proportion
- **Sequence Length:** 4,096 tokens
- **Learning Rate Decay:** Accelerated during this stage (faster cosine or linear decay)

**Curriculum Design:** The data mixture shifts from the S1 distribution to a reasoning-enriched distribution:

$$\pi_{\text{S2}}(d) = \pi_{\text{S1}}(d) \cdot w_{\text{reasoning}}(d)$$

where $w_{\text{reasoning}}(d) > 1$ for STEM/code/reasoning domains and $w_{\text{reasoning}}(d) < 1$ for general web data.

#### Stage 3: Long Context Stage

**Objective:** Extend effective context from 4,096 to 32,768 tokens (with 4× inference extension to 131,072).

- **Data:** Hundreds of billions of tokens
  - 75%: sequences of 16,384–32,768 tokens
  - 25%: sequences of 4,096–16,384 tokens
- **Sequence Length:** 32,768 tokens
- **RoPE Base:** $b = 1{,}000{,}000$ (via ABF technique)

**ABF (Adjusted Base Frequency):**

$$\theta_i^{\text{ABF}} = b_{\text{new}}^{-2i/d_h} = (10^6)^{-2i/d_h}$$

This increases the wavelength of all RoPE frequency components by factor $\sqrt[d_h]{b_{\text{new}}/b_{\text{old}}} = \sqrt[d_h]{100}$, enabling extrapolation to longer sequences.

**YARN (Yet Another RoPE extensioN):**

Applies a non-uniform scaling to different frequency bands:

$$\theta_i^{\text{YARN}} = \begin{cases} \theta_i & \text{if } \lambda_i < \lambda_{\min} \\ \theta_i / s & \text{if } \lambda_i > \lambda_{\max} \\ (1-\alpha_i)\theta_i + \alpha_i \theta_i/s & \text{otherwise} \end{cases}$$

where $\lambda_i = 2\pi / \theta_i$ is the wavelength, $s$ is the scale factor, and $\alpha_i$ interpolates in the transition band.

**Dual Chunk Attention (DCA):**

Segments the long sequence into chunks and computes intra-chunk and inter-chunk attention with relative position adjustment, achieving effective 4× context extension during inference:

$$T_{\text{inference}} = 4 \times T_{\text{train}} = 4 \times 32{,}768 = 131{,}072$$

### 3.2 Scaling Laws for Hyperparameter Prediction

**Formal Framework:** The optimal hyperparameters (learning rate $\eta^*$, batch size $B^*$) depend on:
- Model architecture parameters $\Theta_{\text{arch}}$ (layers, width, heads)
- Dataset size and stage index $s$
- Compute budget $C$

$$(\eta^*, B^*) = \text{ScalingLaw}(\Theta_{\text{arch}}, |\mathcal{D}_s|, s, C)$$

**Methodology:** Extensive experiments on small proxy models systematically study the relationship:

$$\text{ValLoss} = f(\eta, B, \Theta_{\text{arch}}, |\mathcal{D}_s|)$$

The predicted optimal values are then applied to each dense or MoE model configuration.

**Pseudo-Algorithm: Scaling Law Hyperparameter Prediction**
```
PROCEDURE PredictHyperparams(model_config, stage, data_size):
    FOR EACH (η, B) IN grid:
        proxy_model ← InitializeProxy(model_config, scale=small)
        loss ← TrainAndEvaluate(proxy_model, η, B, stage, data_size_scaled)
        Record(η, B, loss)
    FIT scaling_function TO recorded (η, B, loss) points
    (η*, B*) ← MINIMIZE scaling_function OVER (η, B) AT full scale
    RETURN (η*, B*)
END PROCEDURE
```

### 3.3 Pre-Training Pseudo-Algorithm

```
PROCEDURE PretrainQwen3(model, tokenized_data, stage_configs):
    FOR stage IN [S1, S2, S3]:
        data ← SelectData(tokenized_data, stage.mixture)
        seq_len ← stage.sequence_length
        (η, B) ← PredictHyperparams(model.config, stage, |data|)
        IF stage == S3:
            model.rope_base ← 1,000,000  // ABF
            EnableYARN(model)
            EnableDCA(model)
        optimizer ← AdamW(model.parameters(), lr=η, ...)
        scheduler ← stage.lr_schedule  // accelerated decay for S2
        
        FOR step IN 1 TO stage.max_steps:
            batch ← SampleBatch(data, B, seq_len)
            tokens ← Tokenize(batch)  // BBPE, |V|=151,669
            logits ← model(tokens[:, :-1])
            loss ← CrossEntropy(logits, tokens[:, 1:])
            IF model.is_moe:
                loss ← loss + L_balance  // global-batch load balancing
            loss.backward()
            ClipGradients(model, max_norm)
            optimizer.step()
            scheduler.step()
            Checkpoint(model, step)
    RETURN model
END PROCEDURE
```

### 3.4 Training Loss Formulation

**Dense Model Total Loss:**

$$\mathcal{L}_{\text{dense}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t})$$

**MoE Model Total Loss:**

$$\mathcal{L}_{\text{MoE}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t | x_{<t}) + \alpha \cdot \mathcal{L}_{\text{bal}}$$

where $\mathcal{L}_{\text{bal}}$ is the global-batch load balancing loss defined in §2.2.3.

### 3.5 Key Pre-Training Results

**Performance Compression Ratios (parameter efficiency):**

| Qwen3 Model | Matches Qwen2.5 Model | Parameter Ratio |
|-------------|----------------------|-----------------|
| 1.7B | 3B | 0.57× |
| 4B | 7B | 0.57× |
| 8B | 14B | 0.57× |
| 14B | 32B | 0.44× |
| 32B | 72B | 0.44× |
| 30B-A3B (MoE) | 14B (Dense) | 0.21× active |
| 235B-A22B (MoE) | 72B (Dense) | 0.31× active |

**Flagship Comparison:** Qwen3-235B-A22B-Base outperforms DeepSeek-V3-Base on 14/15 benchmarks with 1/3 total parameters and 2/3 activated parameters.

---

## 4. Post-Training Pipeline

### 4.1 Pipeline Overview

The post-training pipeline has **four stages** for flagship models and a **distillation shortcut** for lightweight models:

$$\text{Base} \xrightarrow{\text{S1}} \text{CoT-Init} \xrightarrow{\text{S2}} \text{Reasoning-RL} \xrightarrow{\text{S3}} \text{Fused} \xrightarrow{\text{S4}} \text{General-RL}$$

**Core Objectives:**
1. **Thinking Control:** Unified model supporting both thinking (deep reasoning) and non-thinking (fast response) modes.
2. **Strong-to-Weak Distillation:** Transfer capability from flagship to lightweight models efficiently.

### 4.2 Stage 1: Long-CoT Cold Start

#### 4.2.1 Objective

Instill foundational long Chain-of-Thought reasoning patterns without over-fitting to specific reasoning performance.

$$\mathcal{L}_{\text{cold}} = -\frac{1}{|\mathcal{D}_{\text{cold}}|}\sum_{(q,r) \in \mathcal{D}_{\text{cold}}} \frac{1}{|r|}\sum_{t=1}^{|r|} \log p_\theta(r_t | q, r_{<t})$$

where $r$ is a verified long-CoT response.

#### 4.2.2 Data Construction

**Query Filtering (Phase 1):**

Using Qwen2.5-72B-Instruct as filter:

1. Remove non-verifiable queries (multiple sub-questions, general text generation).
2. Remove queries solvable without CoT (to prevent superficial guessing).
3. Annotate domain for balanced representation.

**Response Generation (Phase 2):**

For each filtered query $q$:
1. Generate $N$ candidate responses using QwQ-32B.
2. For consistent failures: manual human assessment.
3. For Pass@$N > 0$: apply stringent filtering:
   - Remove incorrect final answers
   - Remove substantial repetition
   - Remove guesswork without reasoning
   - Remove thinking/summary inconsistencies
   - Remove inappropriate language mixing
   - Remove potential validation set contamination

**Pseudo-Algorithm: Cold Start Data Construction**
```
PROCEDURE ConstructColdStartData(raw_queries):
    // Phase 1: Query Filtering
    filtered_queries ← ∅
    FOR EACH q IN raw_queries:
        IF NOT IsVerifiable(q, Qwen2.5-72B):
            CONTINUE
        IF SolvableWithoutCoT(q, Qwen2.5-72B):
            CONTINUE
        domain ← AnnotateDomain(q, Qwen2.5-72B)
        filtered_queries ← filtered_queries ∪ {(q, domain)}
    
    // Reserve validation set
    val_queries, train_queries ← Split(filtered_queries)
    
    // Phase 2: Response Generation & Filtering
    cold_start_data ← ∅
    FOR EACH q IN train_queries:
        responses ← GenerateN(QwQ-32B, q, N)
        IF PassAtN(responses) == 0:
            manual_responses ← HumanAssess(q, responses)
            responses ← manual_responses
        filtered_responses ← ApplyStringentFilters(responses)
        cold_start_data ← cold_start_data ∪ {(q, filtered_responses)}
    
    // Select minimal subset for cold start
    RETURN SelectSubset(cold_start_data, min_size=True)
END PROCEDURE
```

#### 4.2.3 Training Strategy

**Key Principle:** Minimize both sample count and training steps. The goal is to initialize reasoning patterns, not maximize reasoning performance. This preserves model plasticity for the subsequent RL stage.

**Invariant:** Cold-start model should show reasoning structure but not saturated performance — ensuring the RL stage has room for exploration and improvement.

### 4.3 Stage 2: Reasoning RL

#### 4.3.1 Query-Verifier Pair Selection

Four criteria for inclusion:

1. **Not used during cold start** (no data overlap)
2. **Learnable** for the cold-start model (within capability frontier)
3. **Maximally challenging** (pushing the boundary)
4. **Broad sub-domain coverage**

**Final Dataset:** 3,995 query-verifier pairs.

#### 4.3.2 GRPO (Group Relative Policy Optimization)

**Objective:** Update policy $\pi_\theta$ using group-relative advantages without a critic network.

For query $q$, generate a group of $G$ rollout responses $\{r_1, \ldots, r_G\} \sim \pi_\theta(\cdot | q)$.

**Reward Computation:**

$$R_i = \text{Verifier}(q, r_i) \in \{0, 1\} \quad \text{(rule-based for math/code)}$$

**Group-Relative Advantage:**

$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}$$

**Policy Gradient Loss:**

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\left(\rho_i \hat{A}_i, \; \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon)\hat{A}_i\right) + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

where $\rho_i = \frac{\pi_\theta(r_i|q)}{\pi_{\text{old}}(r_i|q)}$ is the importance ratio, $\varepsilon$ is the clipping parameter, and $\beta$ controls KL regularization against the reference policy $\pi_{\text{ref}}$.

#### 4.3.3 Training Configuration

| Hyperparameter | Design Choice |
|---|---|
| Batch Size | Large |
| Rollouts per Query | High |
| Off-policy Training | Enabled (improved sample efficiency) |
| Entropy Control | Steady increase or stable (prevents mode collapse) |
| Manual Intervention | None over 170 steps |

**Training Dynamics (Qwen3-235B-A22B):**

$$\text{AIME'24}: 70.1 \xrightarrow{170 \text{ RL steps}} 85.1$$

**Entropy Management:**

The entropy of the policy distribution is monitored and controlled:

$$H(\pi_\theta) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$$

Steady or increasing entropy prevents premature convergence and maintains exploration capability.

**Pseudo-Algorithm: Reasoning RL**
```
PROCEDURE ReasoningRL(model, query_verifier_pairs):
    π_ref ← copy(model)
    optimizer ← AdamW(model.parameters(), lr_rl)
    
    FOR step IN 1 TO 170:
        batch_queries ← SampleBatch(query_verifier_pairs, B_rl)
        
        FOR EACH q IN batch_queries:
            rollouts ← GenerateGroup(model, q, G)
            rewards ← [Verifier(q, r) FOR r IN rollouts]
            advantages ← GroupRelativeNormalize(rewards)
            
            FOR EACH (r_i, A_i) IN zip(rollouts, advantages):
                ρ_i ← π_θ(r_i|q) / π_old(r_i|q)
                loss_i ← -min(ρ_i * A_i, clip(ρ_i) * A_i)
                kl_i ← KL(π_θ(·|q) || π_ref(·|q))
        
        total_loss ← mean(loss_terms) + β * mean(kl_terms)
        total_loss.backward()
        optimizer.step()
        
        // Monitor entropy
        entropy ← ComputeEntropy(model, sample_queries)
        ASSERT entropy ≥ entropy_floor  // prevent mode collapse
    
    RETURN model
END PROCEDURE
```

#### 4.3.4 Failure Modes — Reasoning RL

| Failure Mode | Manifestation | Mitigation |
|---|---|---|
| Reward hacking | Model exploits verifier loopholes | Rule-based verifiers with test cases |
| Entropy collapse | Model converges to narrow response distribution | Entropy monitoring and control |
| Over-optimization | Training reward increases but validation degrades | Separate validation set monitoring |
| Catastrophic forgetting | General capabilities degrade | KL regularization against reference |

### 4.4 Stage 3: Thinking Mode Fusion

#### 4.4.1 Objective

Integrate non-thinking capabilities into the thinking model via continual SFT, enabling a **single model** to serve both modes.

#### 4.4.2 Chat Template Design

**Thinking Mode (default):**

```
<|im_start|>user
{query} /think<|im_end|>
<|im_start|>assistant
<think>{thinking content}</think>
{response}<|im_end|>
```

**Non-Thinking Mode:**

```
<|im_start|>user
{query} /no_think<|im_end|>
<|im_start|>assistant
<think>
</think>
{response}<|im_end|>
```

**Design Properties:**
- `/think` flag is optional (thinking is default behavior).
- `/no_think` produces an **empty think block** (`<think>\n</think>`) — maintains format consistency.
- Multi-turn dialogs: model follows the **last flag encountered**.
- Implementation via Hugging Face tokenizer parameter `enable_thinking=False`.

#### 4.4.3 SFT Data Construction

**Thinking Data:** Generated via rejection sampling on Stage 1 queries using Stage 2 model:

$$r_{\text{think}} \sim \pi_{\text{S2}}(\cdot | q), \quad \text{filtered by correctness}$$

This ensures thinking data matches the current model's capability level.

**Non-Thinking Data:** Curated to cover:
- Coding, mathematics, instruction-following
- Multilingual tasks (with increased translation task proportion for low-resource languages)
- Creative writing, QA, role-playing

**Quality Assessment:** Automatically generated checklists evaluate non-thinking response quality.

#### 4.4.4 Training Loss

$$\mathcal{L}_{\text{fusion}} = -\frac{1}{|\mathcal{D}_{\text{think}}| + |\mathcal{D}_{\text{no\_think}}|}\left[\sum_{(q,r) \in \mathcal{D}_{\text{think}}} \log p_\theta(r | q) + \sum_{(q,r) \in \mathcal{D}_{\text{no\_think}}} \log p_\theta(r | q)\right]$$

where responses in $\mathcal{D}_{\text{no\_think}}$ include the empty think block prefix.

#### 4.4.5 Thinking Budget (Emergent Capability)

**Definition:** After fusion training, the model naturally handles **partial thinking** — generating a response from incomplete reasoning.

**Mechanism:** When thinking length reaches user-defined threshold $B_{\text{think}}$:

1. Halt generation.
2. Insert stop-thinking instruction:
   `"Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"`
3. Model generates final response from accumulated partial reasoning.

**Key Property:** This capability is **not explicitly trained** — it emerges from the fusion of thinking and non-thinking modes.

**Formal Characterization:**

$$p_\theta(\text{response} | q, \text{think}_{1:B}) \approx p_\theta(\text{response} | q, \text{think}_{1:T}) \quad \text{for } B \leq T$$

The model generalizes from trained endpoints ($B = 0$ for no-thinking, $B = T$ for full thinking) to intermediate values $0 < B < T$.

**Empirical Result (Figure 2):** Scalable and smooth performance improvements correlated to thinking budget across Mathematics, Coding, and STEM domains.

### 4.5 Stage 4: General RL

#### 4.5.1 Objective

Broadly enhance capabilities and stability across **20+ distinct tasks** with customized scoring criteria.

#### 4.5.2 Task Categories and Targeted Capabilities

| Capability | Description | Examples |
|---|---|---|
| Instruction Following | Accurate interpretation of content, format, length, structured output | IFEval, Multi-IF |
| Format Following | Correct `/think`/`/no_think` switching, proper `<think>` tokens | ThinkFollow benchmark |
| Preference Alignment | Helpfulness, engagement, style for open-ended queries | Arena-Hard, AlignBench |
| Agent Ability | Tool invocation, multi-turn interaction with environment | BFCL, ToolUse |
| Specialized Scenarios | RAG (minimize hallucination), domain-specific | CounterFactQA |

#### 4.5.3 Reward System

Three reward types:

**Type 1: Rule-Based Reward**

$$R_{\text{rule}}(q, r) = f_{\text{rules}}(q, r) \in [0, 1]$$

- High precision, no ambiguity.
- Applied to: instruction following (format verification), format adherence.
- Prevents reward hacking through deterministic evaluation.

**Type 2: Model-Based Reward with Reference Answer**

$$R_{\text{ref}}(q, r) = g_{\text{Qwen2.5-72B}}(q, r, r^*) \in [0, 1]$$

where $r^*$ is the reference answer. Qwen2.5-72B-Instruct scores the response.
- More flexible than rule-based.
- Avoids false negatives from strict format matching.

**Type 3: Model-Based Reward without Reference Answer**

$$R_{\text{pref}}(q, r) = \text{RM}_\phi(q, r) \in \mathbb{R}$$

where $\text{RM}_\phi$ is a reward model trained on human preference data.
- Broadest coverage.
- Enhances engagement and helpfulness.

**Combined Reward per Task:**

$$R(q, r) = \sum_{j} w_j \cdot R_j(q, r)$$

where $w_j$ are task-specific weighting coefficients.

#### 4.5.4 Agent RL with Environment Interaction

During RL rollout for agent tasks, the model performs **complete multi-turn interaction cycles**:

$$q \xrightarrow{\text{model}} a_1 \xrightarrow{\text{env}} o_1 \xrightarrow{\text{model}} a_2 \xrightarrow{\text{env}} o_2 \cdots \xrightarrow{\text{env}} o_T \xrightarrow{\text{eval}} R$$

where $a_i$ are model actions (tool calls), $o_i$ are environment observations, and $R$ is the final reward.

#### 4.5.5 Stage 4 Training Dynamics (Qwen3-32B Analysis)

From Table 22:

| Metric | After S2 (Think) | After S3 (Think/NoThink) | After S4 (Think/NoThink) |
|---|---|---|---|
| ThinkFollow* | — | 88.7 | 98.9 |
| ToolUse* | 63.3 | 70.4 / 73.2 | 85.5 / 86.5 |
| AIME'24 (Think) | 83.8 | 81.9 | 81.4 |
| LiveCodeBench v5 (Think) | 68.4 | 67.2 | 65.7 |
| CounterFactQA* | 50.4 | 61.3 / 64.3 | 68.1 / 66.4 |

**Observed Trade-off:** Stages 3 and 4 improve general/alignment/agent capabilities but cause slight degradation on hard reasoning tasks (AIME'24: 83.8 → 81.4, LiveCodeBench: 68.4 → 65.7). This is attributed to training on broader task distribution that may compromise specialized capabilities.

**Design Decision:** Accept this trade-off to enhance overall versatility.

### 4.6 Strong-to-Weak Distillation

#### 4.6.1 Scope

Applies to: Qwen3-0.6B, 1.7B, 4B, 8B, 14B (dense) and Qwen3-30B-A3B (MoE).

**Teachers:** Qwen3-32B, Qwen3-235B-A22B.

#### 4.6.2 Phase 1: Off-Policy Distillation

Generate responses from teacher models in both `/think` and `/no_think` modes. Train student via standard SFT:

$$\mathcal{L}_{\text{off-policy}} = -\frac{1}{|\mathcal{D}_{\text{teacher}}|}\sum_{(q,r) \in \mathcal{D}_{\text{teacher}}} \frac{1}{|r|}\sum_{t=1}^{|r|} \log p_{\theta_s}(r_t | q, r_{<t})$$

**Purpose:** Initialize reasoning skills and mode-switching capability.

#### 4.6.3 Phase 2: On-Policy Distillation

Student generates on-policy sequences; fine-tunes by aligning logits with teacher:

$$\mathcal{L}_{\text{on-policy}} = \sum_{t=1}^{|r|} D_{\text{KL}}\left[p_{\theta_T}(\cdot | q, r_{<t}) \;\|\; p_{\theta_S}(\cdot | q, r_{<t})\right]$$

where $r \sim \pi_{\theta_S}(\cdot | q)$ is sampled from the **student**, and $p_{\theta_T}$ provides the target distribution from the **teacher**.

**On-policy sequences** are critical: they ensure the student learns from its own distribution, preventing exposure bias.

#### 4.6.4 Efficiency Analysis

From Table 21 (Qwen3-8B comparison):

| Method | AIME'24 (pass@1 / pass@64) | AIME'25 | MATH500 | LCB v5 | GPU Hours |
|---|---|---|---|---|---|
| Off-policy Distill | 55.0 / 90.0 | 42.8 / 83.3 | 92.4 | 42.0 | — |
| + RL | 67.6 / 90.0 | 55.5 / 83.3 | 94.8 | 52.9 | 17,920 |
| + On-policy Distill | **74.4 / 93.3** | **65.5 / 86.7** | **97.0** | **60.3** | **1,800** |

**Key Findings:**
1. On-policy distillation outperforms RL on all metrics.
2. On-policy distillation requires **~1/10 GPU hours** compared to RL.
3. Distillation improves pass@64 (exploration capability): 90.0 → 93.3 for AIME'24. RL does not improve pass@64.
4. Teacher logits expand the student's exploration space.

**Pseudo-Algorithm: Strong-to-Weak Distillation**
```
PROCEDURE StrongToWeakDistill(student, teacher, prompts):
    // Phase 1: Off-Policy Distillation
    D_off ← ∅
    FOR EACH q IN prompts:
        r_think ← teacher.Generate(q, mode="/think")
        r_no_think ← teacher.Generate(q, mode="/no_think")
        D_off ← D_off ∪ {(q, r_think), (q, r_no_think)}
    TrainSFT(student, D_off)
    
    // Phase 2: On-Policy Distillation
    FOR step IN 1 TO max_steps:
        q_batch ← SamplePrompts(prompts)
        FOR EACH q IN q_batch:
            mode ← RandomChoice(["/think", "/no_think"])
            r_student ← student.Generate(q, mode)
            
            // Compute KL divergence at each token position
            FOR t IN 1 TO |r_student|:
                p_teacher ← teacher.Logits(q, r_student[:t])
                p_student ← student.Logits(q, r_student[:t])
                loss_t ← KL(p_teacher || p_student)
            
            total_loss ← mean(loss_terms)
        total_loss.backward()
        optimizer.step()
    
    RETURN student
END PROCEDURE
```

---

## 5. Optimization Strategy

### 5.1 Optimizer Configuration

**Optimizer:** AdamW with standard hyperparameters.

$$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

where $\hat{m}_t, \hat{v}_t$ are bias-corrected first and second moment estimates, $\lambda$ is weight decay.

### 5.2 Learning Rate Schedule

**Per-stage scheduling:**
- **S1:** Standard cosine decay with warmup.
- **S2:** Accelerated learning rate decay (steeper cosine/linear).
- **S3:** Continued decay for long-context adaptation.

$$\eta_t^{(s)} = \begin{cases} \frac{t}{T_{\text{warmup}}} \cdot \eta_{\max}^{(s)} & t \leq T_{\text{warmup}} \\ \eta_{\min}^{(s)} + \frac{1}{2}(\eta_{\max}^{(s)} - \eta_{\min}^{(s)})(1 + \cos(\frac{\pi \cdot (t - T_{\text{warmup}})}{T_{\text{total}}^{(s)} - T_{\text{warmup}}})) & t > T_{\text{warmup}} \end{cases}$$

For S2, $T_{\text{total}}^{(2)} < T_{\text{total}}^{(1)}$ relative to data size, accelerating decay.

### 5.3 Gradient Clipping and Numerical Stability

**QK-Norm** is the primary numerical stability mechanism:
- Replaces QKV-bias from Qwen2.
- Prevents attention logit divergence at large scales.

**Gradient Clipping:** Applied with maximum gradient norm.

### 5.4 Training Stability for MoE

**Global-batch load balancing** prevents router collapse:
- Without it: router degenerates to selecting the same few experts (dead expert problem).
- With it: all 128 experts maintain non-trivial utilization rates.

**No shared experts:** Removes the crutch that shared experts provide, forcing the router and individual experts to specialize more aggressively.

---

## 6. Inference Path

### 6.1 Inference Modes

**Mode Selection** via chat template flags:

| Flag | Behavior | Output Format |
|---|---|---|
| `/think` (or none) | Full reasoning chain | `<think>...</think>{response}` |
| `/no_think` | Direct response | `<think>\n</think>{response}` |
| Thinking budget $B$ | Partial reasoning | `<think>...{truncated}...stop instruction</think>{response}` |

### 6.2 Thinking Budget Inference Procedure

```
PROCEDURE InferWithBudget(model, query, budget_B):
    prefix ← FormatQuery(query)
    generated ← ""
    in_thinking ← True
    thinking_tokens ← 0
    
    WHILE NOT EndOfSequence:
        token ← model.GenerateNextToken(prefix + generated)
        
        IF in_thinking:
            thinking_tokens += 1
            IF thinking_tokens >= budget_B:
                // Insert stop-thinking instruction
                stop_instr ← "Considering the limited time..."
                generated ← generated + stop_instr + "</think>\n\n"
                in_thinking ← False
                CONTINUE
        
        generated ← generated + token
        IF token == "</think>":
            in_thinking ← False
    
    RETURN ExtractResponse(generated)
END PROCEDURE
```

### 6.3 Context Length at Inference

**Training Context:** 32,768 tokens.

**Inference Extension (4×):** Using YARN + DCA → effective context of 131,072 tokens.

For models with 128K native context (4B–235B), the context is directly supported.

### 6.4 Sampling Configuration

| Parameter | Thinking Mode | Non-Thinking Mode |
|---|---|---|
| Temperature | 0.6 | 0.7 |
| Top-p | 0.95 | 0.8 |
| Top-k | 20 | 20 |
| Presence Penalty | 1.5 (creative tasks) | 1.5 |
| Max Output Length | 32,768 (38,912 for AIME) | 32,768 |

### 6.5 KV Cache Analysis at Inference

For Qwen3-235B-A22B with $n_{kv} = 4$, $L = 94$, $d_h = d / n_q$:

$$\text{KV Cache (FP16)} = 2 \times 94 \times 4 \times d_h \times T \times 2 \;\text{bytes}$$

With GQA ($n_{kv} = 4$ vs $n_q = 64$), KV cache is reduced by $16\times$ compared to MHA.

### 6.6 MoE Inference Considerations

**Memory:** Full 235B parameters must reside in accelerator memory (or be offloaded).

**Compute:** Only 22B parameters activate per token → significant FLOP savings.

**Expert Parallelism:** 128 experts can be distributed across devices with all-to-all communication for token routing.

**Latency Model:**

$$T_{\text{latency}} = T_{\text{attn}} + T_{\text{router}} + T_{\text{alltoall}} + T_{\text{expert\_compute}}$$

where $T_{\text{alltoall}}$ is the dominant bottleneck in distributed MoE inference.

---

## 7. Evaluation Protocol

### 7.1 Pre-Training Evaluation

**15 Benchmarks across 4 categories:**

| Category | Benchmarks | Settings |
|---|---|---|
| General | MMLU (5-shot), MMLU-Pro (5-shot CoT), MMLU-Redux (5-shot), BBH (3-shot CoT), SuperGPQA (5-shot CoT) | |
| Math & STEM | GPQA (5-shot CoT), GSM8K (4-shot CoT), MATH (4-shot CoT) | |
| Coding | EvalPlus (0-shot), MultiPL-E (0-shot, 8 languages), MBPP-3shot, CRUX-O (1-shot) | |
| Multilingual | MGSM (8-shot CoT), MMMLU (5-shot), INCLUDE (5-shot) | |

**Evaluation Invariant:** All models evaluated using the same evaluation pipeline and settings.

### 7.2 Post-Training Evaluation

**Categories and Metrics:**

| Category | Benchmarks | Metric |
|---|---|---|
| General | MMLU-Redux, GPQA-Diamond (avg of 10 samples), C-Eval, LiveBench | Accuracy |
| Alignment | IFEval (strict prompt acc), Arena-Hard, AlignBench v1.1, Creative Writing v3, WritingBench | Score / Win rate |
| Math & Reasoning | MATH-500, AIME'24/25 (avg of 64 samples), ZebraLogic, AutoLogi | Accuracy |
| Agent & Coding | BFCL v3 (FC format), LiveCodeBench v5, CodeForces Elo | Accuracy / Rating |
| Multilingual | Multi-IF, INCLUDE, MMMLU (14 langs), MT-AIME2024 (55 langs), PolyMath (18 langs), MLogiQA (10 langs) | Accuracy |

**AIME Evaluation Protocol:**
- 30 questions per year (Part I + Part II)
- 64 samples per question
- Final score = average accuracy across all samples

**GPQA-Diamond:** 10 samples per query, averaged accuracy.

**LiveCodeBench v5:**
- Non-thinking: Official prompt.
- Thinking: Modified prompt (removed restriction "You will not return anything except for the program").

**CodeForces:** Up to 8 independent reasoning attempts per problem for Elo calculation.

**BFCL v3:** FC format for all Qwen3 models. YARN deployment to 64K context for multi-turn evaluation.

### 7.3 In-House Benchmarks (Stage Analysis)

| Benchmark | Purpose |
|---|---|
| CounterFactQA* | Counterfactual questions; detect hallucination avoidance |
| LengthCtrl* | Creative writing with length requirements |
| ThinkFollow* | Multi-turn with random `/think`/`/no_think` flag switching |
| ToolUse* | Single/multi-turn/multi-step tool calling stability |

### 7.4 Multilingual Evaluation Coverage

From Table 10:

| Benchmark | Languages | Task Type |
|---|---|---|
| Multi-IF | 8 | Instruction Following |
| INCLUDE | 44 | Regional Knowledge |
| MMMLU | 14 | General Knowledge |
| MT-AIME2024 | 55 | Mathematics |
| PolyMath | 18 | Mathematics |
| MLogiQA | 10 | Logical Reasoning |

---

## 8. Deployment Constraints and Systems Analysis

### 8.1 Model Serving Architecture

| Model | Total Params | Active Params | Min GPU Memory (FP16) | Target Deployment |
|---|---|---|---|---|
| 0.6B | 0.6B | 0.6B | ~1.2 GB | Edge/Mobile |
| 1.7B | 1.7B | 1.7B | ~3.4 GB | Edge |
| 4B | 4B | 4B | ~8 GB | Single GPU |
| 8B | 8B | 8B | ~16 GB | Single GPU |
| 14B | 14B | 14B | ~28 GB | Single/Multi GPU |
| 32B | 32B | 32B | ~64 GB | Multi GPU |
| 30B-A3B | 30B | 3B | ~60 GB (full) | Multi GPU (memory-bound) |
| 235B-A22B | 235B | 22B | ~470 GB (full) | Multi-node |

### 8.2 MoE Deployment Challenges

**Memory vs. Compute Asymmetry:**
- 235B parameters must be loaded → ~470 GB in FP16, requiring 6+ H100 80GB GPUs.
- Only 22B parameters activate per token → compute cost is ~1/10 of a 235B dense model.

**Expert Parallelism:** Distribute experts across devices. All-to-all communication for routing.

**Batch Size Impact:** Larger batches amortize the all-to-all communication cost:

$$\text{Efficiency} = \frac{\text{Useful Compute}}{\text{Total Time}} = \frac{B \cdot \text{FLOPs}_{\text{active}}}{B \cdot \text{FLOPs}_{\text{active}} + T_{\text{comm}}}$$

### 8.3 Quantization Compatibility

BBPE tokenizer with 151,669 vocabulary ensures compatibility with standard quantization schemes (GPTQ, AWQ, GGML).

Tie embedding constraint (0.6B, 1.7B) may require special handling: shared weight matrix must maintain precision.

### 8.4 Long-Context Serving

For 128K context models:
- KV cache for Qwen3-235B-A22B with 128K context, $n_{kv}=4$, $L=94$:

$$\text{KV Cache} = 2 \times 94 \times 4 \times d_h \times 131{,}072 \times 2 \;\text{bytes}$$

GQA with $n_{kv}=4$ is critical — without it, KV cache would be $16\times$ larger.

### 8.5 Thinking Budget as Latency Control

The thinking budget mechanism directly maps to latency SLAs:

$$\text{Latency} \propto B_{\text{think}} + T_{\text{response}}$$

Users can set $B_{\text{think}} = 0$ (no-thinking) for minimum latency or increase it for better quality, creating a smooth **latency-quality Pareto frontier**.

---

## 9. Convergence Dynamics and Training Analysis

### 9.1 Pre-Training Convergence

**Three-stage learning rate schedule** with:
- S1: Slow decay over 30T tokens → foundational representations
- S2: Accelerated decay over 5T tokens → reasoning specialization
- S3: Continued decay over hundreds of billions → long-context adaptation

**Scaling Law Predictions** ensure each model trains at its compute-optimal operating point $(\eta^*, B^*)$.

### 9.2 Reasoning RL Convergence (Stage 2)

**Monotonic Improvement:** AIME'24 improves consistently from 70.1 to 85.1 over 170 steps without manual intervention.

**Entropy Dynamics:** Controlled to increase steadily or remain stable:

$$H(\pi_{\theta_t}) \geq H(\pi_{\theta_0}) \quad \forall \; t \in [0, T_{\text{RL}}]$$

This is crucial for:
- Maintaining exploration capability
- Preventing premature mode collapse
- Ensuring stable training without hyperparameter intervention

### 9.3 Thinking Mode Fusion Impact

**Stage 3 Effects:**
- General capabilities improve (CounterFactQA +10.9, LengthCtrl +8.0)
- Mode switching capability established (ThinkFollow: 88.7)
- Hard reasoning slightly degrades (AIME'24: 83.8 → 81.9)

**Stage 4 Effects:**
- Mode switching perfected (ThinkFollow: 88.7 → 98.9)
- Agent capabilities dramatically improve (ToolUse: 70.4 → 85.5)
- Hard reasoning continues slight decline (AIME'24: 81.9 → 81.4)

**Root Cause Analysis:** Training on diverse general tasks introduces a **capability tax** on specialized reasoning. The model's parameter capacity must be shared across more tasks, reducing specialization depth for any single domain.

### 9.4 Distillation vs. RL Convergence

| Property | RL | On-Policy Distillation |
|---|---|---|
| GPU Hours | 17,920 | 1,800 |
| Pass@1 Improvement | Moderate | Superior |
| Pass@64 Improvement | None | Significant |
| Exploration Expansion | No | Yes |
| Training Stability | Requires entropy management | Stable (teacher provides signal) |

**Interpretation:** Teacher logits provide a richer supervisory signal than sparse reward, enabling the student to learn multi-modal response distributions rather than collapsing to high-reward modes.

---

## 10. Compression Analysis

### 10.1 Architectural Compression (MoE)

**Compression Ratio (compute):**

$$\text{CR}_{\text{compute}} = \frac{P_{\text{total}}}{P_{\text{active}}} = \frac{235\text{B}}{22\text{B}} \approx 10.7\times$$

**Information Preservation:** The full parameter set $P_{\text{total}}$ stores specialized knowledge across 128 experts. Per-token routing selects the most relevant 8 experts, preserving task-relevant information while reducing computation.

$$I_{\text{preserved}} = I(\mathbf{x}; \text{MoE}(\mathbf{x})) \approx I(\mathbf{x}; \text{Dense}_{P_{\text{total}}}(\mathbf{x}))$$

Empirically validated: MoE models match dense models with 1/5 activated parameters.

### 10.2 Knowledge Compression (Distillation)

The distillation pipeline compresses the knowledge of a 235B-parameter teacher into student models as small as 0.6B:

$$\text{CR}_{\text{knowledge}} = \frac{P_{\text{teacher}}}{P_{\text{student}}} = \frac{235\text{B}}{0.6\text{B}} \approx 392\times$$

**Information Loss:**

$$\Delta I = D_{\text{KL}}[p_{\theta_T} \| p_{\theta_S}]$$

On-policy distillation minimizes this KL divergence on the student's own distribution, ensuring minimal information loss where it matters most.

### 10.3 Context Compression (BBPE)

$$\text{CR}_{\text{tokenizer}} = \frac{\text{Bytes Input}}{\text{Tokens Output}} \approx 3\text{–}4 \;\text{bytes/token (English)}$$

Multilingual coverage (119 languages) requires larger vocabulary (151,669) to maintain reasonable compression across diverse scripts.

### 10.4 Positional Compression (ABF + YARN + DCA)

Context window expanded 4× at inference without retraining:

$$\frac{T_{\text{inference}}}{T_{\text{train}}} = \frac{131{,}072}{32{,}768} = 4\times$$

**Information Preservation:** YARN's non-uniform frequency scaling preserves relative position sensitivity for short-range dependencies while extending capacity for long-range:

$$\text{PositionResolution}(r) \approx \text{constant} \quad \forall \; r \in [0, T_{\text{inference}}]$$

---

## 11. Summary of Key Results

### 11.1 Flagship Model (Qwen3-235B-A22B)

**Thinking Mode — Selected Peak Results:**

| Benchmark | Score |
|---|---|
| AIME'24 | 85.7 |
| AIME'25 | 81.5 |
| LiveCodeBench v5 | 70.7 |
| CodeForces | 2,056 (98.2nd percentile) |
| BFCL v3 | 70.8 |
| Arena-Hard | 95.6 |
| MT-AIME2024 (55 langs) | 80.8 |

**vs. DeepSeek-R1:** Outperforms on 17/23 benchmarks with 60% activated and 35% total parameters.

**Non-Thinking Mode:** Exceeds DeepSeek-V3, LLaMA-4-Maverick, GPT-4o on 18/23 benchmarks.

### 11.2 Dense Flagship (Qwen3-32B)

- Outperforms QwQ-32B on 17/23 benchmarks (thinking mode).
- Competitive with OpenAI-o3-mini (medium).
- Surpasses Qwen2.5-72B-Instruct (non-thinking mode) with less than half the parameters.

### 11.3 Lightweight Models

Strong-to-Weak Distillation produces models that consistently exceed open-source baselines with equal or larger parameter counts:

- Qwen3-4B (thinking) surpasses DeepSeek-R1-Distill-Qwen-14B on most benchmarks.
- Qwen3-8B (thinking) surpasses DeepSeek-R1-Distill-Qwen-32B on most benchmarks.
- Qwen3-30B-A3B achieves QwQ-32B-level reasoning with <1/10 activated parameters.