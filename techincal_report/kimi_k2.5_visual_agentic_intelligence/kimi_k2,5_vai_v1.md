

# Kimi K2.5: Visual Agentic Intelligence — End-to-End Technical Report

---

## 1. System-Level Architecture Overview

### 1.1 Formal Problem Formulation

**Definition.** Kimi K2.5 is a native multimodal agentic model defined over the joint probability distribution:

$$p_\theta(y \mid x_{\text{text}}, x_{\text{image}}, x_{\text{video}}, x_{\text{tools}}) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t}, x_{\text{text}}, x_{\text{image}}, x_{\text{video}}, x_{\text{tools}})$$

where $\theta$ parameterizes the full system comprising:
- A vision encoder $\mathcal{E}_{\text{vis}}: \mathbb{R}^{H \times W \times C \times F} \to \mathbb{R}^{N_v \times d}$ (MoonViT-3D)
- An MLP projector $\mathcal{P}: \mathbb{R}^{d_{\text{vis}}} \to \mathbb{R}^{d_{\text{llm}}}$
- A Mixture-of-Experts (MoE) language backbone $\mathcal{M}_{\text{MoE}}: \mathbb{R}^{(N_t + N_v) \times d} \to \mathbb{R}^{(N_t + N_v) \times d}$

**Boundary Conditions:**
- Total parameters: $|\theta| = 1.04 \times 10^{12}$ (1.04T)
- Activated parameters per token: $|\theta_{\text{active}}| = 32 \times 10^9$ (32B)
- Expert configuration: 384 total experts, 8 activated per token (sparsity factor = 48)
- Context window: up to 262,144 tokens
- Pre-training token budget: $\sim 15 \times 10^{12}$ mixed vision-text tokens

**Invariants:**
- Joint vision-text optimization must enhance both modalities simultaneously (no modality degradation)
- Weight sharing between image and video encoders must be complete
- Agent orchestration must reduce critical-path latency, not merely increase total work

---

## 2. Data Pipeline

### 2.1 Text Data Curation

**Objective.** Construct a high-quality, deduplicated text corpus spanning Web Text, Code, Mathematics, and Knowledge domains with controlled diversity and epoch constraints.

**Inputs.** Raw web crawls, code repositories (including cross-file structures, issues, code reviews, commit histories), mathematical corpora, knowledge bases, PDF-extracted documents.

**Outputs.** Tokenized, quality-filtered, deduplicated text sequences ready for packing into training batches.

**Processing Stages:**

| Stage | Operation | Details |
|-------|-----------|---------|
| 1 | Correctness validation | Domain-specific quality checks per source |
| 2 | Quality scoring | Rule-based and model-based quality filters |
| 3 | Deduplication | MinHash / exact-match deduplication across shards |
| 4 | Epoch control | Maximum epochs per data source to prevent memorization |
| 5 | Proportion adjustment | Upweighted code-centric data for agentic coding |
| 6 | Tokenization | Byte-level BPE tokenizer consistent with Kimi K2 |

**Enhanced Code Intelligence (K2.5-specific):**
- Repository-level code supporting cross-file reasoning and architectural understanding
- Issues, code reviews, and commit histories capturing real-world development patterns
- Code-related documents retrieved from PDF and webtext corpora

**Invariants:**
- Uniform dataset quality standards across sources
- Deterministic data ordering for reproducible training (seeded shuffling with worker state management)
- No contamination from evaluation benchmarks

**Failure Modes:**
- Distribution drift from aggressive upweighting of any single domain
- Insufficient diversity if epoch limits are too high per source
- Tokenizer mismatch between pre-training and post-training stages

### 2.2 Vision Data Curation

**Objective.** Construct a multimodal corpus spanning 7 categories: caption, interleaving, OCR, knowledge, perception, video, and agent data.

**Inputs.** Raw images, videos, web pages with interleaved image-text, academic PDFs, GUI screenshots, action trajectories.

**Outputs.** Image/video tensors paired with text tokens, packed into multimodal training sequences.

**Category-Level Processing:**

| Category | Source | Processing | Purpose |
|----------|--------|------------|---------|
| Caption | LAION-style datasets [49, 19] | Strict synthetic caption limits to mitigate hallucination | Fundamental modality alignment |
| Interleaving | Books, web pages, tutorials [81, 32] | Multi-image comprehension sequences | Context learning |
| OCR | Multilingual text, dense layouts, multi-page documents | Layout parsing + text extraction | Document understanding |
| Knowledge | Academic materials | Layout parsers → visual reasoning problems | STEM reasoning |
| STEM Problems | Web crawling + retrieval | In-context learning [11] for reformulation into K-12 to university problems | Structured reasoning |
| Image-Code | HTML, React, SVG + rendered screenshots | Paired code-visual geometry alignment | Vision-to-code bridge |
| Agent | GUI screenshots + action trajectories (desktop, mobile, web) | Human-annotated demonstrations | Agentic capability |
| Video | Diverse video sources | Hour-long comprehension + fine-grained spatiotemporal perception | Temporal understanding |
| Grounding | Bounding boxes, point references, contour-level segmentation [51] | Perception annotations | Pixel-level perception |

**Quality Control Pipeline:**

$$\text{PSEUDO-ALGORITHM: VisionDataQC}$$
$$\textbf{Input: } \text{Raw multimodal corpus } \mathcal{D}_{\text{raw}}$$
$$\textbf{Output: } \text{Filtered corpus } \mathcal{D}_{\text{clean}}$$
$$\text{1. For each sample } (x_v, x_t) \in \mathcal{D}_{\text{raw}}:$$
$$\quad \text{2. Apply resolution filter: discard if } \min(H,W) < H_{\min}$$
$$\quad \text{3. Apply aspect ratio filter: discard if } H/W \notin [\gamma_{\min}, \gamma_{\max}]$$
$$\quad \text{4. Apply text quality scorer: discard if } q(x_t) < \tau_q$$
$$\quad \text{5. Apply deduplication (perceptual hashing + text SimHash)}$$
$$\quad \text{6. Apply synthetic caption ratio enforcement: } r_{\text{synth}} \leq r_{\max}$$
$$\text{7. Return } \mathcal{D}_{\text{clean}}$$

**Invariants:**
- Synthetic caption ratio strictly bounded to prevent hallucination amplification
- Stochastic augmentation preserves 2D spatial coordinates and orientation metadata during geometric transformations
- Deterministic training via meticulous random seed and worker state management

**Failure Modes:**
- Hallucination propagation from excessive synthetic captions
- Distribution mismatch between training augmentations and inference conditions
- Missing spatial metadata after geometric transforms

### 2.3 Data Storage and Loading Infrastructure

**Objective.** Bridge data preparation and model training with S3-compatible object storage, retaining visual data in native format.

**Key Properties:**

| Property | Implementation |
|----------|---------------|
| Flexibility | Dynamic shuffling, blending, tokenization, loss masking, sequence packing; adjustable data ratios |
| Augmentation | Stochastic augmentation of visual and textual modalities; spatial coordinate and orientation metadata integrity preserved |
| Determinism | Fully deterministic training via seed management and worker state tracking; seamless resumption after interruptions |
| Scalability | Tiered caching mechanisms; regulated object storage request frequency; robust scaling to large distributed clusters |

**Pseudo-Algorithm: Deterministic Data Loading**

$$\textbf{Input: } \text{Global batch size } B, \text{ world size } W, \text{ step } s, \text{ seed } \sigma$$
$$\textbf{Output: } \text{Batch } \mathcal{B}_s \text{ identical to uninterrupted run}$$
$$\text{1. Compute worker-local seed: } \sigma_w = \text{hash}(\sigma, w, s)$$
$$\text{2. Initialize RNG state from } \sigma_w$$
$$\text{3. Sample data indices via deterministic shuffle over } \mathcal{D}$$
$$\text{4. Apply tokenization, packing, loss masking}$$
$$\text{5. Return } \mathcal{B}_s$$

---

## 3. Model Architecture

### 3.1 MoonViT-3D: Native-Resolution 3D Vision Encoder

**Definition.** MoonViT-3D is a Vision Transformer that natively processes images and videos at their original resolutions within a unified architecture sharing all parameters and a consistent embedding space.

**Initialization.** Continual pre-training from SigLIP-SO-400M [77].

#### 3.1.1 Patch-Level Tokenization (NaViT Packing)

**For images:** Given an image $I \in \mathbb{R}^{H \times W \times 3}$, patches of size $p \times p$ are extracted:

$$N_{\text{patches}} = \left\lfloor \frac{H}{p} \right\rfloor \times \left\lfloor \frac{W}{p} \right\rfloor$$

Patches are flattened and sequentially concatenated into a 1D sequence, enabling variable-resolution inputs without sub-image splitting or splicing.

**For videos (3D extension):** Given a video $V \in \mathbb{R}^{F \times H \times W \times 3}$ with $F$ frames, consecutive frames are grouped in chunks of 4:

$$\text{Number of temporal chunks} = \left\lceil \frac{F}{4} \right\rceil$$

Within each temporal chunk, 2D patches from up to 4 frames are jointly flattened and packed into a single 1D sequence. The identical self-attention mechanism operates across both space and time:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{(4 \cdot N_{\text{patches}}) \times d_k}$ for each temporal chunk.

**Properties:**
- Complete weight sharing between image and video processing paths
- No specialized video modules or architectural bifurcation
- Extra temporal attention within chunks improves understanding of high-speed motions and visual effects

#### 3.1.2 Temporal Compression via Temporal Pooling

**Objective.** Reduce visual token count to extend feasible video length by $4\times$.

Prior to the MLP projector, lightweight temporal pooling aggregates patches within each temporal chunk:

$$\mathbf{z}_{\text{pool}}^{(i)} = \frac{1}{|\mathcal{C}_i|} \sum_{f \in \mathcal{C}_i} \mathbf{z}_f^{(i)}$$

where $\mathcal{C}_i$ is the set of frames in temporal chunk $i$, and $\mathbf{z}_f^{(i)} \in \mathbb{R}^{d_{\text{vis}}}$ is the patch embedding at spatial position $i$ from frame $f$.

**Compression ratio:**

$$\text{Temporal compression factor} = 4\times$$

**Information preservation guarantee:** Averaging within semantically coherent temporal chunks preserves spatiotemporal features while reducing token count. The shared encoder ensures knowledge from image pretraining transfers holistically to videos.

**Complexity Analysis:**

| Component | Time Complexity | Space Complexity |
|-----------|----------------|-----------------|
| Patch extraction | $O(HWF)$ | $O(N_{\text{patches}} \cdot F \cdot d)$ |
| Self-attention per chunk | $O((4N_{\text{patches}})^2 \cdot d)$ | $O((4N_{\text{patches}})^2)$ |
| Temporal pooling | $O(N_{\text{patches}} \cdot 4 \cdot d)$ | $O(N_{\text{patches}} \cdot d)$ |
| Post-pooling tokens | — | $O(N_{\text{patches}} \cdot \lceil F/4 \rceil \cdot d)$ |

#### 3.1.3 ViT Training Stage

**Objective.** Establish robust native-resolution visual encoding aligned with the language backbone.

**Loss function:** Solely cross-entropy caption loss (no contrastive loss unlike Kimi-VL):

$$\mathcal{L}_{\text{caption}} = -\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t}, x_{\text{visual}})$$

**Two-stage alignment:**

| Stage | Trainable Components | Aligned With | Token Budget | Purpose |
|-------|---------------------|--------------|--------------|---------|
| Stage 1 | MoonViT-3D (full) | Moonlight-16B-A3B [34] | $\sim 1$T | High-resolution image + video understanding |
| Stage 2 | MLP projector only | 1T MoE LLM | Short | Smooth bridge for joint pre-training |

### 3.2 MLP Projector

**Definition.** A multi-layer perceptron mapping vision encoder output dimension $d_{\text{vis}}$ to LLM hidden dimension $d_{\text{llm}}$:

$$\mathcal{P}(\mathbf{z}) = W_2 \cdot \text{GELU}(W_1 \cdot \mathbf{z} + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{d_{\text{mid}} \times d_{\text{vis}}}$, $W_2 \in \mathbb{R}^{d_{\text{llm}} \times d_{\text{mid}}}$.

**Tensor flow:**

$$\mathbb{R}^{N_v \times d_{\text{vis}}} \xrightarrow{W_1, b_1} \mathbb{R}^{N_v \times d_{\text{mid}}} \xrightarrow{\text{GELU}} \mathbb{R}^{N_v \times d_{\text{mid}}} \xrightarrow{W_2, b_2} \mathbb{R}^{N_v \times d_{\text{llm}}}$$

### 3.3 Kimi K2 MoE Language Backbone

**Definition.** Trillion-parameter Mixture-of-Experts transformer [59] pre-trained on 15T text tokens.

**Configuration:**

| Parameter | Value |
|-----------|-------|
| Total parameters | 1.04T |
| Activated parameters per token | 32B |
| Total experts | 384 |
| Experts activated per token | 8 |
| Sparsity factor | 48 |
| Optimizer | MuonClip [30, 34] with QK-Clip |
| Context length (base) | 4,096 |
| Context length (extended) | 262,144 |

**MoE routing:** For input token $\mathbf{x} \in \mathbb{R}^d$, the router selects top-$k$ experts:

$$\text{Router}(\mathbf{x}) = \text{TopK}\left(\text{softmax}(\mathbf{W}_r \mathbf{x}), k=8\right)$$

$$\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} g_i \cdot E_i(\mathbf{x})$$

where $g_i$ is the gating weight for expert $E_i$.

### 3.4 Full Multimodal Forward Pass

**Pseudo-Algorithm: Kimi K2.5 Forward Pass**

$$\textbf{Input: } x_{\text{text}} \in \mathbb{R}^{N_t}, x_{\text{visual}} \in \mathbb{R}^{F \times H \times W \times 3}$$
$$\textbf{Output: } y \in \mathbb{R}^{T \times |V|}$$
$$\text{1. } \mathbf{Z}_{\text{vis}} \leftarrow \text{MoonViT-3D}(x_{\text{visual}}) \quad \triangleright \mathbb{R}^{N_{\text{chunks}} \times 4N_p \times d_{\text{vis}}}$$
$$\text{2. } \mathbf{Z}_{\text{pool}} \leftarrow \text{TemporalPool}(\mathbf{Z}_{\text{vis}}) \quad \triangleright \mathbb{R}^{N_{\text{chunks}} \cdot N_p \times d_{\text{vis}}}$$
$$\text{3. } \mathbf{Z}_{\text{proj}} \leftarrow \mathcal{P}(\mathbf{Z}_{\text{pool}}) \quad \triangleright \mathbb{R}^{N_v \times d_{\text{llm}}}$$
$$\text{4. } \mathbf{E}_{\text{text}} \leftarrow \text{TextEmbed}(x_{\text{text}}) \quad \triangleright \mathbb{R}^{N_t \times d_{\text{llm}}}$$
$$\text{5. } \mathbf{H}_0 \leftarrow \text{Concat}(\mathbf{Z}_{\text{proj}}, \mathbf{E}_{\text{text}}) \quad \triangleright \mathbb{R}^{(N_v + N_t) \times d_{\text{llm}}}$$
$$\text{6. For } l = 1 \ldots L:$$
$$\quad \text{7. } \mathbf{H}_l \leftarrow \text{MoETransformerBlock}_l(\mathbf{H}_{l-1})$$
$$\text{8. } y \leftarrow \text{LMHead}(\mathbf{H}_L) \quad \triangleright \mathbb{R}^{T \times |V|}$$

---

## 4. Compression Pipeline

### 4.1 Temporal Visual Token Compression

**Objective.** Reduce visual token count for videos by $4\times$ while preserving spatiotemporal information.

**Compression equation:**

$$N_{\text{tokens}}^{\text{compressed}} = N_p \cdot \left\lceil \frac{F}{4} \right\rceil$$

$$\text{Compression ratio} = \frac{N_p \cdot F}{N_p \cdot \lceil F/4 \rceil} \approx 4$$

**Information preservation analysis:** Given frames $\{I_1, I_2, I_3, I_4\}$ in a temporal chunk, consecutive frames exhibit high temporal correlation. The temporal averaging operation:

$$\mathbf{z}_{\text{pool}}^{(i)} = \frac{1}{4}\sum_{f=1}^{4} \mathbf{z}_f^{(i)}$$

preserves the mean activation while reducing variance by factor $4$. Under the assumption of additive Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ on each frame representation:

$$\text{Var}\left(\mathbf{z}_{\text{pool}}^{(i)}\right) = \frac{\sigma^2}{4}$$

This yields a $\sqrt{4} = 2\times$ improvement in signal-to-noise ratio per spatial position.

**Failure Modes:**
- Information loss for high-frequency temporal events within a 4-frame window
- Averaging may blur rapid scene transitions or fast motion boundaries

### 4.2 Activation Compression for Memory Efficiency

**Objective.** Reduce GPU memory footprint during training.

| Technique | Target | Compression |
|-----------|--------|-------------|
| Selective recomputation | LayerNorm, SwiGLU, MLA up-projections | Eliminate intermediate activation storage |
| FP8-E4M3 compression | Insensitive activations | $2\times$ memory reduction vs FP16 |
| CPU offloading | Remaining activations | GPU → CPU with overlapped streaming |

**FP8-E4M3 encoding:** For activation tensor $\mathbf{A} \in \mathbb{R}^{B \times S \times d}$ in FP16:

$$\mathbf{A}_{\text{FP8}} = \text{Quantize}_{\text{E4M3}}\left(\frac{\mathbf{A}}{\max(|\mathbf{A}|)} \cdot 2^{E_{\max}}\right)$$

where $E_{\max} = 8$ for E4M3 format, providing dynamic range $[2^{-9}, 448]$ with 4 exponent bits and 3 mantissa bits.

**Memory savings per layer:**

$$\Delta M_{\text{layer}} = S \cdot d \cdot (2 - 1) = S \cdot d \text{ bytes} \quad (\text{FP16} \to \text{FP8})$$

### 4.3 Context Compression via Agent Swarm (Context Sharding)

**Definition.** Agent Swarm implements proactive context management through explicit orchestration rather than reactive context truncation.

**Mechanism:** Long-horizon tasks are decomposed into parallel, semantically isolated subtasks:

$$\mathcal{C}_{\text{orchestrator}} = \mathcal{C}_{\text{instruction}} \cup \bigcup_{i=1}^{N_{\text{sub}}} \text{Summary}(\mathcal{C}_{\text{subagent}_i})$$

instead of:

$$\mathcal{C}_{\text{single}} = \mathcal{C}_{\text{instruction}} \cup \bigcup_{t=1}^{T} \mathcal{C}_{\text{step}_t} \quad (\text{grows linearly with } T)$$

**Context compression ratio:**

$$\text{Effective context} = |\mathcal{C}_{\text{orchestrator}}| + \max_i |\mathcal{C}_{\text{subagent}_i}|$$

$$\text{vs. sequential: } |\mathcal{C}_{\text{single}}| = \sum_{t=1}^{T} |\mathcal{C}_{\text{step}_t}|$$

**Advantage over Discard-all:** Preserves task-level coherence at orchestrator level while keeping subagent contexts tightly bounded. No structural information or intermediate reasoning is lost—only summaries are propagated.

---

## 5. Training Stages

### 5.1 Stage Overview

| Stage | Data | Tokens | Seq Length | Trainable Components |
|-------|------|--------|------------|---------------------|
| ViT Training | Alt text, synthesis caption, grounding, OCR, video | $\sim 1$T | 4,096 | ViT only |
| Joint Pre-training | Text + Vision (constant ratio) | $\sim 15$T | 4,096 | ViT & LLM |
| Long-context Mid-training | High-quality text & multimodal, long text, long video, reasoning, long-CoT | 500B → 200B | 32,768 → 262,144 | ViT & LLM |
| Supervised Fine-Tuning | Synthesized high-quality responses | — | — | Full model |
| Reinforcement Learning | Outcome-verifiable tasks + GRM tasks | — | — | Full model |

### 5.2 Stage 1: ViT Training

**Objective.** Establish robust native-resolution visual encoding with cross-entropy caption loss only.

**Loss:**

$$\mathcal{L}_{\text{ViT}} = \mathcal{L}_{\text{caption}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t}, x_{\text{visual}})$$

**Key design decision:** No contrastive loss (departure from Kimi-VL [54]).

**Sub-stages:**
1. **Full ViT update:** Align MoonViT-3D with Moonlight-16B-A3B via caption loss ($\sim 1$T tokens, minimal FLOPs due to small ViT)
2. **MLP projector only:** Bridge ViT to 1T MoE LLM for smoother joint pre-training (very short stage)

**Invariant:** After Stage 1, visual encoder must produce semantically meaningful patch embeddings for both images and videos within a unified representation space.

### 5.3 Stage 2: Joint Pre-training (Native Multimodal Pre-training)

**Objective.** Simultaneously enhance language and multimodal capabilities through early fusion with constant vision-text ratio.

#### 5.3.1 Vision-Text Ratio Analysis

**Key finding (Table 1):** Early fusion with lower vision ratio yields better results given fixed total vision-text token budget. This contradicts conventional wisdom [8, 21] that late-stage, high-ratio vision injection accelerates multimodal capability.

**Ablation results under fixed total budget:**

| Vision Injection Timing | Vision:Text Ratio | Vision Knowledge | Vision Reasoning | OCR | Text Knowledge | Text Reasoning | Code |
|------------------------|-------------------|-----------------|-----------------|-----|---------------|---------------|------|
| Early (0%) | 10:90 | **25.8** | **43.8** | **65.7** | **45.5** | **58.5** | **24.8** |
| Mid (50%) | 20:80 | 25.0 | 40.7 | 64.1 | 43.9 | 58.6 | 24.0 |
| Late (80%) | 50:50 | 24.2 | 39.0 | 61.5 | 43.1 | 57.8 | 24.0 |

**Dip-and-recover phenomenon (Figure 9):** Mid-fusion and late-fusion configurations exhibit temporary text performance degradation when vision data is first introduced, attributed to modality domain shift disrupting established linguistic representations.

**Mathematical justification:** Let $\mathcal{L}(\theta; \mathcal{D}_t, \mathcal{D}_v)$ be the joint loss over text data $\mathcal{D}_t$ and vision data $\mathcal{D}_v$. Under early fusion, the gradient landscape is:

$$\nabla_\theta \mathcal{L} = (1-\alpha)\nabla_\theta \mathcal{L}_t + \alpha \nabla_\theta \mathcal{L}_v$$

where $\alpha = 0.1$ (10:90 ratio). Early co-optimization allows the model to develop unified multimodal representations without representation collapse observed in late fusion, facilitating smoother gradient landscapes for both modalities.

**Training configuration:**
- Continues from near-end Kimi K2 checkpoint
- 15T additional vision-text tokens at 4K sequence length
- Data recipe extends K2's distribution with unique tokens, increased coding content weight, controlled maximum epochs per source

### 5.4 Stage 3: Long-Context Mid-Training

**Objective.** Extend context window and refine capabilities with higher-quality data.

**Context extension via YaRN [44] interpolation:**

$$\text{RoPE}_{\text{YaRN}}(\theta_i, m) = \begin{cases} \theta_i & \text{if } \theta_i > \theta_{\text{high}} \\ s \cdot \theta_i & \text{if } \theta_i < \theta_{\text{low}} \\ (1-\gamma)\frac{\theta_i}{s} + \gamma \theta_i & \text{otherwise} \end{cases}$$

where $s$ is the scaling factor, $\gamma$ is an interpolation coefficient, and $\theta_{\text{low}}, \theta_{\text{high}}$ define frequency boundaries.

**Sequential extension schedule:**

$$4,096 \to 32,768 \to 65,536 \to 131,072 \to 262,144$$

**Token budget:** 500B → 200B across extension stages.

**Outcomes:** Significant generalization improvements in long-context text understanding and long video comprehension (processing $>2000$ frames for benchmarks like LVBench).

### 5.5 Stage 4: Supervised Fine-Tuning (SFT)

#### 5.5.1 Standard SFT

**Objective.** Train model to prioritize interactive reasoning and precise tool-calling for complex real-world applications.

**Data generation:**
- Synthesized high-quality candidate responses from K2, K2 Thinking, and proprietary expert models
- Specialized pipelines per domain integrating human annotation, prompt engineering, and multi-stage verification
- Large-scale instruction-tuning dataset with diverse prompts and intricate reasoning trajectories

**Loss:**

$$\mathcal{L}_{\text{SFT}} = -\frac{1}{|\mathcal{D}_{\text{SFT}}|}\sum_{(x,y) \in \mathcal{D}_{\text{SFT}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid y_{<t}, x)$$

#### 5.5.2 Zero-Vision SFT

**Definition.** A post-training technique using exclusively text SFT data to activate visual and agentic capabilities, without any vision-specific supervision.

**Core insight:** Joint pretraining (Stage 2) already establishes strong vision-text alignment, enabling capabilities to generalize naturally across modalities. Adding human-designed visual trajectories at SFT stage actually hurts generalization.

**Mechanism:**
- All image manipulations are proxied through programmatic operations in IPython
- Serves as a generalization of traditional vision tool-use
- Enables diverse reasoning behaviors including:
  - Pixel-level operations (object size estimation via binarization and counting)
  - Visually grounded tasks (object localization, counting, OCR)

**Pseudo-Algorithm: Zero-Vision SFT Activation**

$$\textbf{Input: } \text{Joint-pretrained model } \theta_{\text{pretrain}}, \text{ text-only SFT data } \mathcal{D}_{\text{text}}$$
$$\textbf{Output: } \text{Model with activated visual + agentic capabilities } \theta_{\text{zvSFT}}$$
$$\text{1. } \theta \leftarrow \theta_{\text{pretrain}}$$
$$\text{2. For each } (x_{\text{text}}, y_{\text{text}}) \in \mathcal{D}_{\text{text}}:$$
$$\quad \text{3. } y_{\text{text}} \text{ includes IPython tool-use traces (no visual inputs)}$$
$$\quad \text{4. } \mathcal{L} = -\sum_t \log p_\theta(y_t \mid y_{<t}, x_{\text{text}})$$
$$\quad \text{5. } \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$$
$$\text{6. Post-condition: } \theta \text{ generalizes to visual inputs via cross-modal transfer}$$

**Empirical validation:** Compared to text-vision SFT, zero-vision SFT yields substantially better performance on visual agentic tasks, likely because:
- Joint pretraining establishes cross-modal alignment
- Text-vision SFT introduces low-quality vision data that limits generalization
- Text-only SFT preserves the natural modality generalization from pretraining

**Failure modes of zero-vision SFT (addressed by subsequent RL):**
- Visual inputs sometimes ignored
- Images may not be attended to when necessary

### 5.6 Stage 5: Reinforcement Learning

#### 5.6.1 Policy Optimization

**Objective.** Optimize model policy $\pi_\theta$ via token-level clipped reinforcement learning.

For each problem $x \sim \mathcal{D}$, $K$ responses $\{y_1, \ldots, y_K\}$ are sampled from $\pi_{\text{old}}$. The optimization objective:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|} \mathbf{1}\left[\alpha \leq \log\frac{\pi_\theta(y_j^i \mid y_{j,<i}, x)}{\pi_{\text{old}}(y_j^i \mid y_{j,<i}, x)} \leq \beta\right] \cdot \frac{r(x, y_j) - \bar{r}(x)}{\tau} \cdot \log \pi_\theta(y_j^i \mid y_{j,<i}, x)$$

where:
- $\alpha, \beta, \tau > 0$ are hyperparameters
- $N = \sum_{j=1}^{K} |y_j|$ is the total generated tokens
- $\bar{r}(x) = \frac{1}{K}\sum_{j=1}^{K} r(x, y_j)$ is the mean reward
- $y_{j,0:i}$ denotes the prefix up to the $i$-th token of the $j$-th response

**Token-level clipping mechanism:** Gradient masking scheme where:

$$\text{Gradient for token } (j, i) = \begin{cases} \nabla_\theta \log \pi_\theta(y_j^i \mid \cdot) \cdot A_{j,i} & \text{if } \alpha \leq \log\frac{\pi_\theta}{\pi_{\text{old}}} \leq \beta \\ 0 & \text{otherwise} \end{cases}$$

**Key distinction from PPO [50]:** Relies strictly on the log-ratio to explicitly bound off-policy drift, regardless of the sign of advantages. This is critical for maintaining training stability in long-horizon, multi-step tool-use reasoning.

**Optimizer:** MuonClip [30, 34].

#### 5.6.2 Reward Function Design

**Multi-component reward system:**

| Reward Type | Application | Mechanism |
|-------------|-------------|-----------|
| Rule-based outcome | Reasoning, agentic tasks | Verifiable solution matching |
| Budget-control | Token efficiency | Penalizes excessive token usage |
| GRM | General-purpose tasks | Generative Reward Model evaluation |
| Visual task-specific | Vision tasks | Task-dependent metric computation |

**Visual task-specific rewards:**

**Visual Grounding (IoU-based F1):**

$$r_{\text{grounding}} = F1(\text{IoU-soft-match})$$

where soft matches are derived from Intersection over Union between predicted and ground-truth bounding boxes.

**Point Localization (Gaussian-weighted):**

$$r_{\text{point}} = F1(\text{Gaussian-distance-soft-match})$$

with optimal matching between predicted and ground-truth points.

**Polygon Segmentation:**

$$r_{\text{polygon}} = \text{IoU}\left(\text{Rasterize}(\hat{P}), M_{\text{gt}}\right)$$

where $\hat{P}$ is the predicted polygon rasterized into a binary mask, and $M_{\text{gt}}$ is the ground-truth mask.

**OCR:**

$$r_{\text{OCR}} = 1 - \frac{\text{EditDist}(\hat{y}, y^*)}{\max(|\hat{y}|, |y^*|)}$$

**Counting:**

$$r_{\text{count}} = f(|n_{\text{pred}} - n_{\text{gt}}|)$$

where $f$ is a decreasing function of absolute error.

**Complex visual puzzles:** LLM verifier (Kimi K2) provides feedback.

#### 5.6.3 Generative Reward Models (GRMs)

**Definition.** Fine-grained evaluators aligned with Kimi's internal value criteria, applied across diverse environments (chat assistants, coding agents, search agents, artifact-generating agents).

**Evaluation dimensions:**
- Helpfulness
- Response readiness
- Contextual relevance
- Appropriate level of detail
- Aesthetic quality of generated artifacts
- Strict instruction following

**Anti-reward-hacking measures:** Multiple alternative GRM rubrics tailored to different task contexts to mitigate overfitting to a single preference signal.

#### 5.6.4 Outcome-Based Visual RL

**Objective.** Refine model to reliably incorporate visual inputs into reasoning after zero-vision SFT.

**Task categories:**
1. **Visual grounding and counting:** Accurate localization and enumeration of objects within images
2. **Chart and document understanding:** Interpretation of structured visual information and text extraction
3. **Vision-critical STEM problems:** Mathematical and scientific questions filtered to require visual inputs

**Cross-modal transfer result (Table 2):**

| Benchmark | Before Vision-RL | After Vision-RL | Improvement |
|-----------|------------------|-----------------|-------------|
| MMLU-Pro | 84.7% | 86.4% | +1.7% |
| GPQA-Diamond | 84.3% | 86.4% | +2.1% |
| LongBench v2 | 56.7% | 58.9% | +2.2% |

**Analysis:** Visual RL enhances calibration in areas requiring structured information extraction, reducing uncertainty on queries resembling visually grounded reasoning (counting, OCR). Bidirectional enhancement: text bootstraps vision, vision refines text.

**Self-improving data pipeline:** Extracting successful RL trajectories for rejection-sampling fine-tuning (RFT), enabling subsequent joint RL stages to leverage richer multimodal reasoning traces.

#### 5.6.5 Joint Multimodal RL

**Definition.** RL paradigm organized by abilities (knowledge, reasoning, coding, agentic) rather than input modality. Domain experts jointly learn from both pure-text and multimodal queries.

**Design principle:** Capability improvements acquired through either textual or visual inputs generalize to enhance related abilities across the alternate modality, maximizing cross-modal capability transfer.

**GRM integration:** Optimizes across heterogeneous traces without modality barriers.

#### 5.6.6 Token-Efficient RL (Toggle)

**Objective.** Reconcile reasoning capabilities with computational efficiency, preventing length-overfitting while maintaining test-time scaling ability.

**Problem:** Models trained under rigid budget constraints exhibit length-overfitting—they cannot effectively leverage additional inference-time tokens for complex problems, defaulting to truncated reasoning patterns.

**Toggle reward function:** For learning iteration $t$:

$$r_{\text{Toggle}}^{(t)}(x, y) = \begin{cases} r_{\text{perf}}(x, y) + r_{\text{budget}}(x, y) & \text{if Phase0 and } \bar{a}(x) > \lambda \\ r_{\text{perf}}(x, y) & \text{if Phase1} \end{cases}$$

where $\lambda$ is the accuracy threshold and phases alternate every $m$ iterations.

**Phase definitions:**
- **Phase 0 (budget-limited):** Model trained to solve within task-dependent token budget. Budget constraint conditionally applied only when mean accuracy exceeds threshold $\lambda$
- **Phase 1 (standard scaling):** Model generates responses up to maximum token limit, encouraging full inference-time scaling

**Problem-dependent budget estimation:**

$$B(x) = \text{Percentile}_\rho\left(\{|y| : y \in \mathcal{Y}_{\text{correct}}(x)\}\right)$$

where $\mathcal{Y}_{\text{correct}}(x)$ is the set of correct responses for problem $x$, and $\rho$ is the percentile parameter. Budget estimated once at training start and fixed thereafter.

**Toggle as optimization:** Stochastic alternating optimization for a bi-objective problem reconciling:
- $\min_\theta \mathbb{E}[\text{error}(x, y)]$ (performance)
- $\min_\theta \mathbb{E}[|y|]$ (efficiency)

**Empirical results (Figure 5):**
- Average 25–30% output token reduction with negligible performance impact
- Redundant CoT patterns (repeated verifications, mechanical calculations) decrease substantially
- Strong domain generalization: training on math + programming → consistent token reductions on GPQA and MMLU-Pro

---

## 6. Agent Swarm Architecture

### 6.1 Problem Formulation

**Definition.** Agent Swarm is a self-directed parallel agent orchestration framework that dynamically decomposes complex tasks into heterogeneous sub-problems and executes them concurrently.

**Sequential execution bottleneck:** Existing agent systems scale linearly in inference time with task complexity. Even systems capable of hundreds of reasoning steps suffer from:
- Linear scaling of latency
- Exhaustion of practical reasoning depth
- Tool-call budget limits
- Inability to handle broad information gathering + multi-branch reasoning

### 6.2 PARL: Parallel-Agent Reinforcement Learning

**Architecture:** Decoupled design comprising:
- **Trainable orchestrator:** Updated via RL
- **Frozen subagents:** Instantiated from fixed intermediate policy checkpoints

**Design rationale for decoupling:**
1. **Credit assignment ambiguity:** Outcome-based rewards are sparse and noisy; correct final answer does not guarantee flawless subagent execution
2. **Training instability:** End-to-end co-optimization across multiple agents amplifies gradient variance

By freezing subagents and treating their outputs as environmental observations rather than differentiable decision points, high-level coordination logic is disentangled from low-level execution proficiency.

**Efficiency strategy:** Train orchestrator using small-size subagents first, then transition to larger models. RL framework supports dynamically adjusting inference instance ratios between subagents and orchestrator to maximize resource utilization.

### 6.3 PARL Reward Formulation

$$r_{\text{PARL}}(x, y) = r_{\text{perf}}(x, y) + \lambda_1 \cdot r_{\text{parallel}}(x, y) + \lambda_2 \cdot r_{\text{finish}}(x, y)$$

**Components:**

| Reward | Purpose | Addresses |
|--------|---------|-----------|
| $r_{\text{perf}}(x, y)$ | Overall solution quality | Primary objective |
| $r_{\text{parallel}}(x, y)$ | Incentivizes subagent instantiation | Serial collapse (local optimum defaulting to single-agent) |
| $r_{\text{finish}}(x, y)$ | Rewards successful subtask completion | Spurious parallelism (reward-hacking via empty subagent spawning) |

**Annealing schedule:** $\lambda_1, \lambda_2 \to 0$ over training to ensure final policy optimizes primary objective.

**Failure modes addressed:**
- **Serial collapse:** Orchestrator defaults to single-agent execution, a local optimum. Mitigated by $r_{\text{parallel}}$.
- **Spurious parallelism:** Orchestrator spawns many subagents without meaningful task decomposition to inflate parallel metrics. Mitigated by $r_{\text{finish}}$ enforcing feasibility.

### 6.4 Critical Steps Metric

**Definition.** Computational time cost metric defined by analogy to the critical path in a computation graph.

Model an episode as a sequence of execution stages $t = 1, \ldots, T$. In each stage, the main agent executes an action (direct tool invocation or subagent instantiation). Let:
- $S_{\text{main}}^{(t)}$: steps by main agent in stage $t$ (typically $S_{\text{main}}^{(t)} = 1$)
- $S_{\text{sub},i}^{(t)}$: steps by $i$-th subagent in parallel group of stage $t$

**Total critical steps:**

$$C = \sum_{t=1}^{T} \left(S_{\text{main}}^{(t)} + \max_i S_{\text{sub},i}^{(t)}\right)$$

**Properties:**
- Constraining by critical steps (not total steps) explicitly incentivizes effective parallelization
- Excessive subtask creation without reducing max execution time yields no benefit
- Well-balanced task decomposition that shortens the longest parallel branch directly reduces $C$
- Orchestrator learns to allocate work to minimize end-to-end latency, not merely maximize concurrency

### 6.5 Prompt Construction for Capability Induction

**Objective.** Shape task distribution such that parallel decomposition is naturally favored without explicitly instructing parallelization.

**Synthetic prompt categories:**

| Category | Description | Sequential Bottleneck |
|----------|-------------|----------------------|
| Wide search | Simultaneous exploration of many independent information sources | Exceeds tool-call budget sequentially |
| Deep search | Multiple reasoning branches with delayed aggregation | Exceeds reasoning-step budget |
| Real-world workloads | Long-context document analysis, large-scale file downloading | Infeasible within sequential limits |

### 6.6 Agent Swarm Results

**Performance gains (Table 6):**

| Benchmark | Agent Swarm | Single K2.5 | Δ |
|-----------|------------|-------------|---|
| BrowseComp | 78.4% | 60.6% | +17.8% |
| WideSearch (Item-F1) | 79.0% | 72.7% | +6.3% |
| In-house Swarm Bench | 58.3% | 41.6% | +16.7% |

**Execution time savings (Figure 8):**
- WideSearch: $3\times$ to $4.5\times$ faster than single-agent baseline
- As target Item-F1 increases from 30% to 70%, single-agent execution time grows from $1.8\times$ to $>7.0\times$ baseline; Agent Swarm maintains $0.6\times$ to $1.6\times$
- Near-constant low latency regardless of task complexity

---

## 7. Training Infrastructure

### 7.1 Hardware Configuration

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA H800 |
| Interconnect | $8 \times 400$ Gbps RoCE per node |
| Parallelism | 16-way PP (virtual stages) + 16-way EP + ZeRO-1 DP |
| Minimum unit | 32 nodes (any multiple thereof) |

### 7.2 Parallelism Strategy

| Parallelism Dimension | Configuration | Scope |
|-----------------------|--------------|-------|
| Pipeline Parallelism (PP) | 16-way with virtual stages [27, 40] | Layer distribution |
| Expert Parallelism (EP) | 16-way [33] | Expert distribution |
| Data Parallelism | ZeRO-1 | Optimizer state sharding |
| Scheduling | Interleaved 1F1B | EP all-to-all overlapped with computation |

### 7.3 Memory Optimization

| Technique | Target | Effect |
|-----------|--------|--------|
| Selective recomputation | LayerNorm, SwiGLU, MLA up-projections | Eliminate activation storage |
| FP8-E4M3 compression | Insensitive activations | $2\times$ memory reduction |
| CPU offloading | Remaining activations | Overlapped streaming to CPU |

### 7.4 Decoupled Encoder Process (DEP)

**Problem.** In standard PP, vision encoder and text embedding are co-located in Stage-0, causing:
- Drastic fluctuations in computational load and memory from variable multimodal input sizes
- Load imbalance from PP
- Cannot reuse highly optimized text-only parallel strategies

**Solution: DEP (three phases per training step):**

**Pseudo-Algorithm: Decoupled Encoder Process**

$$\textbf{Phase 1: Balanced Vision Forward}$$
$$\text{1. Replicate MoonViT-3D on ALL GPUs (small model)}$$
$$\text{2. Distribute forward workload evenly across GPUs based on load metrics}$$
$$\quad \text{(image count, patch count)}$$
$$\text{3. Execute forward pass for all visual data in global batch}$$
$$\text{4. Discard ALL intermediate activations, retain only final output activations}$$
$$\text{5. Gather results to PP Stage-0}$$

$$\textbf{Phase 2: Backbone Training}$$
$$\text{6. Execute forward + backward for main MoE transformer}$$
$$\text{7. Fully leverages any efficient parallel strategy validated in text-only training}$$
$$\text{8. After backward: gradients accumulated at visual encoder output}$$

$$\textbf{Phase 3: Vision Recomputation \& Backward}$$
$$\text{9. Re-compute vision encoder forward pass (recomputation)}$$
$$\text{10. Execute backward pass for vision encoder parameters}$$

**Properties:**
- Load-balanced vision forward across all GPUs regardless of PP assignment
- Decoupled optimization strategy for vision encoder vs main backbone
- K2.5 seamlessly inherits K2's parallel strategy
- Multimodal training efficiency: **90% relative to text-only training**

### 7.5 Unified Agentic RL Environment

**Architecture (Figure 10):**

| Component | Function |
|-----------|----------|
| Core Agent Loop | Act → Obs → Act cycle |
| Env Pool | Managed environment instances with sandboxes |
| Rollout Manager | Orchestrates up to 100,000 concurrent agent tasks |
| LLM Gateway | Proxy service for black-box environments under custom protocol |
| Pluggable Components | Toolset, Judge, Prompt Enhancement modules |

**Design principles:**
- Gym-like interface [10] for standardized environment implementation
- Each agent task runs as an independent asynchronous coroutine
- Tasks can recursively trigger sub-task rollouts (supports PARL and Agent-as-Judge)
- Token-in-Token-out paradigm with log probability recording
- Train-inference mismatch correction via logged log-probabilities

**Monitoring:** Performance monitoring, profiling, data visualization, and data verification tools for correctness assurance of the highly-parallel asynchronous system.

---

## 8. Optimization Strategy

### 8.1 Optimizer: MuonClip

**Base optimizer:** Muon [30] with QK-Clip for training stability, as specified in Kimi K2 [53].

**QK-Clip mechanism:** Prevents attention logit explosion by clipping query-key dot products:

$$\text{QK-Clip}: \quad \mathbf{A}_{ij} = \min\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}, \; \gamma\right)$$

where $\gamma$ is the clip threshold.

### 8.2 Token-Level Clipping for Off-Policy Stability

The clipping mechanism bounds the log-ratio:

$$\rho_j^i = \log\frac{\pi_\theta(y_j^i \mid y_{j,<i}, x)}{\pi_{\text{old}}(y_j^i \mid y_{j,<i}, x)}$$

Gradient masking: $\nabla_\theta = 0$ when $\rho_j^i \notin [\alpha, \beta]$.

**Distinction from PPO clipping [50]:** PPO clips the probability ratio $\frac{\pi_\theta}{\pi_{\text{old}}}$ conditioned on advantage sign. Kimi K2.5's method clips the log-ratio unconditionally, explicitly bounding off-policy drift regardless of advantage sign. This aligns with recent stability strategies [74, 78] and is empirically essential for long-horizon, multi-step tool-use reasoning.

### 8.3 Convergence Dynamics

**Training stability conditions:**
- Token-level clipping prevents catastrophic policy drift from train-inference framework discrepancies
- MuonClip with QK-Clip prevents attention logit explosion
- Selective recomputation + FP8 maintains numerical stability
- IcePop-style MoE gradient stabilization [78]

**Convergence indicators (Figure 4):**
- PARL training accuracy increases smoothly as training progresses
- Level of parallelism gradually increases during training
- Vision RL curves (Figure 2) show monotonic improvement across MMMU-Pro, MathVision, CharXiv(RQ), OCRBench as RL FLOPs scale

---

## 9. Inference Path

### 9.1 Standard Inference Configuration

| Parameter | Value |
|-----------|-------|
| Temperature | 1.0 |
| Top-$p$ | 0.95 |
| Context length | 256K tokens |
| Max completion (reasoning benchmarks) | 96K tokens |
| Max completion (vision benchmarks) | 64K tokens |

### 9.2 Agent Inference

**Single agent path:**
- Model equipped with web search, code interpreter (Python/IPython), web browsing tools
- Context management: none (failures counted when exceeding context window), or:
  - Hide-Tool-Result: retain only most recent tool messages when exceeding threshold
  - Discard-all: truncate all history beyond threshold

**Agent Swarm path:**
- Orchestrator equipped with additional tools: `create_subagent` and `assign_task`
- Subagents instantiated dynamically with custom system prompts
- Subagents use search + browser tools independently
- Only task-relevant outputs routed back to orchestrator

**Step limits:**

| Benchmark | Orchestrator Steps | Subagent Steps |
|-----------|--------------------|----------------|
| BrowseComp | 15 | 100 |
| WideSearch | 100 | 100 |
| In-house Bench | 100 | 50 |

### 9.3 Inference-Time Scaling

**Toggle-trained models** balance efficiency and quality:
- Phase 0 training constrains token usage → shorter, more efficient responses
- Phase 1 training preserves full inference-time scaling → complex problems can use more tokens
- 25–30% average token reduction with negligible performance degradation

### 9.4 Video Inference Sampling

| Video Type | Frames Sampled | Spatial Resolution |
|------------|---------------|-------------------|
| Short (VideoMMMU, MMVU, MotionBench) | 128 uniform | 896 |
| Long (Video-MME, LongVideoBench, LVBench) | 2,048 uniform | 448 |

After MoonViT-3D processing with $4\times$ temporal compression, a 2,048-frame video yields:

$$N_{\text{visual tokens}} = N_p \cdot \frac{2048}{4} = 512 \cdot N_p$$

where $N_p$ depends on spatial resolution.

---

## 10. Evaluation Protocol

### 10.1 Benchmark Taxonomy

**Capability axes and benchmarks:**

| Axis | Benchmarks |
|------|-----------|
| Reasoning & General | HLE, AIME 2025, HMMT 2025, IMO-AnswerBench, GPQA-Diamond, MMLU-Pro, SimpleQA Verified, AdvancedIF, LongBench v2 |
| Coding | SWE-Bench Verified/Pro/Multilingual, TerminalBench 2.0, PaperBench, CyberGym, SciCode, OJBench, LiveCodeBench v6 |
| Agentic | BrowseComp, WideSearch, DeepSearchQA, FinSearchComp, Seal-0, GDPVal |
| Image Understanding | MMMU-Pro, MMMU, CharXiv, MathVision, MathVista, SimpleVQA, WorldVQA, ZeroBench, BabyVision, BLINK, MMVP, OCRBench, OmniDocBench 1.5, InfoVQA |
| Video Understanding | VideoMMMU, MMVU, MotionBench, Video-MME, LongVideoBench, LVBench |
| Computer Use | OSWorld-Verified, WebArena |

### 10.2 Variance Reduction Protocols

| Benchmark Category | Averaging | Rationale |
|--------------------|-----------|-----------|
| AIME 2025, HMMT 2025 | Avg@64 | Stochastic reasoning path variance |
| GPQA-Diamond | Avg@8 | Reasoning variance |
| Image/Video benchmarks | Avg@3 | Visual processing variance |
| Coding tasks | Avg@5 | Environment + test case ordering |
| Seal-0, WideSearch | Avg@4 | Search engine result stochasticity |

### 10.3 Specialized Metrics

| Benchmark | Metric | Formula |
|-----------|--------|---------|
| Visual Grounding | F1 with IoU soft-match | $F1 = 2\frac{P \cdot R}{P + R}$ over IoU-thresholded matches |
| OCR | Normalized edit distance | $1 - \frac{\text{EditDist}(\hat{y}, y^*))}{\max(|\hat{y}|, |y^*|)}$ |
| OmniDocBench 1.5 | $(1 - \text{NLD}) \times 100$ | NLD = normalized Levenshtein distance |
| WideSearch | Item-F1 | F1 over retrieved items |
| Agent Swarm | Critical steps | $C = \sum_t (S_{\text{main}}^{(t)} + \max_i S_{\text{sub},i}^{(t)})$ |

### 10.4 Baseline Configuration

| Model | Configuration |
|-------|--------------|
| Claude Opus 4.5 | Extended thinking mode |
| GPT-5.2 | xhigh reasoning effort |
| Gemini 3 Pro | High thinking level |
| DeepSeek-V3.2 | Thinking mode (text-only benchmarks) |
| Qwen3-VL-235B-A22B | Thinking mode (vision benchmarks) |

**Note:** GPT-5.2-xhigh exhibited $\sim 10\%$ failure rate on vision evaluations (no output despite 3 retries); failures treated as incorrect predictions.

---

## 11. Key Results Summary

### 11.1 SOTA and Near-SOTA Results

| Domain | Benchmark | K2.5 Score | Best Competitor | Δ vs Best |
|--------|-----------|-----------|----------------|-----------|
| Reasoning | AIME 2025 | 96.1% | GPT-5.2: 100% | -3.9% |
| Reasoning | HMMT 2025 | 95.4% | GPT-5.2: 99.4% | -4.0% |
| Knowledge | HLE-Full w/ tools | **50.2%** | Gemini 3 Pro: 45.8% | **+4.4%** |
| Coding | LiveCodeBench v6 | **85.0%** | Gemini 3 Pro: 87.4% | -2.4% |
| Agentic | BrowseComp | **60.6%** | GPT-5.2: 65.8% | -5.2% |
| Agentic | BrowseComp (Swarm) | **78.4%** | GPT-5.2 Pro: 77.9% | **+0.5%** |
| Agentic | DeepSearchQA | **77.1%** | Claude: 76.1% | **+1.0%** |
| Agentic | WideSearch (Swarm) | **79.0%** | Claude: 76.2% | **+2.8%** |
| Image | MMMU-Pro | 78.5% | Gemini 3 Pro: 81.0% | -2.5% |
| Image | MathVista (mini) | **90.1%** | Gemini 3 Pro: 89.8% | **+0.3%** |
| Image | SimpleVQA | **71.2%** | Claude: 69.7% | **+1.5%** |
| Image | OCRBench | **92.3%** | Gemini 3 Pro: 90.3% | **+2.0%** |
| Video | LVBench | **75.9%** (SOTA) | Gemini 3 Pro: 73.5% | **+2.4%** |
| Video | LongVideoBench | **79.8%** (SOTA) | Gemini 3 Pro: 77.7% | **+2.1%** |
| Video | MotionBench | **70.4%** | Gemini 3 Pro: 70.3% | **+0.1%** |
| Computer Use | OSWorld-Verified | **63.3%** | Claude: 66.3% | -3.0% |

### 11.2 Token Efficiency Analysis (Table 5)

| Benchmark | K2.5 Score (Tokens) | K2 Thinking Score (Tokens) | Token Reduction |
|-----------|-------------------|--------------------------|----------------|
| AIME 2025 | 96.1% (25k) | 94.5% (30k) | -17% tokens, +1.6% accuracy |
| HMMT 2025 | 95.4% (27k) | 89.4% (35k) | -23% tokens, +6.0% accuracy |
| LiveCodeBench | 85.0% (18k) | 82.6% (25k) | -28% tokens, +2.4% accuracy |

---

## 12. Failure Mode Analysis

### 12.1 Pre-training Failure Modes

| Failure Mode | Mechanism | Mitigation |
|--------------|-----------|------------|
| Dip-and-recover in text performance | Modality domain shift from sudden vision token introduction in mid/late fusion | Early fusion with constant ratio |
| Hallucination from synthetic captions | Over-reliance on generated captions propagates errors | Strict synthetic caption ratio limits |
| Distribution collapse from code upweighting | Excessive code data disrupts general capability | Epoch limits + proportion controls |

### 12.2 Post-training Failure Modes

| Failure Mode | Mechanism | Mitigation |
|--------------|-----------|------------|
| Visual input ignorance (post zero-vision SFT) | Model fails to attend to images when necessary | Outcome-based visual RL |
| Off-policy divergence | Train-inference framework discrepancies amplify over long trajectories | Token-level log-ratio clipping |
| Length-overfitting | Budget-constrained training prevents leveraging additional compute | Toggle (alternating budget/scaling phases) |
| Reward hacking (GRMs) | Model exploits single reward signal | Multiple alternative GRM rubrics per task context |

### 12.3 Agent Swarm Failure Modes

| Failure Mode | Definition | Mitigation |
|--------------|------------|------------|
| Serial collapse | Orchestrator defaults to single-agent execution | $r_{\text{parallel}}$ incentive (annealed) |
| Spurious parallelism | Many subagents spawned without meaningful decomposition | $r_{\text{finish}}$ enforcing subtask completion |
| Credit assignment ambiguity | Correct outcome ≠ all subagents correct; failure ≠ all subagents wrong | Frozen subagents (outputs as observations, not differentiable) |
| Training instability | End-to-end co-optimization across agents amplifies gradient variance | Decoupled architecture (trainable orchestrator + frozen subagents) |

---

## 13. Complexity Analysis

### 13.1 Computational Complexity

| Component | Forward Pass | Backward Pass | Memory |
|-----------|-------------|---------------|--------|
| MoonViT-3D (per chunk) | $O(16N_p^2 \cdot d)$ | $O(16N_p^2 \cdot d)$ | $O(16N_p^2)$ (attention maps) |
| Temporal pooling | $O(4 N_p d)$ | $O(4 N_p d)$ | $O(N_p d)$ |
| MLP projector | $O(N_v \cdot d_{\text{vis}} \cdot d_{\text{llm}})$ | $O(N_v \cdot d_{\text{vis}} \cdot d_{\text{llm}})$ | $O(N_v \cdot d_{\text{mid}})$ |
| MoE attention (per layer) | $O((N_v+N_t)^2 d)$ | $O((N_v+N_t)^2 d)$ | $O((N_v+N_t)^2)$ |
| MoE FFN (per layer) | $O(8 \cdot (N_v+N_t) \cdot d \cdot d_{\text{ffn}})$ | $O(8 \cdot (N_v+N_t) \cdot d \cdot d_{\text{ffn}})$ | $O(8 \cdot (N_v+N_t) \cdot d_{\text{ffn}})$ |
| Router | $O(384 \cdot (N_v+N_t) \cdot d)$ | $O(384 \cdot (N_v+N_t) \cdot d)$ | $O(384 \cdot d)$ |

### 13.2 Agent Swarm Complexity

**Sequential agent:**

$$T_{\text{seq}} = \sum_{t=1}^{T_{\text{total}}} \tau_t$$

where $\tau_t$ is the wall-clock time for step $t$. This scales linearly: $T_{\text{seq}} = O(T_{\text{total}})$.

**Agent Swarm:**

$$T_{\text{swarm}} = \sum_{t=1}^{T_{\text{orchestrator}}} \left(\tau_{\text{main}}^{(t)} + \max_i \tau_{\text{sub},i}^{(t)}\right)$$

Under ideal parallelization where work is evenly distributed across $k$ subagents:

$$T_{\text{swarm}} \approx \frac{T_{\text{seq}}}{k} + T_{\text{overhead}}$$

**Empirical speedup:** $3\times$ to $4.5\times$ on WideSearch benchmark.

---

## 14. Deployment Constraints

### 14.1 Serving Topology

| Constraint | Specification |
|------------|--------------|
| Model size | 1.04T total, 32B activated |
| Minimum GPU cluster | 32 nodes (16 PP × 16 EP × ZeRO-1 DP) |
| Interconnect requirement | $\geq 400$ Gbps per GPU for EP all-to-all |
| Context window | 262K tokens |
| Max reasoning tokens | 96K (reasoning), 64K (vision) |

### 14.2 Memory Budget

| Component | Estimated Memory |
|-----------|-----------------|
| Model weights (FP16) | $\sim 2.08$ TB |
| Model weights (FP8) | $\sim 1.04$ TB |
| KV cache (262K context, per activated params) | Depends on MLA configuration |
| Optimizer states (MuonClip, ZeRO-1 sharded) | $\sim 4\times$ activated parameter size per shard |
| Activations (with recomputation + FP8 + offload) | Bounded by selective strategy |

### 14.3 Agent Swarm Deployment

| Parameter | Constraint |
|-----------|-----------|
| Max concurrent coroutines | 100,000 per rollout manager |
| Subagent instantiation | Dynamic, from frozen intermediate checkpoints |
| Context isolation | Independent working memories per subagent |
| Result propagation | Selective routing (summaries only, not full traces) |
| Instance ratio adjustment | Dynamic between subagents and orchestrator |

### 14.4 DEP Deployment Overhead

- Vision encoder replicated on all GPUs (small model → negligible memory)
- Multimodal training efficiency: **90% of text-only throughput**
- No modification to highly-optimized text-only parallel strategies required

### 14.5 Data Infrastructure Deployment

| Feature | Implementation |
|---------|---------------|
| Storage | S3-compatible object storage |
| Data format | Native visual format (no pre-conversion) |
| Caching | Tiered caching mechanisms |
| Resumption | Fully deterministic—interrupted training produces identical data sequence after resume |
| Scaling | Regulated request frequency, robust to cluster size changes |
| Quality governance | Unified platform for registration, visualization, statistics, cross-cloud sync, lifecycle |

---

## 15. End-to-End Training-to-Inference Flow

$$\boxed{\text{ViT Training (1T tokens)}} \to \boxed{\text{Joint Pre-training (15T tokens, 4K ctx)}} \to \boxed{\text{Long-context Mid-training (700B tokens, →262K ctx)}}$$

$$\to \boxed{\text{Zero-Vision SFT (text-only)}} \to \boxed{\text{Visual RL (outcome-based)}} \to \boxed{\text{Joint Multimodal RL (ability-organized)}}$$

$$\to \boxed{\text{Toggle Token-Efficient RL}} \to \boxed{\text{PARL (Agent Swarm training)}}$$

$$\to \boxed{\text{Serving: Standard Inference | Agent Mode | Agent Swarm Mode}}$$

**Stage-to-stage invariants:**
- Each stage's output satisfies the next stage's input requirements
- Cross-modal alignment established in pre-training is preserved through all post-training stages
- Vision capabilities activated by zero-vision SFT are refined (not degraded) by visual RL
- Visual RL enhances (not degrades) text performance
- Toggle preserves scaling ability while improving efficiency
- PARL training produces orchestrator policies that generalize across task types

**Information flow guarantees:**
- Complete weight sharing between image and video throughout all stages
- No modality-specific parameter bifurcation
- GRM signals applied without modality barriers
- Deterministic data loading ensures reproducibility across all stages