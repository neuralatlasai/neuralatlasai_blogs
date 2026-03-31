# Kimi K2.5: Visual Agentic Intelligence — End-to-End Technical Report

---

## 1. Formal Problem Definition

### 1.1 Objective

Given a pretrained trillion-parameter Mixture-of-Experts (MoE) language backbone $\pi_{\text{K2}}$, construct a native multimodal agentic model $\pi_{\text{K2.5}}$ that:

1. Jointly optimizes text and vision modalities across pre-training, supervised fine-tuning, and reinforcement learning such that each modality enhances the other without degradation.
2. Supports parallel multi-agent orchestration (Agent Swarm) for complex, heterogeneous agentic tasks with provable latency reduction.
3. Achieves state-of-the-art performance across reasoning, coding, vision, video, agentic search, and computer-use benchmarks.

### 1.2 Formal Notation and Boundary Conditions

- **Token space**: $\mathcal{V}_{\text{text}}$ (text vocabulary), $\mathcal{V}_{\text{vis}}$ (visual patch tokens after projection).
- **Input**: Mixed sequences $x = (x_1, \ldots, x_L)$, where each $x_i \in \mathcal{V}_{\text{text}} \cup \mathcal{V}_{\text{vis}}$.
- **Model**: $\pi_\theta(y \mid x)$, autoregressive next-token predictor, $\theta$ parameterizing MoE transformer + vision encoder + projector.
- **Vision encoder**: $f_{\text{ViT}}: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{N_p \times d_v}$, native-resolution, patch-packed.
- **MLP projector**: $g_{\text{proj}}: \mathbb{R}^{d_v} \rightarrow \mathbb{R}^{d_{\text{LLM}}}$.
- **MoE backbone**: $h_{\text{MoE}}: \mathbb{R}^{L \times d_{\text{LLM}}} \rightarrow \mathbb{R}^{L \times |\mathcal{V}_{\text{text}}|}$.

**Invariants:**
- Total pre-training token budget fixed at $\sim$15T tokens (text + vision combined).
- Vision-to-text token ratio is constant throughout joint pre-training (early fusion).
- All image and video encoders share weights identically.
- Context window: 4096 → 262144 tokens (stage-dependent).

**Assumptions:**
- Text SFT data is abundant, high-quality, and diverse; vision SFT data is scarce and low-diversity.
- Outcome-based rewards are verifiable for reasoning, coding, and agentic tasks.
- Subagent execution is independent and parallelizable; orchestrator-subagent credit assignment is decoupled.

---

## 2. Data Pipeline

### 2.1 Text Data Curation

**Domains:** Web text, code, mathematics, knowledge.

**Processing:**
- Deduplication (exact and near-duplicate removal via MinHash/LSH).
- Quality filtering: heuristic + classifier-based scoring.
- Correctness validation: per-domain (mathematical proof checking, code compilation, knowledge fact-checking).
- Enhanced code intelligence: repository-level code, issues, code reviews, commit histories, code-related PDFs and web documents upweighted.
- Maximum epoch control per data source to prevent memorization.

### 2.2 Vision Data Curation

**Seven categories:** Caption, interleaving, OCR, knowledge, perception, video, agent.

| Category | Content | Purpose |
|---|---|---|
| Caption | Alt text, synthetic captions (limited ratio to suppress hallucination) | Modality alignment |
| Interleaving | Books, web pages, tutorials (image-text interleaved) | Multi-image comprehension, long-context |
| OCR | Multilingual text, dense layouts, multi-page documents | Document understanding |
| Knowledge | Academic materials parsed via layout parsers | Visual reasoning |
| STEM Problems | Web-crawled + in-context reformulated K-12 to university problems | Visual mathematical/scientific reasoning |
| Image-Code | HTML, React, SVG paired with rendered screenshots | Code–visual geometry alignment |
| Agent | GUI screenshots + action trajectories (desktop, mobile, web) | Agentic tool use |
| Video | Diverse sources, hour-long + fine-grained spatiotemporal | Temporal understanding |
| Grounding | Bounding boxes, point references, contour-level segmentation | Pixel-level perception |

**Quality control:** Rigorous filtering, deduplication, quality scoring across all categories.

### 2.3 Data Loading Infrastructure

**Design principles:**
- **Flexibility**: Dynamic shuffling, blending, tokenization, loss masking, sequence packing; adjustable ratios at runtime.
- **Augmentation**: Stochastic augmentation of visual and textual modalities; 2D spatial coordinate and orientation metadata preserved during geometric transforms.
- **Determinism**: Fully deterministic via seed and worker-state management; any interruption resumes identically.
- **Scalability**: Tiered caching; scales to large distributed clusters; request frequency to S3-compatible object storage regulated.
- **Storage**: Native visual format retained; no pre-tokenization of images/video.

**Pseudo-Algorithm: Data Loading**

```
PROCEDURE DataLoader(global_batch_size, vision_ratio, text_ratio, seed):
    Initialize RNG with seed
    FOR each training step:
        Sample text_batch from text_corpus with proportion text_ratio
        Sample vision_batch from vision_corpus with proportion vision_ratio
        FOR each sample in vision_batch:
            Load image/video in native format from S3 cache
            Apply stochastic augmentation preserving spatial metadata
            Tokenize via patch packing (NaViT)
        FOR each sample in text_batch:
            Tokenize via BPE
        Pack mixed sequences via sequence packing to target length
        Apply loss masking (mask vision encoder tokens, unmask LLM output tokens)
        Yield packed_batch
```

---

## 3. Vision Encoder: MoonViT-3D

### 3.1 Architecture

**Initialization:** Continual pre-training from SigLIP-SO-400M.

**Core design:**
- Native-resolution ViT: no fixed input resolution; images processed at original aspect ratio and resolution.
- NaViT patch-packing: single images divided into $P \times P$ patches, flattened, and sequentially concatenated into 1D sequences.
- Shared encoder for images and videos: identical weights, identical attention mechanism.

**Image path:**

Given an image $I \in \mathbb{R}^{H \times W \times 3}$:

$$
z_{\text{img}} = f_{\text{ViT}}(I) \in \mathbb{R}^{N_p \times d_v}, \quad N_p = \left\lfloor \frac{H}{P} \right\rfloor \times \left\lfloor \frac{W}{P} \right\rfloor
$$

**Video path (3D extension):**

Given a video segment of $F$ frames, group consecutive frames into chunks of 4:

$$
\{I_1, I_2, I_3, I_4\}, \{I_5, I_6, I_7, I_8\}, \ldots
$$

For each chunk $c$ of 4 frames:
1. Each frame is independently patchified: $z^{(f)} = f_{\text{ViT}}(I_f) \in \mathbb{R}^{N_p \times d_v}$, $f \in \{1,2,3,4\}$.
2. 2D patches from 4 frames are jointly flattened and packed into a single 1D sequence (spatiotemporal volume).
3. The shared attention mechanism operates over both spatial and temporal patch positions.

**Temporal compression (before MLP projector):**

Lightweight temporal pooling aggregates patches within each 4-frame chunk:

$$
\bar{z}_c = \frac{1}{4} \sum_{f=1}^{4} z^{(f)} \in \mathbb{R}^{N_p \times d_v}
$$

This yields $4\times$ temporal compression, extending feasible video length by $4\times$ within the same context window.

### 3.2 ViT Training Stage

**Objective:** Cross-entropy caption loss only (no contrastive loss):

$$
\mathcal{L}_{\text{caption}} = -\sum_{t=1}^{T} \log p_\theta(w_t \mid w_{<t}, f_{\text{ViT}}(I))
$$

where $w_t$ are caption tokens.

**Targets:** Alt texts, synthetic captions (image + video), grounding bounding boxes, OCR texts.

**Two-stage alignment:**
1. **Stage 1:** Update MoonViT-3D to align with Moonlight-16B-A3B via $\mathcal{L}_{\text{caption}}$; $\sim$1T tokens; minimal FLOPs. Enables high-resolution image and video understanding.
2. **Stage 2:** Update only MLP projector $g_{\text{proj}}$ to bridge ViT with 1T LLM backbone. Very short.

**Sequence length:** 4096.

### 3.3 Compression Equations

**Spatial compression (patch embedding):**

$$
\text{CR}_{\text{spatial}} = \frac{H \times W \times 3}{N_p \times d_v} = \frac{H \times W \times 3}{\left\lfloor \frac{H}{P} \right\rfloor \times \left\lfloor \frac{W}{P} \right\rfloor \times d_v}
$$

**Temporal compression (4-frame pooling):**

$$
\text{CR}_{\text{temporal}} = 4
$$

**Total video compression:**

$$
\text{CR}_{\text{video}} = \text{CR}_{\text{spatial}} \times \text{CR}_{\text{temporal}}
$$

**Information preservation invariant:**
- Spatial: patch-level features retain full local receptive field; no downsampling within patches.
- Temporal: mean pooling preserves first-order statistics; temporal attention within each 4-frame chunk captures higher-order dynamics before averaging.
- Weight sharing between image and video paths ensures that image pretraining knowledge transfers holistically to video.

**Failure modes in compression:**
- Mean temporal pooling can lose high-frequency motion information (fast transitions, rapid occlusion changes).
- Mitigated by the extra temporal attention within each spatiotemporal volume prior to pooling.

---

## 4. Model Architecture

### 4.1 Overall Structure

Three components:

$$
\pi_{\text{K2.5}}(y \mid x) = h_{\text{MoE}}\!\left( \text{Concat}\!\left[ g_{\text{proj}}(f_{\text{ViT}}(I_1)), \ldots, g_{\text{proj}}(f_{\text{ViT}}(I_k)), \text{Embed}_{\text{text}}(x_{\text{text}}) \right] \right)
$$

| Component | Parameters | Design |
|---|---|---|
| MoonViT-3D | SigLIP-SO-400M init | Native resolution, NaViT packing, shared image/video |
| MLP Projector | $d_v \rightarrow d_{\text{LLM}}$ | Bridges ViT to LLM embedding space |
| Kimi K2 MoE LLM | 1.04T total, 32B activated | 384 experts, 8 active/token, sparsity 48 |

### 4.2 MoE Backbone Specifics

- Total parameters: $|\theta| = 1.04 \times 10^{12}$.
- Activated parameters per token: $|\theta_{\text{active}}| = 32 \times 10^{9}$.
- Expert count: $E = 384$, top-$k = 8$ activated per token.
- Sparsity: $384 / 8 = 48$.
- Optimizer: MuonClip with QK-Clip for training stability.

### 4.3 Tensor Flow and Memory

**Forward pass (per token):**

1. **Vision branch** (if visual token):

$$
z_v = g_{\text{proj}}(f_{\text{ViT}}(I)) \in \mathbb{R}^{d_{\text{LLM}}}
$$

2. **Text branch** (if text token):

$$
z_t = \text{Embed}_{\text{text}}(x) \in \mathbb{R}^{d_{\text{LLM}}}
$$

3. **Unified sequence** $Z = [z_1, z_2, \ldots, z_L] \in \mathbb{R}^{L \times d_{\text{LLM}}}$.
4. **MoE transformer layers**: For each layer $\ell$:
   - Self-attention (MLA variant):

   $$
   Z^{(\ell)} = \text{MLA}(Z^{(\ell-1)}) + Z^{(\ell-1)}
   $$

   - MoE FFN:

   $$
   Z^{(\ell)} = \sum_{e \in \text{TopK}(Z^{(\ell)})} g_e(Z^{(\ell)}) \cdot \text{FFN}_e(Z^{(\ell)}) + Z^{(\ell)}
   $$

   where $g_e$ is the gating weight for expert $e$.

**Memory considerations:**
- Activation checkpointing: selective recomputation for LayerNorm, SwiGLU, MLA up-projections.
- FP8-E4M3 compression for insensitive activations.
- CPU offload with overlapped streaming for remaining activations.
- KV cache at inference: MLA compresses KV state.

---

## 5. Pre-Training Pipeline

### 5.1 Stage Overview

| Stage | Data | Seq Length | Tokens | Trainable |
|---|---|---|---|---|
| ViT Training | Alt text, synthetic caption, grounding, OCR, video-text | 4096 | ~1T | ViT only → Projector only |
| Joint Pre-Training | Vision + Text + Knowledge + Interleaving + Video + OS | 4096 | ~15T | ViT + LLM |
| Long-Context Mid-Training | Long text, long video, reasoning, long-CoT | 32768 → 262144 | 500B → 200B | ViT + LLM |

### 5.2 Joint Pre-Training: Early Fusion with Constant Vision Ratio

**Key finding (ablation):**

| Vision Injection Timing | Vision:Text Ratio | Vision Knowledge | Vision Reasoning | OCR | Text Knowledge | Text Reasoning | Code |
|---|---|---|---|---|---|---|---|
| Early (0%) | 10:90 | **25.8** | **43.8** | **65.7** | **45.5** | **58.5** | **24.8** |
| Mid (50%) | 20:80 | 25.0 | 40.7 | 64.1 | 43.9 | 58.6 | 24.0 |
| Late (80%) | 50:50 | 24.2 | 39.0 | 61.5 | 43.1 | 57.8 | 24.0 |

**Design decision:** Early fusion with $\sim$10% vision ratio, constant throughout all 15T tokens.

**Rationale:** Under fixed total token budget, early fusion:
- Avoids "dip-and-recover" phenomenon where late vision injection causes transient text degradation (modality domain shift).
- Enables co-optimization from the outset, yielding unified multimodal representations.
- Smooths gradient landscapes for both modalities.

**Objective function during joint pre-training:**

$$
\mathcal{L}_{\text{joint}} = -\frac{1}{N} \sum_{i=1}^{N} \log \pi_\theta(x_i \mid x_{<i})
$$

where $x$ is a mixed text-vision sequence. Loss is computed only on output tokens (vision encoder input tokens are masked from loss).

**Data recipe extension over K2:**
- Unique tokens introduced.
- Increased weight on coding-related content.
- Maximum epochs per data source controlled.

### 5.3 Long-Context Mid-Training

**Context extension via YaRN interpolation:**

$$
\text{RoPE}'(\theta_i, m) = \text{RoPE}\!\left(\frac{\theta_i}{s(r_i)}, m\right)
$$

where $s(r_i)$ is the frequency-dependent scaling function, $r_i = \theta_i / \theta_{\max}$, and YaRN provides smooth interpolation from $L_{\text{base}} = 4096$ to $L_{\text{target}} = 262144$.

**Data:** High-quality long text, long video, reasoning traces, long-CoT.

**Staged extension:** $4096 \rightarrow 32768 \rightarrow 262144$.

---

## 6. Post-Training

### 6.1 Supervised Fine-Tuning (SFT)

**Data synthesis:**
- Candidate responses from K2, K2 Thinking, and proprietary expert models.
- Domain-specialized pipelines with human annotation, prompt engineering, and multi-stage verification.
- Large-scale instruction-tuning dataset: diverse prompts, intricate reasoning trajectories.

### 6.2 Zero-Vision SFT

**Definition:** Use only text SFT data to activate visual and agentic capabilities.

**Mechanism:**
- All image manipulations proxied through programmatic operations in IPython.
- Generalizes traditional vision tool-use to arbitrary pixel-level operations (object size estimation via binarization and counting, object localization, counting, OCR).
- No human-designed visual trajectories used; these were found to hurt generalization.

**Why it works:**
- Joint pre-training (Section 5.2) already establishes strong vision-text alignment.
- Capabilities generalize naturally across modalities without explicit visual SFT data.
- Text-vision SFT yields worse performance on visual agentic tasks due to lack of high-quality vision data diversity.

**Failure modes of alternative (text-vision SFT):**
- Restricted visual reasoning to simple diagrams and primitive tool manipulations (crop, rotate, flip).
- Limited diversity in chain-of-thought data.

### 6.3 Reinforcement Learning

#### 6.3.1 Policy Optimization

For each problem $x \sim \mathcal{D}$, generate $K$ responses $\{y_1, \ldots, y_K\}$ from $\pi_{\text{old}}$. Optimize $\pi_\theta$ via:

$$
\mathcal{L}(\theta) = -\frac{1}{N} \sum_{j=1}^{K} \sum_{i=1}^{|y_j|} \mathbb{1}\!\left[\alpha \leq \log \frac{\pi_\theta(y_{j,i} \mid y_{j,<i}, x)}{\pi_{\text{old}}(y_{j,i} \mid y_{j,<i}, x)} \leq \beta \right] \cdot \frac{r(x, y_j) - \bar{r}(x)}{\tau} \cdot \log \pi_\theta(y_{j,i} \mid y_{j,<i}, x)
$$

where:
- $\alpha, \beta, \tau > 0$ are hyperparameters.
- $N = \sum_{j=1}^{K} |y_j|$ is total generated tokens in the batch.
- $\bar{r}(x) = \frac{1}{K} \sum_{j=1}^{K} r(x, y_j)$ is the mean reward.
- The indicator function $\mathbb{1}[\alpha \leq \log \text{ratio} \leq \beta]$ implements token-level log-ratio clipping.

**Key distinctions from PPO:**
- Clipping operates on the log-ratio directly, regardless of advantage sign.
- This explicitly bounds off-policy drift caused by discrepancies between training and inference frameworks.
- Functions as gradient masking: gradients zeroed for tokens outside $[\alpha, \beta]$.

**Optimizer:** MuonClip.

#### 6.3.2 Reward Function Design

**Taxonomy of rewards:**

| Domain | Reward Type | Formulation |
|---|---|---|
| Reasoning, agentic (verifiable) | Rule-based outcome reward | Binary or continuous correctness signal |
| Token efficiency | Budget-control reward | Penalty for exceeding token budget |
| General-purpose | Generative Reward Models (GRMs) | Granular multi-criteria evaluation |
| Visual grounding | F1 with soft matching | $r = F_1(\text{IoU-based soft matches})$ |
| Point localization | Gaussian-weighted distance F1 | $r = F_1(\text{Gaussian-weighted dist under optimal matching})$ |
| Polygon segmentation | Segmentation IoU | $r = \text{IoU}(\text{rasterize}(\hat{P}), M_{\text{gt}})$ |
| OCR | Normalized edit distance | $r = 1 - \frac{d_{\text{edit}}(\hat{y}, y^*)}{\max(|\hat{y}|, |y^*|)}$ |
| Counting | Absolute difference | $r = f(|\hat{n} - n^*|)$ |
| Visual puzzles | LLM verifier (K2) | K2 provides feedback as reward |

**Generative Reward Models (GRMs):**
- Self-critique rubric reward for open-ended generation.
- Applied across chat assistants, coding agents, search agents, artifact-generating agents.
- Evaluate: helpfulness, response readiness, contextual relevance, detail level, aesthetic quality, instruction following.
- Multiple alternative GRM rubrics per task context to mitigate reward hacking.

#### 6.3.3 Outcome-Based Visual RL

**Three visual domains:**
1. Visual grounding and counting.
2. Chart and document understanding.
3. Vision-critical STEM problems (filtered to require visual inputs).

**Cross-modal transfer result (Vision RL improves text):**

| Benchmark | Before Vision-RL | After Vision-RL | Improvement |
|---|---|---|---|
| MMLU-Pro | 84.7% | 86.4% | +1.7 |
| GPQA-Diamond | 84.3% | 86.4% | +2.1 |
| LongBench v2 | 56.7% | 58.9% | +2.2 |

**Mechanism:** Visual RL enhances calibration in structured information extraction, reducing uncertainty on queries resembling visually grounded reasoning. No degradation of language capabilities observed.

#### 6.3.4 Joint Multimodal RL

**Design:**
- RL domains organized by **ability** (knowledge, reasoning, coding, agentic), not by input modality.
- Domain experts jointly learn from both pure-text and multimodal queries.
- GRM optimizes across heterogeneous traces without modality barriers.
- Cross-modal capability transfer: improvements from either modality generalize to the other.

**Self-improving data pipeline:**
- Outcome-based RL trajectories → rejection-sampling fine-tuning (RFT) → richer multimodal reasoning traces for subsequent RL stages.

#### 6.3.5 Token-Efficient RL: Toggle

**Problem:** Models trained under rigid budget constraints exhibit length-overfitting: they cannot leverage additional inference-time tokens for complex problems.

**Toggle algorithm:** Alternates between inference-time scaling and budget-constrained optimization.

For learning iteration $t$:

$$
r_{\text{Toggle}}(x, y, t) =
\begin{cases}
r_{\text{perf}}(x, y) + \lambda_{\text{budget}} \cdot r_{\text{budget}}(x, y), & \text{if } t \bmod 2m < m \text{ and } \bar{a}(x) > \lambda \\
r_{\text{perf}}(x, y), & \text{otherwise}
\end{cases}
$$

where:
- $\lambda$ is the accuracy threshold for applying budget constraint.
- $m$ is the alternation period.
- $K$ is the number of rollouts per problem.

**Phase 0 (budget-limited):** Train to solve within task-dependent token budget, conditionally applied only when mean accuracy $\bar{a}(x) > \lambda$.

**Phase 1 (standard scaling):** Generate up to maximum token limit, encouraging full inference-time scaling.

**Problem-dependent budget estimation:**

$$
B(x) = \text{Percentile}_\rho\!\left(\{|y_j| : y_j \text{ is correct}\}\right)
$$

Estimated once at training start, fixed thereafter.

**Empirical results on K2 Thinking:**
- 25–30% output token reduction across nearly all benchmarks.
- Negligible performance impact.
- Redundant patterns (repeated verifications, mechanical calculations) decrease substantially.
- Strong domain generalization: trained on math + programming, generalizes to GPQA and MMLU-Pro.

---

## 7. Agent Swarm: Parallel Agent Orchestration

### 7.1 Architecture

**Decoupled design:**
- **Orchestrator** $\pi_{\text{orch}}$: trainable; equipped with tools for sub-agent creation (`create_subagent`) and task delegation (`assign_task`).
- **Subagents** $\{\pi_{\text{sub}}^{(i)}\}$: frozen; instantiated from fixed intermediate policy checkpoints.
- Subagent outputs treated as environmental observations, not differentiable decision points.

**Rationale for decoupling:**
1. **Credit assignment ambiguity:** Outcome-based rewards are sparse; correct final answer does not guarantee flawless subagent execution.
2. **Training instability:** End-to-end co-optimization of orchestrator + subagents is unstable under sparse, non-stationary feedback.

### 7.2 PARL Reward

$$
r_{\text{PARL}}(x, y) = r_{\text{perf}}(x, y) + \lambda_1 \cdot r_{\text{parallel}}(x, y) + \lambda_2 \cdot r_{\text{finish}}(x, y)
$$

| Component | Purpose | Failure Mode Addressed |
|---|---|---|
| $r_{\text{perf}}$ | Overall solution quality | Primary objective |
| $r_{\text{parallel}}$ | Incentivizes subagent instantiation | Serial collapse (defaults to single-agent) |
| $r_{\text{finish}}$ | Rewards successful subtask completion | Spurious parallelism (spawning many subagents without meaningful decomposition) |

**Annealing:** $\lambda_1, \lambda_2 \rightarrow 0$ over training to ensure final policy optimizes for $r_{\text{perf}}$.

### 7.3 Critical Steps (Latency Metric)

Model an episode as $T$ execution stages. In stage $t$:
- Main agent executes $S_{\text{main}}^{(t)}$ steps (typically 1).
- Parallel subagent group: subagent $i$ executes $S_{\text{sub},i}^{(t)}$ steps.
- Stage duration governed by longest-running subagent.

$$
C = \sum_{t=1}^{T} \left( S_{\text{main}}^{(t)} + \max_i S_{\text{sub},i}^{(t)} \right)
$$

**Incentive structure:** Constraining by critical steps rather than total steps:
- Excessive subtask creation without reducing max execution time yields no benefit.
- Well-balanced decomposition directly reduces $C$.
- Orchestrator learns to minimize end-to-end latency, not merely maximize concurrency.

### 7.4 Prompt Construction for Capability Induction

**Synthetic prompt design:**
- **Wide search**: Simultaneous exploration of many independent information sources.
- **Deep search**: Multiple reasoning branches with delayed aggregation.
- **Real-world workloads**: Long-context document analysis, large-scale file downloading.
- Tasks are difficult to complete within fixed reasoning-step and tool-call budgets when executed sequentially.
- Prompts do not explicitly instruct parallelization; task distribution naturally favors parallel decomposition.

### 7.5 Training Procedure

**Pseudo-Algorithm: PARL Training**

```
PROCEDURE PARL_Training(D, π_orch, {π_sub}, max_iters):
    Initialize π_orch from SFT checkpoint
    Freeze all π_sub
    FOR iter = 1 TO max_iters:
        Sample batch {x_1, ..., x_B} from D
        FOR each x in batch (async):
            Initialize orchestrator context C_orch = x
            WHILE not terminated:
                action = π_orch(C_orch)  // tool call or subagent creation
                IF action is create_subagent:
                    Instantiate frozen π_sub with given system prompt
                IF action is assign_task:
                    Dispatch subtask to created subagent (async, independent context)
                    Execute subagent to completion (frozen, not optimized)
                    Return subagent output as observation to C_orch
                IF action is direct_tool_call:
                    Execute tool, return observation
                Update C_orch with observation
            Compute r_PARL(x, y) = r_perf + λ_1 * r_parallel + λ_2 * r_finish
        Compute policy gradient for π_orch using token-level clipped objective
        Update π_orch via MuonClip
        Anneal λ_1, λ_2 toward 0
```

### 7.6 Agent Swarm as Proactive Context Management

**Mechanism:** Context sharding rather than context truncation.

- Each subagent maintains independent working memory and local context.
- Subagents perform local reasoning without contaminating orchestrator's global context.
- Only task-relevant outputs selectively routed back to orchestrator.
- Scales effective context length along an additional architectural dimension.
- Preserves modularity, information locality, and reasoning integrity.

**Comparison with reactive strategies:**

| Strategy | Type | Mechanism | Information Loss |
|---|---|---|---|
| Hide-Tool-Result | Reactive | Retain only recent tool messages | Structural info loss |
| Summary | Reactive | Compress accumulated history | Reasoning trace compression |
| Discard-all | Reactive | Truncate all history beyond threshold | Full history loss |
| Agent Swarm | Proactive | Context sharding via subagent isolation | Minimal (selective routing) |

### 7.7 Agent Swarm Results

| Benchmark | K2.5 Agent Swarm | K2.5 Single Agent | Claude Opus 4.5 | GPT-5.2 | GPT-5.2 Pro |
|---|---|---|---|---|---|
| BrowseComp | **78.4** | 60.6 | 37.0 | 65.8 | 77.9 |
| WideSearch (Item-F1) | **79.0** | 72.7 | 76.2 | — | — |
| In-house Swarm Bench | **58.3** | 41.6 | 45.8 | — | — |

**Latency reduction:** 3×–4.5× on WideSearch. As target Item-F1 increases from 30% to 70%, single-agent execution time grows from ~1.8× to >7.0× baseline, while Agent Swarm maintains 0.6×–1.6×.

---

## 8. Training Infrastructure

### 8.1 Hardware

- NVIDIA H800 GPU clusters.
- 8 × 400 Gbps RoCE interconnects per node.

### 8.2 Parallelism Strategy

| Dimension | Configuration |
|---|---|
| Pipeline Parallelism (PP) | 16-way with virtual stages |
| Expert Parallelism (EP) | 16-way |
| Data Parallelism | ZeRO-1 |
| Minimum allocation | Multiple of 32 nodes |

EP all-to-all communication overlapped with computation under interleaved 1F1B scheduling.

### 8.3 Memory Optimization

| Technique | Target |
|---|---|
| Selective recomputation | LayerNorm, SwiGLU, MLA up-projections |
| FP8-E4M3 compression | Insensitive activations |
| CPU offload | Remaining activations, overlapped streaming |

### 8.4 Decoupled Encoder Process (DEP)

**Problem:** In standard PP, vision encoder and text embedding co-located in Stage-0. Variable multimodal input sizes cause drastic computational load and memory fluctuations in Stage-0. Manual layer adjustment in Stage-0 is a compromise that does not fundamentally resolve load imbalance and precludes reuse of text-only parallel strategies.

**Solution: DEP exploits the topological position of the vision encoder** (start of forward, end of backward).

**Three phases per training step:**

**Phase 1: Balanced Vision Forward**
- Forward pass for all visual data in global batch.
- Vision encoder replicated on all GPUs (small model).
- Forward workload evenly distributed across all GPUs based on load metrics (image/patch counts).
- All intermediate activations discarded; only final output activations retained.
- Results gathered back to PP Stage-0.

**Phase 2: Backbone Training**
- Standard forward + backward for main transformer backbone.
- Fully leverages text-only parallel strategies (no vision encoder overhead).
- Gradients accumulated at visual encoder output.

**Phase 3: Vision Recomputation & Backward**
- Recompute vision encoder forward pass (activation checkpointing).
- Backward pass to compute gradients for vision encoder parameters.

**Pseudo-Algorithm: DEP**

```
PROCEDURE DEP_TrainStep(global_batch, π_ViT, g_proj, h_MoE):
    // Phase 1: Balanced Vision Forward
    visual_data = extract_visual_inputs(global_batch)
    Distribute visual_data evenly across all GPUs by patch count
    FOR each GPU in parallel:
        z_vis = π_ViT(local_visual_data)  // forward only
        Discard all intermediate activations
        Retain only z_vis (final output)
    AllGather z_vis to PP Stage-0 GPUs

    // Phase 2: Backbone Training
    mixed_embeddings = Concat[g_proj(z_vis), Embed_text(text_data)]
    loss = h_MoE.forward_backward(mixed_embeddings)
    // Gradients accumulated at g_proj output (= ViT output boundary)

    // Phase 3: Vision Recomputation & Backward
    z_vis_recomputed = π_ViT(local_visual_data)  // recompute forward
    backward(z_vis_recomputed, accumulated_gradients)
    Update π_ViT parameters
```

**Performance:** Multimodal training efficiency = 90% relative to text-only training.

**Properties:**
- Load-balanced across GPUs regardless of variable vision input sizes.
- Decouples optimization strategy of vision encoder and main backbone.
- Seamlessly inherits K2's parallel strategy.

---

## 9. Unified Agentic RL Environment

### 9.1 Architecture

**Gym-like interface** with pluggable components:

| Component | Function |
|---|---|
| Toolset module | Tools with sandboxes (web search, code interpreter, web browsing) |
| Judge module | Multi-faceted reward signals |
| Prompt diversification module | Instruction variation |
| Instruction-following enhancement | Format compliance |

### 9.2 Execution Model

- Every agent task: independent asynchronous coroutine.
- Tasks can recursively trigger sub-task rollouts (supports PARL and Agent-as-Judge).
- Rollout Manager orchestrates up to 100,000 concurrent agent tasks.
- Each task acquires environment instance from managed pool (sandbox + specialized tools).
- Partial rollout support.

### 9.3 Inference Engine Co-design

- Token-in-Token-out paradigm.
- Log probabilities recorded for all inference engine outputs.
- Train-inference mismatch correction via log-probability comparison.
- Custom inference APIs for RL requirements.

### 9.4 Black-Box Environment Support

**LLM Gateway:** Proxy service for black-box environments operating under standard LLM API protocol. Records rollout requests/responses under custom protocol, enabling model optimization even when advanced features are unavailable.

---

## 10. Complexity Analysis

### 10.1 Computational Complexity

**Per-token forward pass (MoE backbone):**

$$
\mathcal{O}_{\text{attn}} = \mathcal{O}(L \cdot d_{\text{LLM}}^2) \quad \text{(per attention layer, with MLA compression)}
$$

$$
\mathcal{O}_{\text{MoE-FFN}} = \mathcal{O}(k \cdot d_{\text{LLM}} \cdot d_{\text{FFN}}) \quad \text{(per MoE layer, } k = 8 \text{ active experts)}
$$

**Vision encoder per image:**

$$
\mathcal{O}_{\text{ViT}} = \mathcal{O}(N_p^2 \cdot d_v) \quad \text{(self-attention over } N_p \text{ patches)}
$$

**Video per chunk of 4 frames:**

$$
\mathcal{O}_{\text{ViT-3D}} = \mathcal{O}((4 \cdot N_p)^2 \cdot d_v) = \mathcal{O}(16 \cdot N_p^2 \cdot d_v)
$$

After temporal pooling, context cost reduced to $N_p$ tokens per 4-frame chunk.

### 10.2 Memory Complexity

**Activation memory (per layer, per sequence):**

$$
M_{\text{act}} = \mathcal{O}(L \cdot d_{\text{LLM}}) \quad \text{(after selective recomputation and FP8 compression)}
$$

**KV cache at inference (MLA):**

$$
M_{\text{KV}} = \mathcal{O}(L \cdot d_{\text{compressed}}) \quad \text{(compressed KV via MLA, } d_{\text{compressed}} \ll d_{\text{LLM}} \text{)}
$$

### 10.3 Agent Swarm Complexity

**Sequential agent:** Total steps $= \sum_{t=1}^{T} S^{(t)}$, scaling linearly with task complexity.

**Agent Swarm critical steps:**

$$
C = \sum_{t=1}^{T} \left( S_{\text{main}}^{(t)} + \max_i S_{\text{sub},i}^{(t)} \right)
$$

**Ideal speedup** (balanced decomposition into $n$ parallel subagents):

$$
\text{Speedup} = \frac{\sum_i S_{\text{sub},i}^{(t)}}{\max_i S_{\text{sub},i}^{(t)}} \leq n
$$

Empirically: 3×–4.5× latency reduction on WideSearch.

---

## 11. Convergence Dynamics

### 11.1 Pre-Training

**Early fusion stability:** Monotonic improvement in both vision and text metrics throughout training. No "dip-and-recover" pattern observed (unlike mid/late fusion).

**Mid/Late fusion instability:** Introduction of vision tokens at 50%/80% of training causes transient text performance degradation (modality domain shift), followed by gradual recovery. Final performance inferior to early fusion under same total budget.

### 11.2 Vision RL

Training curves on MMMU-Pro, MathVision, CharXiv(RQ), OCRBench show monotonic accuracy improvement as vision RL FLOPs scale, starting from zero-vision SFT checkpoint.

### 11.3 PARL

Cumulative reward increases smoothly as orchestrator optimizes parallelization strategy. Parallelism level gradually increases during training. Annealing of $\lambda_1, \lambda_2$ ensures convergence to $r_{\text{perf}}$-dominant policy.

### 11.4 Toggle

Stochastic alternating optimization for bi-objective (quality vs. efficiency). Converges to Pareto-efficient solutions: 25–30% token reduction with negligible accuracy loss.

---

## 12. Evaluation Protocol

### 12.1 General Configuration

- Temperature: 1.0
- Top-p: 0.95
- Context length: 256k tokens

### 12.2 Benchmark Taxonomy

| Axis | Benchmarks |
|---|---|
| Reasoning & General | HLE, AIME 2025, HMMT 2025, IMO-AnswerBench, GPQA-Diamond, MMLU-Pro, SimpleQA Verified, AdvancedIF, LongBench v2 |
| Coding | SWE-Bench Verified/Pro/Multilingual, TerminalBench 2.0, PaperBench, CyberGym, SciCode, OJBench, LiveCodeBench v6 |
| Agentic | BrowseComp, WideSearch, DeepSearchQA, FinSearchComp, Seal-0, GDPVal |
| Image | MMMU-Pro, MMMU, CharXiv, MathVision, MathVista, SimpleVQA, WorldVQA, ZeroBench, BabyVision, BLINK, MMVP, OCRBench, OmniDocBench, InfoVQA |
| Video | VideoMMMU, MMVU, MotionBench, Video-MME, LongVideoBench, LVBench |
| Computer Use | OSWorld-Verified, WebArena |

### 12.3 Sampling and Variance Reduction

| Benchmark Type | Protocol |
|---|---|
| AIME 2025, HMMT 2025 | Avg@64 |
| GPQA-Diamond | Avg@8 |
| Image/Video | Avg@3, max 64k tokens |
| Coding | Avg@5 |
| Seal-0, WideSearch | Avg@4 |
| Others | Single run |

### 12.4 Specialized Evaluation Configurations

**Reasoning:** Max 96k completion tokens.

**ZeroBench w/ tools:** Max 24k tokens per step, max 30 steps.

**Video sampling:**
- Short video (VideoMMMU, MMVU, MotionBench): 128 frames, 896 max spatial resolution.
- Long video (Video-MME, LongVideoBench, LVBench): 2048 frames, 448 spatial resolution.

**Computer use:**
- max_steps_per_episode = 100.
- OSWorld: temperature = 0; WebArena: temperature = 0.1.
- One-shot evaluation; last 3 history images + complete thought history + task instruction in context.

### 12.5 Metrics

| Task | Metric |
|---|---|
| Reasoning/General | Accuracy (%) |
| Coding (SWE-Bench) | Resolved rate (%) |
| Agentic search | Item-F1 (%), accuracy (%) |
| Visual grounding | IoU-based F1 |
| OCR | Normalized edit distance |
| OmniDocBench | $(1 - \text{normalized Levenshtein distance}) \times 100$ |
| Computer use | Task success rate (%) |

---

## 13. Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| Visual input ignored after text SFT | Model attends to text, ignores images | Outcome-based visual RL on vision-critical tasks |
| Text-vision SFT hurts generalization | Human-designed visual trajectories are low-diversity | Zero-vision SFT (text-only SFT activates vision) |
| Dip-and-recover in mid/late fusion | Modality domain shift disrupts text representations | Early fusion with constant low vision ratio |
| Serial collapse (Agent Swarm) | Orchestrator defaults to single-agent execution | $r_{\text{parallel}}$ auxiliary reward |
| Spurious parallelism (Agent Swarm) | Spawning many subagents without meaningful decomposition | $r_{\text{finish}}$ auxiliary reward |
| Length-overfitting (Token-efficient RL) | Budget-constrained models fail to leverage extra compute | Toggle alternating optimization |
| Reward hacking (GRM) | Overfitting to single preference signal | Multiple alternative GRM rubrics |
| Credit assignment ambiguity (multi-agent) | Outcome reward cannot attribute to individual subagents | Freeze subagents, optimize only orchestrator |
| Train-inference mismatch (RL) | Off-policy divergence from framework discrepancies | Token-level log-ratio clipping in $[\alpha, \beta]$ |
| PP Stage-0 load imbalance (infra) | Variable vision input sizes cause memory/compute spikes | DEP: decoupled encoder process |
| Context overflow (agentic) | Accumulated trajectory exceeds context window | Agent Swarm as proactive context sharding |
| Temporal information loss (video) | Mean temporal pooling loses high-frequency dynamics | Temporal attention within spatiotemporal volume before pooling |

---

## 14. Key Empirical Results Summary

### 14.1 State-of-the-Art Performance

| Domain | Highlight | Score |
|---|---|---|
| HLE-Full w/ tools | SOTA over Gemini 3 Pro, GPT-5.2 | 50.2% |
| AIME 2025 | Near-perfect | 96.1% |
| BrowseComp (Agent Swarm) | SOTA, surpasses GPT-5.2 Pro | 78.4% |
| OSWorld-Verified | SOTA among open-source | 63.3% |
| MathVision | Frontier competitive | 84.2% |
| LVBench | Global SOTA long-video | 75.9% |

### 14.2 Token Efficiency (Toggle)

| Benchmark | Before Toggle (tokens) | After Toggle (tokens) | Accuracy Change |
|---|---|---|---|
| Average across benchmarks | ~30k | ~21k (25–30% reduction) | Negligible |

### 14.3 Cross-Modal Transfer

Vision RL training improves text benchmarks (MMLU-Pro +1.7, GPQA-Diamond +2.1, LongBench v2 +2.2) without any text-specific RL.

---

## 15. Deployment Constraints

### 15.1 Inference Configuration

- Context window: 256k tokens.
- Activated parameters per token: 32B (MoE routing).
- KV cache: MLA-compressed.
- Video: up to 2048 frames at 448 spatial resolution.

### 15.2 Serving Topology

- MoE routing requires expert parallelism at serving time.
- Agent Swarm: orchestrator + dynamically instantiated subagents require multi-instance serving with async dispatch.
- Rollout Manager supports up to 100,000 concurrent agentic tasks.

### 15.3 Agentic Deployment

- Tool set: web search, code interpreter (Python/Shell), web browsing.
- Agent Swarm tools: `create_subagent`, `assign_task`.
- Step budgets: configurable per benchmark/deployment scenario.
- Context management: proactive (Agent Swarm sharding) preferred over reactive truncation.

### 15.4 Reproducibility

- Fully deterministic data loading via seed/worker-state management.
- Checkpoint release: post-trained Kimi K2.5 on HuggingFace.
- Evaluation scripts and system prompts fully specified.

---

## 16. Pseudo-Algorithm: Full Training Pipeline

```
PROCEDURE KimiK25_Training():

    // === STAGE 1: ViT Training ===
    Load SigLIP-SO-400M as f_ViT
    Load Moonlight-16B-A3B as alignment target
    FOR 1T tokens (caption data: alt text, synthetic, grounding, OCR, video-text):
        z = f_ViT(image_or_video)
        loss = CrossEntropy(MLP(z), caption_tokens)
        Update f_ViT via loss
    Freeze f_ViT
    FOR short_stage:
        Update only g_proj to bridge f_ViT with K2 1T LLM

    // === STAGE 2: Joint Pre-Training ===
    Load K2 near-end checkpoint as h_MoE
    Unfreeze f_ViT and h_MoE
    Set vision_ratio = 10%, text_ratio = 90%  // constant throughout
    FOR 15T mixed vision-text tokens at seq_len=4096:
        Pack mixed sequences (NaViT packing for vision, BPE for text)
        z_vis = g_proj(f_ViT(images_and_videos))
        z_text = Embed(text_tokens)
        z = Concat[z_vis, z_text]
        loss = -mean(log π_θ(x_i | x_{<i}))  // masked for vision input tokens
        Update f_ViT, g_proj, h_MoE via MuonClip

    // === STAGE 3: Long-Context Mid-Training ===
    Apply YaRN interpolation: 4096 → 32768 → 262144
    FOR 500B → 200B tokens (long text, long video, reasoning, long-CoT):
        Same objective, extended context
        Update f_ViT, g_proj, h_MoE

    // === STAGE 4: SFT ===
    Synthesize responses from K2, K2 Thinking, expert models
    Zero-vision SFT: text-only SFT data with IPython tool proxying
    Fine-tune π_θ on instruction-tuning dataset

    // === STAGE 5: Reinforcement Learning ===
    // Phase A: Outcome-Based Visual RL
    FOR visual_rl_iterations:
        Sample vision tasks (grounding, charts, STEM)
        Generate K rollouts from π_old
        Compute rewards (IoU, edit distance, correctness)
        Update π_θ via token-level clipped policy gradient

    // Phase B: Joint Multimodal RL
    FOR joint_rl_iterations:
        Sample tasks organized by ability (not modality)
        Generate K rollouts (text and multimodal queries)
        Compute rewards (rule-based + GRM)
        Apply Toggle: alternate budget-constrained and scaling phases
        Update π_θ via token-level clipped policy gradient + MuonClip

    // Phase C: PARL (Agent Swarm)
    FOR parl_iterations:
        Sample agentic tasks (wide search, deep search, real-world)
        Execute orchestrator with frozen subagents
        Compute r_PARL = r_perf + λ_1 * r_parallel + λ_2 * r_finish
        Update only π_orch
        Anneal λ_1, λ_2 → 0

    RETURN π_K2.5
```

---

## 17. Pseudo-Algorithm: Inference Path

```
PROCEDURE KimiK25_Inference(query, mode):
    IF mode == "single_agent":
        context = [system_prompt, query]
        WHILE not terminated and steps < max_steps:
            IF query contains images/video:
                z_vis = g_proj(f_ViT(visual_inputs))
                context = interleave(z_vis, text_tokens)
            output = π_K2.5.generate(context, temp=1.0, top_p=0.95, max_ctx=256k)
            IF output contains tool_call:
                result = execute_tool(tool_call)
                context.append(result)
            ELSE:
                RETURN output

    IF mode == "agent_swarm":
        Initialize orchestrator context
        WHILE not terminated and critical_steps < budget:
            action = π_orch.generate(context)
            IF action == create_subagent:
                Register new frozen subagent with system prompt
            IF action == assign_task:
                Dispatch to subagent (async, independent context)
                subagent_result = await subagent_completion
                context.append(subagent_result)  // selective routing
            IF action == direct_tool_call:
                result = execute_tool(action)
                context.append(result)
            Update critical_steps += 1 + max(subagent_steps_this_stage)
        RETURN orchestrator final output
```

---

## 18. Pseudo-Algorithm: Validation

```
PROCEDURE Validate(π_K2.5, benchmark_suite):
    FOR each benchmark B in benchmark_suite:
        Load evaluation configuration (temp, top_p, max_tokens, num_runs)
        FOR run = 1 TO num_runs:
            FOR each sample (x, y*) in B:
                IF B.type == "agentic":
                    y_hat = AgenticInference(π_K2.5, x, tools, step_limit)
                ELIF B.type == "vision":
                    y_hat = VisionInference(π_K2.5, x, frames, resolution)
                ELIF B.type == "coding":
                    y_hat = CodingInference(π_K2.5, x, repo_context)
                ELSE:
                    y_hat = π_K2.5.generate(x)
                score[run] = B.metric(y_hat, y*)
        Report mean(score) across runs
```