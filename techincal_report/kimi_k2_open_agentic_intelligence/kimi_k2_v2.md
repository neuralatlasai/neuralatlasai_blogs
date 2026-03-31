

# Kimi K2: End-to-End Technical Report

---

## 1. Data Pipeline

### 1.1 Definition and Objective

The data pipeline for Kimi K2 is a multi-stage system that ingests raw corpora across four primary domains—**Web Text, Code, Mathematics, Knowledge**—and produces 15.5 trillion curated, high-quality tokens optimized for maximum token utility. Token utility is defined as the effective learning signal each token contributes to parameter updates, formally:

$$U(\mathcal{D}) = \frac{\Delta \mathcal{L}(\theta; \mathcal{D})}{|\mathcal{D}|}$$

where $\Delta \mathcal{L}(\theta; \mathcal{D})$ is the reduction in validation loss attributable to dataset $\mathcal{D}$ and $|\mathcal{D}|$ is the token count.

**Inputs:** Raw web crawls, code repositories, mathematical corpora, knowledge-intensive documents.

**Outputs:** Deduplicated, quality-filtered, optionally rephrased token sequences with domain labels and curriculum metadata.

**Invariants:**
- Factual fidelity preservation under rephrasing
- No information loss in chunk-wise processing
- Domain coverage balance maintained throughout pipeline

**Failure Modes:**
- Hallucination injection during rephrasing
- Toxicity amplification in synthetic data
- Over-deduplication causing coverage gaps
- Distribution shift between rephrased and original content

---

### 1.2 Correctness and Quality Validation

For each domain, rigorous correctness and quality validation is performed alongside targeted data experiments to ensure both high diversity and effectiveness. Processing pipelines largely follow methodologies from Kimi K1.5.

---

### 1.3 Rephrasing Pipeline: Knowledge Domain

#### 1.3.1 Motivation

Single-epoch training on knowledge-intensive text is insufficient for comprehensive knowledge absorption. Multi-epoch repetition yields diminishing returns and increases overfitting risk. Rephrasing increases token utility by creating semantically equivalent but linguistically diverse variants.

#### 1.3.2 Pipeline Architecture

The knowledge rephrasing pipeline consists of three components:

**Component 1: Style- and Perspective-Diverse Prompting**

Inspired by WRAP, carefully engineered prompts guide a large language model to generate faithful rephrasings in varied styles and perspectives. Prompts enforce factual integrity constraints while maximizing linguistic diversity.

**Component 2: Chunk-Wise Autoregressive Generation**

To preserve global coherence and avoid information loss in long documents:

1. Input text is split into chunks of approximately 256 tokens
2. Each chunk is rephrased sequentially in an autoregressive manner
3. Previously rephrased chunks serve as context for subsequent chunks
4. Rephrased chunks are concatenated to form the complete output passage

This mitigates implicit output length limitations of LLMs. The pipeline is:

$$\text{Input} \xrightarrow{\text{split}} \{c_1, c_2, \ldots, c_n\} \xrightarrow{\text{rewrite}(c_i | c_{<i})} \{\hat{c}_1, \hat{c}_2, \ldots, \hat{c}_n\} \xrightarrow{\text{concat}} \text{Output}$$

**Component 3: Fidelity Verification**

Semantic alignment between each rephrased passage and its source is verified via a fidelity check that compares source-target consistency as a quality control step prior to training.

#### 1.3.3 Pseudo-Algorithm: Knowledge Rephrasing

```
ALGORITHM: KnowledgeRephrasing
INPUT: Document D, rewrite model M, chunk_size=256, output_size=4096
OUTPUT: Rephrased document D'

1:  chunks ← SPLIT(D, chunk_size)
2:  context ← ∅
3:  rephrased_chunks ← []
4:  FOR i = 1 TO |chunks| DO
5:      prompt ← SAMPLE_STYLE_PROMPT()
6:      c_hat_i ← M.generate(prompt, chunks[i], context)
7:      APPEND(rephrased_chunks, c_hat_i)
8:      context ← context ∪ {c_hat_i}
9:  END FOR
10: D' ← CONCATENATE(rephrased_chunks)
11: fidelity_score ← FIDELITY_CHECK(D, D')
12: IF fidelity_score < threshold THEN
13:     DISCARD(D')
14: END IF
15: RETURN D'
```

#### 1.3.4 Empirical Validation

| # Rephrasings | # Epochs | SimpleQA Accuracy |
|---|---|---|
| 0 (raw wiki-text) | 10 | 23.76 |
| 1 | 10 | 27.39 |
| 10 | 1 | 28.94 |

- Rephrasing 10× with single-pass training outperforms 10-epoch repetition of raw data by **+5.18 points**
- Each corpus is rephrased at most twice in full-scale training to balance quality and risk

---

### 1.4 Rephrasing Pipeline: Mathematics Domain

High-quality mathematical documents are rewritten into a "learning-note" style following SwallowMath methodology. Additionally, high-quality mathematical materials from other languages are translated into English to increase data diversity.

---

### 1.5 Pre-training Data Composition

The final corpus: **15.5 trillion tokens** spanning:

| Domain | Description |
|---|---|
| Web Text | Filtered, deduplicated web crawl |
| Code | Source code with quality validation |
| Mathematics | Native + rephrased + translated mathematical content |
| Knowledge | Knowledge-intensive text with chunk-wise rephrasing |

---

## 2. Model Architecture

### 2.1 Definition

Kimi K2 is a **1.04 trillion-parameter Mixture-of-Experts (MoE) Transformer** with **32 billion activated parameters** per forward pass. The architecture employs Multi-head Latent Attention (MLA) as the attention mechanism with an ultra-sparse expert configuration.

### 2.2 Architectural Specification

| Parameter | DeepSeek-V3 | Kimi K2 | Delta |
|---|---|---|---|
| Layers | 61 | 61 | = |
| Total Parameters | 671B | 1.04T | ↑ 54% |
| Activated Parameters | 37B | 32.6B | ↓ 13% |
| Total Experts | 256 | 384 | ↑ 50% |
| Active Experts per Token | 8 | 8 | = |
| Shared Experts | 1 | 1 | = |
| Attention Heads | 128 | 64 | ↓ 50% |
| Dense Layers | 3 | 1 | ↓ 67% |
| Expert Grouping | Yes | No | - |
| Model Hidden Dimension | — | 7168 | — |
| Expert Hidden Dimension | — | 2048 | — |

### 2.3 Core Architectural Components

#### 2.3.1 Multi-Head Latent Attention (MLA)

MLA compresses key-value representations into a low-rank latent space to reduce KV-cache memory during inference. For each attention head $h$:

$$q_h = W_h^q X, \quad k_h = W_h^k X, \quad v_h = W_h^v X$$

The attention output:

$$\text{Attn}(q_h, k_h, v_h) = \text{softmax}\left(\frac{q_h k_h^\top}{\sqrt{d_k}}\right) v_h$$

MLA decomposes projections into:
- $q^C, k^C$: **head-specific compressed components**
- $q^R$: **head-specific rotary component**
- $k^R$: **shared rotary component** (shared across heads)

This decomposition is critical for QK-Clip applicability (Section 3).

#### 2.3.2 Mixture-of-Experts (MoE) Layer

Each MoE layer contains 384 routed experts plus 1 shared expert. Per token, a gating function selects 8 out of 384 experts:

$$\text{MoE}(x) = \sum_{i \in \text{TopK}(g(x), 8)} g_i(x) \cdot E_i(x) + E_{\text{shared}}(x)$$

where $g(x)$ is the gating logits and $E_i$ is the $i$-th expert FFN with hidden dimension 2048.

**Sparsity ratio:**

$$\text{Sparsity} = \frac{\text{Total Experts}}{\text{Active Experts}} = \frac{384}{8} = 48$$

#### 2.3.3 Sparsity Scaling Law

Under fixed activated parameters (constant FLOPs), increasing total experts (increasing sparsity) consistently lowers training and validation loss.

Quantitative gains at validation loss 1.5:
- Sparsity 48 reduces FLOPs by **1.69×** vs. sparsity 8
- Sparsity 48 reduces FLOPs by **1.39×** vs. sparsity 16
- Sparsity 48 reduces FLOPs by **1.15×** vs. sparsity 32

The compute-optimal sparsity scaling law for MoE with Muon shows monotonic improvement with increased sparsity, subject to infrastructure complexity trade-offs.

#### 2.3.4 Attention Head Count Analysis

Doubling attention heads from 64 to 128 yields only **0.5%–1.2%** improvement in validation loss under iso-token conditions. However, at 128K sequence length with 384 experts, increasing heads from 64→128 causes **83% increase in inference FLOPs**. Given sparsity 48 already provides strong performance, 64 heads is the Pareto-optimal choice balancing quality and inference cost.

### 2.4 Tensor Dimensions and Memory Flow

For a single forward pass through one transformer layer:

**Input:** $X \in \mathbb{R}^{B \times L \times 7168}$ where $B$ = batch size, $L$ = sequence length

**MLA Block:**
- Latent compression: $X \rightarrow c \in \mathbb{R}^{B \times L \times d_c}$ (compressed KV)
- Head-specific decomposition: $c \rightarrow q_h, k_h, v_h$ for $h = 1, \ldots, 64$
- Attention computation: $\mathbb{R}^{B \times 64 \times L \times d_k} \rightarrow \mathbb{R}^{B \times L \times 7168}$

**MoE Block:**
- Gating: $X \in \mathbb{R}^{B \times L \times 7168} \rightarrow g \in \mathbb{R}^{B \times L \times 384}$
- Expert routing: Top-8 selection per token
- Expert FFN: $\mathbb{R}^{7168} \rightarrow \mathbb{R}^{2048} \rightarrow \mathbb{R}^{7168}$ (per expert with SwiGLU)
- Weighted combination + shared expert output

**Total parameter count:**
- 61 layers × (MLA parameters + MoE parameters with 384 experts + shared expert + LayerNorm)
- Total: **1.04T parameters**, **32.6B activated** per token

---

## 3. Optimization Strategy: MuonClip

### 3.1 Definition

MuonClip is a novel optimizer that integrates the token-efficient Muon algorithm with weight decay, consistent RMS scaling, and a stability-enhancing mechanism called **QK-Clip**. It addresses training instability caused by exploding attention logits that occur more frequently with Muon than AdamW at scale.

### 3.2 Muon Base Optimizer

For each weight $W \in \mathbb{R}^{n \times m}$:

**Momentum update:**

$$M_t = \mu M_{t-1} + G_t, \quad M_0 = 0$$

where $G_t = \nabla_W \mathcal{L}$ is the gradient at step $t$, $\mu$ is the momentum coefficient.

**Newton-Schulz orthogonalization:**

$$O_t = \text{Newton-Schulz}(M_t) \cdot \sqrt{\max(n, m)} \cdot 0.2$$

The $\sqrt{\max(n, m)} \cdot 0.2$ factor provides **consistent RMS matching** to align update magnitudes with AdamW-scale dynamics.

**Parameter update with weight decay:**

$$W_t = W_{t-1} - \eta (O_t + \lambda W_{t-1})$$

where $\eta$ is the learning rate and $\lambda$ is the weight decay coefficient.

**Newton-Schulz iteration** approximates the matrix sign function / polar decomposition to produce an orthogonal update direction, which is the core mechanism enabling Muon's superior token efficiency over AdamW.

### 3.3 QK-Clip Mechanism

#### 3.3.1 Problem Statement

Scaling Muon training reveals instability due to exploding attention logits. In mid-scale experiments (9B activated, 53B total MoE), maximum attention logits exceed 1000 rapidly, leading to loss spikes and potential divergence.

**Existing mitigations and their failures:**
- **Logit soft-cap:** Clips attention logits post-computation, but dot products between queries and keys can grow excessively before capping
- **QK-Norm:** Not applicable to MLA because Key matrices are not fully materialized during inference

#### 3.3.2 Core Principle

QK-Clip rescales query and key projection weights **post-update** to bound attention logit growth. It does not alter forward/backward computation in the current step—it uses the max logit as a guiding signal to control weight growth.

#### 3.3.3 Mathematical Formulation

**Max logit computation per head $h$:**

$$S_h^{\max} = \max_{i,j} q_h(x_i)^\top k_h(x_j)$$

where $i, j$ are token indices within training batch $B$.

**Per-head scaling factor:**

$$\gamma_h = \min\left(1, \frac{\tau}{S_h^{\max}}\right)$$

where $\tau$ is the target threshold (set to 100 for Kimi K2).

**Naïve implementation (global clip):**

$$W^q \leftarrow W^q \cdot \gamma^\alpha, \quad W^k \leftarrow W^k \cdot \gamma^{1-\alpha}$$

where $\gamma = \min(1, \tau / S^{\max})$ with $S^{\max} = \max_h S_h^{\max}$, and $\alpha = 0.5$.

**Per-head MLA implementation (preferred):**

Since only a small subset of heads exhibit exploding logits, per-head clipping minimizes intervention:

- $W_h^{q^C}$ (head-specific compressed query): scaled by $\sqrt{\gamma_h}$
- $W_h^{k^C}$ (head-specific compressed key): scaled by $\sqrt{\gamma_h}$
- $W_h^{q^R}$ (head-specific rotary query): scaled by $\gamma_h$
- $k^R$ (shared rotary key): **untouched** to avoid cross-head interference

**Correctness invariant:** The product $q \cdot k^\top$ is scaled by $\gamma_h$ total:

$$\sqrt{\gamma_h} \cdot \sqrt{\gamma_h} = \gamma_h \quad \text{(compressed components)}$$

$$\gamma_h \cdot 1 = \gamma_h \quad \text{(rotary components)}$$

#### 3.3.4 Convergence Dynamics

During Kimi K2 training with $\tau = 100$:
1. **Early phase:** Max logits are actively capped at 100 by QK-Clip
2. **Transition phase (~30% of training):** Max logits gradually decay from the cap
3. **Stable phase:** Max logits settle into natural operating range without further intervention
4. **Result:** Zero loss spikes throughout entire 15.5T token training

### 3.4 Pseudo-Algorithm: MuonClip

```
ALGORITHM: MuonClip
INPUT: Model parameters Θ, learning rate η, weight decay λ,
       momentum μ, clip threshold τ
OUTPUT: Updated parameters Θ

1:  FOR each training step t DO
2:      // Phase 1: Muon optimizer step
3:      FOR each weight W ∈ ℝ^{n×m} DO
4:          G_t ← GRADIENT(W_t, loss)
5:          M_t ← μ · M_{t-1} + G_t
6:          O_t ← NEWTON_SCHULZ(M_t) · √(max(n,m)) · 0.2
7:          W_t ← W_{t-1} - η · (O_t + λ · W_{t-1})
8:      END FOR
9:
10:     // Phase 2: QK-Clip
11:     FOR each attention head h in every layer DO
12:         S_h^max ← retrieve max attention logit from forward pass
13:         IF S_h^max > τ THEN
14:             γ ← τ / S_h^max
15:             W_h^{qC} ← W_h^{qC} · √γ
16:             W_h^{kC} ← W_h^{kC} · √γ
17:             W_h^{qR} ← W_h^{qR} · γ
18:             // k^R (shared rotary) left untouched
19:         END IF
20:     END FOR
21: END FOR
```

### 3.5 Comparison: MuonClip vs. Alternatives

| Property | AdamW | Muon | MuonClip |
|---|---|---|---|
| Token efficiency | Baseline | Substantially higher | Same as Muon |
| Training stability at scale | Stable | Unstable (logit explosion) | Stable (zero spikes) |
| MLA compatibility | Yes | Yes | Yes (per-head clip) |
| QK-Norm applicable | Yes | Yes | N/A (uses QK-Clip instead) |
| Logit soft-cap sufficient | Yes | No (pre-cap growth) | N/A (uses QK-Clip) |

---

## 4. Training Pipeline

### 4.1 Training Infrastructure

#### 4.1.1 Compute Cluster

- **Hardware:** NVIDIA H800 GPUs
- **Per-node:** 2 TB RAM, 8 GPUs connected via NVLink + NVSwitch (intra-node)
- **Inter-node:** 8 × 400 Gbps RoCE interconnects

#### 4.1.2 Parallelism Strategy

The parallelism configuration allows training on any number of nodes that is a multiple of 32:

| Parallelism Type | Degree | Purpose |
|---|---|---|
| Pipeline Parallelism (PP) | 16-way (virtual stages) | Model layer distribution |
| Expert Parallelism (EP) | 16-way | Expert distribution across devices |
| ZeRO-1 Data Parallelism | Variable | Optimizer state sharding |

**Memory budget per model-parallel group (256 GPUs):**

$$\text{Memory}_{\text{params+grads}} = \text{BF16 params} + \text{FP32 grad accum} \approx 6 \text{ TB}$$

Per GPU: ~30 GB for model states, remainder for activations.

**Design principle:** Identical parallelism configuration for both small- and large-scale experiments to maximize research efficiency.

#### 4.1.3 EP Communication Overlap

EP all-to-all communication is overlapped with computation under interleaved 1F1B schedule by increasing warm-up micro-batches.

**Why not DualPipe:** DualPipe doubles memory for parameters and gradients, requiring increased parallelism to compensate. For 1T+ models, these costs are prohibitive.

**PP communication overlap:** Weight-gradient computation is decoupled from backward pass and executed in parallel with PP communication. All PP communications are overlapped except during warm-up phase.

**EP = 16 (smallest feasible):** The reduced attention computation (64 heads vs. 128) necessitates minimizing EP operation time. Smaller EP also relaxes expert-balance constraints.

### 4.2 Activation Reduction

#### 4.2.1 Selective Recomputation

Recomputation applied to:
- LayerNorm
- SwiGLU activations
- MLA up-projections
- MoE down-projections (optional, prevents crashes from expert imbalance)

#### 4.2.2 FP8 Storage for Insensitive Activations

$$\text{Inputs to MoE up-projections and SwiGLU} \xrightarrow{\text{quantize}} \text{FP8-E4M3}$$

Quantization scheme: $1 \times 128$ tiles with FP32 scales. Small-scale experiments show no measurable loss increase. FP8 is **not** applied in computation due to observed performance degradation risks.

#### 4.2.3 Activation CPU Offload

All remaining activations are offloaded to CPU RAM via a copy engine that overlaps offload/onload with computation and communication:

- **1F1B phase:** Offload forward activations of previous micro-batch; prefetch backward activations of next micro-batch
- **Warm-up/cool-down:** Handled similarly
- PCIe traffic congestion slightly affects EP traffic, but EP communication remains fully overlapped

### 4.3 Training Recipe

#### 4.3.1 Pre-training Schedule

| Phase | Tokens | Context Length | Learning Rate | Batch Size |
|---|---|---|---|---|
| Warm-up | 500 steps | 4,096 | 0 → 2e-4 | 67M tokens |
| Constant LR | 10T | 4,096 | 2e-4 | 67M tokens |
| Cosine Decay | 5.5T | 4,096 | 2e-4 → 2e-5 | 67M tokens |
| Annealing | 400B | 4,096 | 2e-5 → 7e-6 | 67M tokens |
| Long-context | 60B | 32,768 | continued decay | 67M tokens |

**Total: 15.5T tokens**

**Weight decay:** $\lambda = 0.1$ throughout.

**Learning rate schedule:** WSD (Warm-up, Stable, Decay)

$$\eta(t) = \begin{cases} \eta_{\max} \cdot \frac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\ \eta_{\max} & t_{\text{warmup}} < t \leq t_{\text{stable}} \\ \eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\pi \cdot \frac{t - t_{\text{stable}}}{t_{\text{total}} - t_{\text{stable}}}\right)\right) & t > t_{\text{stable}} \end{cases}$$

#### 4.3.2 Long-Context Extension

Context window extended to **128K** using YaRN (Yet another RoPE extensioN) method applied after the 32K training phase.

#### 4.3.3 Training Stability Result

**Zero loss spikes** throughout the entire 15.5T token pre-training, validated by unsmoothed, unsubsampled per-step loss curves. This is the direct result of MuonClip with $\tau = 100$.

---

## 5. Post-Training Pipeline

### 5.1 Supervised Fine-Tuning (SFT)

#### 5.1.1 Optimizer

Muon optimizer is used for SFT, following the conclusion that Muon-pretrained checkpoints produce best performance with Muon fine-tuning.

#### 5.1.2 Data Construction Principles

1. **Maximize prompt diversity** across domains
2. **Ensure high response quality** via automated and human evaluation

**Pipeline:** K1.5 and domain-specialized expert models generate candidate responses → LLM-based or human judges perform quality evaluation and filtering.

### 5.2 Agentic Data Synthesis Pipeline

#### 5.2.1 Pipeline Architecture (Three Stages)

**Stage 1: Tool Spec Generation**

Two complementary approaches:

| Source | Count | Method |
|---|---|---|
| Real MCP tools | 3,000+ | Fetched from GitHub repositories |
| Synthetic tools | 20,000+ | Hierarchical domain evolution |

**Domain evolution process:**
1. Begin with key categories (financial trading, software applications, robot control, etc.)
2. Evolve multiple specific application domains within each category
3. Synthesize specialized tools for each domain with clear interfaces, descriptions, operational semantics

**Stage 2: Agent and Task Generation**

- **Agent diversification:** Thousands of distinct agents generated via synthesized system prompts + different tool-set combinations from repository
- **Rubric-based task generation:** Tasks range from simple to complex operations. Each task paired with explicit rubric specifying:
  - Success criteria
  - Expected tool-use patterns
  - Evaluation checkpoints

**Stage 3: Trajectory Generation**

Multi-turn trajectory generation through:

- **User Simulation:** LLM-generated user personas with distinct communication styles engage in multi-turn dialogues
- **Tool Execution Environment:** A tool simulator (world model) that:
  - Executes tool calls with realistic feedback
  - Maintains and updates state after each execution
  - Introduces controlled stochasticity (successes, partial failures, edge cases)
- **Quality Evaluation:** LLM-based judge evaluates each trajectory against task rubrics; only passing trajectories retained

#### 5.2.2 Hybrid Approach: Simulation + Real Execution

Simulated environments provide scalability. Real execution sandboxes provide authenticity for coding/software engineering tasks:
- Execute actual code
- Interact with genuine development environments
- Provide ground-truth feedback via test suite pass rates

This combination implements **large-scale rejection sampling** through quality filtering.

#### 5.2.3 Pseudo-Algorithm: Agentic Data Synthesis

```
ALGORITHM: AgenticDataSynthesis
INPUT: Real MCP tools R, domain categories C, LLM generator G, judge J
OUTPUT: Filtered trajectory dataset T

1:  // Stage 1: Tool Repository Construction
2:  T_real ← FETCH_MCP_TOOLS(GitHub)
3:  T_synth ← ∅
4:  FOR each category c ∈ C DO
5:      domains ← EVOLVE_DOMAINS(c, G)
6:      FOR each domain d ∈ domains DO
7:          tools ← SYNTHESIZE_TOOLS(d, G)
8:          T_synth ← T_synth ∪ tools
9:      END FOR
10: END FOR
11: TOOL_REPO ← T_real ∪ T_synth

12: // Stage 2: Agent and Task Generation
13: AGENTS ← ∅, TASKS ← ∅
14: FOR i = 1 TO N_agents DO
15:     toolset ← SAMPLE(TOOL_REPO)
16:     system_prompt ← GENERATE_PROMPT(G, toolset)
17:     agent ← CREATE_AGENT(system_prompt, toolset)
18:     tasks_with_rubrics ← GENERATE_TASKS(G, agent)
19:     AGENTS ← AGENTS ∪ {agent}
20:     TASKS ← TASKS ∪ tasks_with_rubrics
21: END FOR

22: // Stage 3: Trajectory Generation
23: T ← ∅
24: FOR each (agent, task, rubric) ∈ (AGENTS × TASKS) DO
25:     user ← GENERATE_USER_PERSONA(G)
26:     simulator ← INITIALIZE_TOOL_SIMULATOR(agent.toolset)
27:     trajectory ← []
28:     WHILE NOT task_complete AND NOT max_turns DO
29:         user_msg ← user.generate_message()
30:         agent_response ← agent.act(user_msg, simulator)
31:         observation ← simulator.execute(agent_response.tool_call)
32:         simulator.update_state(observation)
33:         APPEND(trajectory, (user_msg, agent_response, observation))
34:     END WHILE
35:     score ← J.evaluate(trajectory, rubric)
36:     IF score ≥ threshold THEN
37:         T ← T ∪ {trajectory}
38:     END IF
39: END FOR
40: RETURN T
```

---

### 5.3 Reinforcement Learning

#### 5.3.1 RL Objective

For each problem $x$, sample $K$ responses $\{y_1, \ldots, y_K\}$ from previous policy $\pi_{\text{old}}$, and optimize $\pi_\theta$:

$$\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\} \sim \pi_{\text{old}}(\cdot|x)} \left[ \sum_{i=1}^{K} \frac{\left(r(x, y_i) - \bar{r}(x)\right)}{\max\left(\text{std}(r(x,y_i)), \tau\right)} \log \pi_\theta(y_i | x) \right]$$

where:
- $\bar{r}(x) = \frac{1}{K} \sum_{i=1}^{K} r(x, y_i)$ is the mean reward of sampled responses
- $\tau > 0$ is a regularization parameter promoting stable learning
- The advantage is normalized by the standard deviation (clipped at $\tau$)

Muon optimizer is used for RL training.

#### 5.3.2 Verifiable Rewards Gym

**Math, STEM, Logical Tasks:**

Two key data principles:
1. **Diverse coverage:** Expert annotations + internal QA extraction + open datasets + tagging system for under-covered domains. Logical tasks: tabular reasoning, cross-table aggregation, 24-game, Sudoku, riddles, cryptarithms, Morse-code decoding.
2. **Moderate difficulty:** Assessed via SFT model's pass@k accuracy; only problems with moderate difficulty selected. Too-easy and too-hard problems produce little learning signal.

**Complex Instruction Following:**

Hybrid verification framework:

- **Deterministic verification:** Code interpreters for verifiable outputs (length, style constraints)
- **LLM-as-judge:** For nuanced constraint understanding
- **Hack-check layer:** Detects adversarial behaviors where models claim compliance without actual fulfillment

Multi-source instruction generation:
1. Expert-crafted complex conditional prompts/rubrics
2. Agentic instruction augmentation (AutoIF-inspired)
3. Fine-tuned model for generating instructions targeting specific failure modes

**Faithfulness:**

Sentence-level faithfulness judge model trained using FACTS Grounding evaluation framework. Detects factual claims without supporting evidence in context. Serves as reward model.

**Coding & Software Engineering:**

- Competition-level problems with judges from open-source datasets and synthetic sources
- Human-written unit tests retrieved from pre-training data for diversity and correctness
- Pull requests and issues from GitHub for software development environments
- Sandbox infrastructure powered by Kubernetes: 10,000+ concurrent instances

**Safety:**

Human-curated seed prompts across risk categories → automated prompt evolution pipeline:
1. **Attack Model:** Generates adversarial prompts
2. **Target Model:** Produces responses
3. **Judge Model:** Binary success/failure label per rubric

#### 5.3.3 Self-Critique Rubric Reward

**Purpose:** Extend alignment beyond verifiable-reward tasks to subjective domains (helpfulness, creativity, factuality, safety).

**Mechanism: Self-Critiqued Policy Optimization**

1. K2 actor generates responses for general prompts
2. K2 critic ranks all results via pairwise evaluation against rubrics:
   - **Core rubrics:** Fundamental AI assistant values
   - **Prescriptive rubrics:** Eliminate reward hacking
   - **Human-annotated rubrics:** Specific instructional contexts
3. Certain rubrics are mandatory; K2 retains flexibility to weigh others against internal priors

**Closed-Loop Critic Refinement:**

$$\text{RLVR signals} \xrightarrow{\text{on-policy rollouts}} \text{Critic update} \xrightarrow{\text{transfer learning}} \text{Subjective judgment improvement}$$

On-policy rollouts from verifiable-reward prompts continuously update the critic, distilling objective performance signals directly into evaluation capabilities. This grounds subjective judgments in verifiable data.

#### 5.3.4 RL Algorithm Additions

**Budget Control:**

Per-sample maximum token budget enforced based on task type. Responses exceeding budget are truncated and assigned a penalty:

$$r_{\text{budget}}(x, y) = \begin{cases} r(x, y) & \text{if } |y| \leq B(x) \\ r_{\text{penalty}} & \text{if } |y| > B(x) \end{cases}$$

where $B(x)$ is the task-type-specific budget. This incentivizes concise yet effective solutions.

**PTX Loss:**

Auxiliary pre-training loss on curated high-quality samples integrated into RL objective to prevent catastrophic forgetting:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}} + \beta \cdot \mathcal{L}_{\text{PTX}}$$

where $\mathcal{L}_{\text{PTX}}$ is the standard language modeling loss on hand-selected samples and $\beta$ is a balancing coefficient.

**Temperature Decay:**

$$T(t) = T_{\text{init}} \cdot \text{decay\_schedule}(t)$$

- **Early stages:** High temperature promotes exploration, diverse responses, discovery of effective strategies
- **Later stages:** Low temperature shifts to exploitation, consistent high-quality outputs
- Addresses premature convergence risk while ensuring final output stability

#### 5.3.5 Pseudo-Algorithm: RL Training Loop

```
ALGORITHM: KimiK2_RL_Training
INPUT: SFT checkpoint θ_0, task environments E, reward functions R,
       PTX dataset D_ptx, budget map B, temperature schedule T
OUTPUT: RL-trained parameters θ*

1:  θ ← θ_0
2:  FOR each RL iteration t DO
3:      // Rollout phase
4:      temp ← T(t)
5:      rollouts ← ∅
6:      FOR each task x sampled from E DO
7:          FOR k = 1 TO K DO
8:              y_k ← GENERATE(π_old, x, temperature=temp)
9:              IF |y_k| > B(x) THEN
10:                 y_k ← TRUNCATE(y_k, B(x))
11:                 r_k ← r_penalty
12:             ELSE
13:                 // Verifiable reward or self-critique reward
14:                 IF x ∈ VERIFIABLE_TASKS THEN
15:                     r_k ← VERIFY(x, y_k)
16:                 ELSE
17:                     r_k ← SELF_CRITIQUE(π_critic, x, y_k, rubrics)
18:                 END IF
19:             END IF
20:             APPEND(rollouts, (x, y_k, r_k))
21:         END FOR
22:     END FOR
23:
24:     // Compute advantages
25:     FOR each problem x DO
26:         r_bar ← MEAN(rewards for x)
27:         r_std ← STD(rewards for x)
28:         FOR each (x, y_k, r_k) DO
29:             advantage_k ← (r_k - r_bar) / max(r_std, τ)
30:         END FOR
31:     END FOR
32:
33:     // Training phase
34:     L_RL ← Σ advantage_k · log π_θ(y_k | x)
35:     L_PTX ← -Σ log π_θ(y | x) for (x,y) ∈ D_ptx
36:     L_total ← L_RL + β · L_PTX
37:     θ ← MUON_UPDATE(θ, ∇L_total)
38:
39:     // Critic refinement (closed-loop)
40:     IF t mod critic_update_freq == 0 THEN
41:         UPDATE_CRITIC(π_critic, verifiable_rollouts)
42:     END IF
43:
44:     // Sync parameters to inference engine
45:     CHECKPOINT_ENGINE_BROADCAST(θ)
46: END FOR
47: RETURN θ
```

---

## 6. RL Infrastructure

### 6.1 Colocated Architecture

Training and inference engines colocate on the same workers. When one engine is active, the other releases/offloads GPU resources.

**Iteration cycle:**
1. Centralized controller calls inference engine → generate rollout data
2. Controller notifies training engine → train on new data
3. Updated parameters sent to inference engine for next iteration

### 6.2 Efficient Engine Switching

#### 6.2.1 Problem

Inference engine requires updated parameters from training engine with a **different sharding paradigm**. Network file system resharding is impractical at 1T scale (requires petabytes/second aggregate bandwidth).

#### 6.2.2 Distributed Checkpoint Engine

Co-located on training nodes:

1. Each checkpoint engine worker obtains local copy of parameters from training engine
2. Full parameter set broadcast across all checkpoint engine workers
3. Inference engine retrieves only its required shard from checkpoint engine

**Pipelining:** Updates performed parameter-by-parameter to minimize memory footprint.

**Design choice:** Full parameter broadcast across entire cluster (transfers several times more than optimal) trades minor overhead for full decoupling of training and inference engines.

**Performance:** Full parameter update for 1T model completed in **< 30 seconds**.

### 6.3 Efficient System Startup

**Training engine startup:**
- Each worker selectively reads part/none of parameters from disk
- Broadcasts necessary parameters to peers
- All workers collectively read checkpoint only once (minimizes disk IO)

**Inference engine startup:**
- Checkpoint engine collectively reads from disk
- Updates uninitialized inference engine state
- No synchronization barriers between inference replicas
- Robust to single-point failures (replicas restart independently)

### 6.4 Agentic Rollout Optimizations

**Challenge 1: GPU idle time during environment interactions**

Solutions:
1. Deploy heavy environments as dedicated scalable services
2. Large number of concurrent rollouts to amortize expensive interaction latency

**Challenge 2: Long-tail trajectories blocking rollout**

Solution: **Partial rollout** — long-tail unfinished tasks are paused and resumed in the next RL iteration.

**Unified interface:** OpenAI Gym-inspired framework for streamlined integration of new environments.

---

## 7. Compression Pipeline (Activation Compression)

### 7.1 FP8 Activation Compression

#### 7.1.1 Compression Equation

For activation tensor $A \in \mathbb{R}^{B \times L \times D}$:

$$A_{\text{FP8}} = \text{Quantize}_{\text{E4M3}}(A, \text{tile\_size} = 1 \times 128)$$

Each tile of 128 elements shares a single FP32 scale factor $s$:

$$A_{\text{FP8}}[i] = \text{round}\left(\frac{A[i]}{s}\right), \quad s = \frac{\max(|A_{\text{tile}}|)}{E4M3\_\text{max}}$$

**Compression ratio:**

$$\text{CR} = \frac{16 \text{ bits (BF16)}}{8 \text{ bits (FP8)} + \frac{32}{128} \text{ bits (scale amortized)}} = \frac{16}{8.25} \approx 1.94\times$$

**Information preservation:** Small-scale experiments confirm no measurable loss increase from FP8 activation storage.

#### 7.1.2 Scope

Applied to: MoE up-projection inputs, SwiGLU inputs.
**Not** applied to: Computation kernels (observed degradation risk).

### 7.2 Selective Recomputation as Implicit Compression

Instead of storing activations, cheap-to-compute but memory-heavy activations are recomputed:

| Component | Stored? | Rationale |
|---|---|---|
| LayerNorm output | Recomputed | Low FLOP, high memory |
| SwiGLU intermediate | Recomputed | Low FLOP, high memory |
| MLA up-projection | Recomputed | Low FLOP, high memory |
| MoE down-projection | Recomputed | Prevents OOM from expert imbalance |

### 7.3 CPU Offload as Memory Extension

Remaining activations offloaded to 2 TB CPU RAM per node. Copy engine overlaps transfers with computation/communication.

**Effective memory hierarchy:**

$$\text{GPU HBM} \xrightarrow{\text{FP8 quantized}} \text{GPU HBM (compressed)} \xrightarrow{\text{offload}} \text{CPU DRAM}$$

---

## 8. Inference Path

### 8.1 Architecture at Inference

**Activated parameters per token:** 32.6B out of 1.04T total

**MLA Inference Advantage:**
- KV-cache compression via latent representation
- Shared $k^R$ across heads reduces storage
- 64 attention heads (vs. 128 in DeepSeek-V3) → lower KV-cache footprint

**Inference FLOP analysis at 128K sequence length:**
- 64 heads: baseline inference FLOPs
- 128 heads: +83% inference FLOPs
- Justification for 64-head choice validated at deployment scale

### 8.2 Context Window

- Pre-trained at 4K tokens
- Extended to 32K via continued training (60B tokens)
- Extended to 128K via YaRN positional interpolation

### 8.3 Inference Constraints for Agentic Deployment

**Output token cap:** 8,192 tokens standard; 16,384 for SWE-bench Verified Agentless.

**Agentic modes:**
- **Agentless Coding:** Single patch generation without test execution
- **Agentic Coding (Single Attempt):** Bash/editor tools, single trajectory
- **Agentic Coding (Multi Attempt):** Best-of-N selection with internal verifier

### 8.4 Pseudo-Algorithm: Inference Serving

```
ALGORITHM: KimiK2_Inference
INPUT: Request x, model θ, max_tokens, temperature, tools (optional)
OUTPUT: Response y

1:  tokens ← TOKENIZE(x)
2:  IF |tokens| > 128K THEN
3:      tokens ← TRUNCATE(tokens, 128K)
4:  END IF
5:  kv_cache ← INITIALIZE_MLA_CACHE()
6:  generated ← []
7:  WHILE |generated| < max_tokens AND NOT EOS DO
8:      // Forward pass: 32.6B activated params
9:      FOR layer l = 1 TO 61 DO
10:         // MLA attention (64 heads, latent KV)
11:         h ← MLA_ATTENTION(tokens, kv_cache[l])
12:         UPDATE_KV_CACHE(kv_cache[l], h)
13:         // MoE routing (8/384 experts + 1 shared)
14:         gate_logits ← GATING(h)
15:         top8 ← TOP_K(gate_logits, 8)
16:         h ← Σ gate[i] · EXPERT[i](h) + SHARED_EXPERT(h)
17:     END FOR
18:     logits ← LM_HEAD(h)
19:     // Tool use check (agentic mode)
20:     IF TOOL_CALL_DETECTED(logits) AND tools ≠ ∅ THEN
21:         tool_call ← PARSE_TOOL_CALL(logits)
22:         observation ← EXECUTE_TOOL(tool_call)
23:         tokens ← APPEND(tokens, observation)
24:         CONTINUE
25:     END IF
26:     next_token ← SAMPLE(logits, temperature)
27:     APPEND(generated, next_token)
28: END WHILE
29: RETURN DETOKENIZE(generated)
```

---

## 9. Evaluation Protocol

### 9.1 Benchmark Suite

| Category | Benchmarks |
|---|---|
| **Coding** | LiveCodeBench v6, OJBench, MultiPL-E, SWE-bench Verified, TerminalBench, Multi-SWE-bench, SWE-Lancer, PaperBench, Aider-Polyglot |
| **Tool Use** | τ2-Bench, ACEBench |
| **Math & STEM** | AIME 2024/2025, MATH-500, HMMT 2025, CNMO 2024, PolyMath-en, ZebraLogic, AutoLogi, GPQA-Diamond, SuperGPQA, HLE |
| **Long Context** | MRCR, DROP, FRAMES, LongBench v2 |
| **Factuality** | FACTS Grounding, HHEM v2.1, FaithJudge |
| **General** | MMLU, MMLU-Redux, MMLU-Pro, IFEval, Multi-Challenge, SimpleQA, LiveBench, Arena Hard v2.0 |

### 9.2 Evaluation Configurations

| Setting | Value |
|---|---|
| Mode | Non-thinking for all models |
| Max output tokens | 8,192 (standard), 16,384 (SWE-bench Agentless) |
| Context window | 128K (truncate if exceeded) |
| High-variance benchmarks | Avg@k (repeated sampling k times, averaged) |
| SWE-bench Verified modes | Agentless-Single-Patch, Agentic-Single-Attempt, Agentic-Multi-Attempt |

### 9.3 Baselines

**Open-source:** DeepSeek-V3-0324, Qwen3-235B-A22B (no-thinking mode)

**Proprietary:** Claude Sonnet 4, Claude Opus 4, GPT-4.1, Gemini 2.5 Flash Preview

### 9.4 Pre-training Evaluation

**Perplexity-based:** MMLU, MMLU-Redux, GPQA-Diamond, HellaSwag, ARC-Challenge, C-Eval, CMMLU

**Generation-based:** MMLU-Pro, SuperGPQA, TriviaQA, BBH, CSimpleQA, MATH, CMATH, GSM8K, GSM8K-Platinum, CRUXEval, LiveCodeBench, EvalPlus

GPQA-Diamond: mean across 8 independent runs (variance mitigation).

### 9.5 Key Results Summary

| Benchmark | Kimi K2 | Best Baseline | Gap |
|---|---|---|---|
| SWE-bench Verified (Agentic) | 65.8 | 72.5 (Opus 4) | -6.7 |
| SWE-bench Multilingual | 47.3 | 51.0 (Sonnet 4) | -3.7 |
| LiveCodeBench v6 | **53.7** | 47.4 (Opus 4) | **+6.3** |
| OJBench | **27.1** | 24.0 (DS-V3) | **+3.1** |
| τ2-Bench (micro-avg) | 66.1 | 67.6 (DS-V3) | -1.5 |
| ACEBench (en) | 76.5 | 80.1 (GPT-4.1) | -3.6 |
| AIME 2025 (Avg@64) | **49.5** | 46.7 (DS-V3) | **+2.8** |
| GPQA-Diamond (Avg@8) | **75.1** | 74.9 (Opus 4) | **+0.2** |

**LMSYS Arena (July 17, 2025):** Rank 1 open-source, Rank 5 overall (3,000+ user votes).

---

## 10. Safety Evaluation Protocol

### 10.1 Red-Teaming Framework

**Tool:** Promptfoo (automated adversarial prompt generation and response analysis)

### 10.2 Attack Surface

| Dimension | Categories |
|---|---|
| **Plugins** | Harmful (8 subcategories), Criminal (11), Misinformation (9), Privacy (5), Security (11) |
| **Strategies** | Basic, Base64, Prompt Injection, Iterative Jailbreak, Crescendo |

### 10.3 Test Configuration

- 3 attack prompts per plugin-strategy combination
- Dual-language (English + Chinese) where supported → 6 prompts per combination
- Multiple review rounds with consistent reviewer assignment per test set

### 10.4 Safety Results (Kimi K2 Pass Rates)

| Plugin | Basic | Base64 | Prompt Injection | Iterative Jailbreak | Crescendo |
|---|---|---|---|---|---|
| Harmful | 98.04 | 100 | 93.14 | 92.16 | 64.71 |
| Criminal | 100 | 96.97 | 75.76 | 57.57 | 56.06 |
| Misinformation | 97.28 | 98.48 | 98.39 | 63.97 | 85.71 |
| Privacy | 100 | 100 | 88.33 | 76.67 | 96.67 |
| Security | 77.84 | 82.93 | 87.80 | 43.90 | 68.29 |

**Key observations:**
- Base64 encoding has minimal impact on robustness (near-100% pass rates)
- Crescendo and Iterative Jailbreak are most effective adversarial strategies
- Complex attacks do not always outperform basic prompts (adversarial transformations can destroy original semantic intent)

---

## 11. Deployment Constraints

### 11.1 Memory Requirements

| Component | Precision | Memory |
|---|---|---|
| Model parameters | BF16 | ~2 TB |
| Gradient accumulation | FP32 | ~4 TB |
| Total (params + grads) | Mixed | ~6 TB across 256 GPUs |
| Per-GPU model states | — | ~30 GB |
| KV-cache (128K context) | MLA latent | Significantly reduced vs. standard MHA |

### 11.2 Inference Efficiency

- **64 attention heads:** Critical for long-context agentic inference efficiency
- **MLA:** Compressed KV-cache reduces memory bandwidth bottleneck
- **Sparsity 48:** Only 32.6B/1.04T parameters activated per token

### 11.3 Agentic Deployment

- Sandbox infrastructure: Kubernetes-powered, 10,000+ concurrent instances
- Partial rollout support for long-horizon tasks
- Heavy environments deployed as dedicated scalable services
- Unified Gym-like interface for environment integration

### 11.4 Fault Tolerance

- Checkpoint engine enables sub-30-second parameter synchronization
- Inference replicas restart independently (no cross-replica synchronization)
- Training workers collectively read checkpoint once (minimizes disk IO)
- Robust to single-point failures via distributed checkpoint architecture

---

## 12. Complexity Analysis

### 12.1 Forward Pass Complexity

Per token through one transformer layer:

**MLA Attention:**

$$\mathcal{O}(L \cdot d_{\text{model}} \cdot d_c + H \cdot L^2 \cdot d_k)$$

where $H = 64$ heads, $d_{\text{model}} = 7168$, $d_c$ = latent dimension, $L$ = sequence length.

**MoE FFN:**

$$\mathcal{O}(d_{\text{model}} \cdot N_{\text{experts\_total}} + K_{\text{active}} \cdot d_{\text{model}} \cdot d_{\text{expert}})$$

$$= \mathcal{O}(7168 \cdot 384 + 8 \cdot 7168 \cdot 2048)$$

**Total per token per layer:**

$$\mathcal{O}(d_{\text{model}} \cdot d_c + H \cdot L \cdot d_k + K \cdot d_{\text{model}} \cdot d_{\text{expert}})$$

**Full model:** Multiply by 61 layers.

### 12.2 Communication Complexity

**EP all-to-all per layer:**

$$\mathcal{O}\left(\frac{B \cdot L \cdot d_{\text{model}}}{EP} \cdot (EP - 1)\right)$$

with $EP = 16$.

**PP communication:** Overlapped except during warm-up, bounded by activation tensor sizes at stage boundaries.

### 12.3 Training FLOPs

For 15.5T tokens with 32.6B activated parameters:

$$\text{FLOPs}_{\text{total}} \approx 6 \times N_{\text{activated}} \times T = 6 \times 32.6 \times 10^9 \times 15.5 \times 10^{12} \approx 3.03 \times 10^{24}$$

(Approximate; actual value depends on attention sequence-length-dependent terms and gating overhead.)

---

## 13. Known Limitations and Failure Modes

| Failure Mode | Description | Mitigation Status |
|---|---|---|
| **Excessive token generation** | On hard reasoning tasks or unclear tool definitions, generates excessive tokens leading to truncation | Active work |
| **Incomplete tool calls** | Truncated outputs result in malformed tool invocations | Active work |
| **Unnecessary tool use** | Performance declines when tool use is enabled but unnecessary | Active work |
| **One-shot project generation** | Success rate for complete software projects via single prompt is suboptimal vs. agentic framework | Use agentic coding framework |
| **Crescendo attacks** | ~56–65% pass rate on criminal/harmful crescendo jailbreaks | Active work |
| **Iterative jailbreak** | ~44–92% pass rate depending on category | Active work |
| **Synthetic data risks** | Generalization across diverse domains, hallucination minimization, scalability remain open challenges | Active investigation |