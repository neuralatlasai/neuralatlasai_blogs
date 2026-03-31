# Llama 3 Post-Training: End-to-End Technical Report

---

## 1. System Overview and Stage Ordering

### 1.1 Formal Problem Definition

**Objective:** Given a pre-trained autoregressive language model $\pi_{\text{base}}$ with parameters $\theta_0 \in \mathbb{R}^d$ trained on corpus $\mathcal{D}_{\text{pre}}$, produce an aligned model $\pi_{\text{aligned}}$ that maximizes human utility $U(y \mid x)$ while preserving the base model's knowledge and calibration.

**Formal Optimization:**

$$\theta^* = \arg\max_{\theta} \; \mathbb{E}_{x \sim \mathcal{P}_{\text{user}}} \left[ U\big(\pi_\theta(\cdot \mid x)\big) \right] \quad \text{s.t.} \quad D_{\text{KL}}\!\left[\pi_\theta \| \pi_{\text{base}}\right] \leq \epsilon$$

**Boundary Conditions:**
- Base checkpoint: Llama 3 405B pre-trained model
- Context window: extended from 8K to 128K tokens during late pre-training
- Post-training defined as any model training outside pre-training scope

### 1.2 Deterministic Stage Ordering

The pipeline executes in $R = 6$ iterative rounds. Each round $r \in \{1, \ldots, 6\}$ follows:

$$\text{Round } r: \quad \underbrace{\text{Data Collection}}_{\text{Stage A}} \rightarrow \underbrace{\text{RM Training}}_{\text{Stage B}} \rightarrow \underbrace{\text{Rejection Sampling}}_{\text{Stage C}} \rightarrow \underbrace{\text{SFT}}_{\text{Stage D}} \rightarrow \underbrace{\text{DPO}}_{\text{Stage E}} \rightarrow \underbrace{\text{Model Averaging}}_{\text{Stage F}}$$

**Invariants across rounds:**
- Each subsequent round uses latest model from previous round for data generation
- Preference data accumulates across rounds for RM; DPO uses only most recent batches
- Prompt complexity increases monotonically with round index

<img src="assets/llama_3_405b_technical_blueprint_p06.png" alt="Iterative six-round post-training protocol with data collection, reward modeling, rejection sampling, supervised finetuning, direct preference optimization, and model averaging" width="100%" />

*Figure. High-level six-round post-training protocol, matching the deterministic stage ordering defined at the start of this report.*

---

## 2. Chat Dialog Format

### 2.1 Definition

A multi-message chat protocol enabling generation of multiple messages routed to different destinations (e.g., `user`, `ipython`) within a single dialog turn.

### 2.2 Token Taxonomy

| Token Class | Function | Examples |
|---|---|---|
| **Header tokens** | Indicate source and destination of each message | `<|start_header_id|>`, `<|end_header_id|>` |
| **Termination tokens** | Signal alternation between human and AI turns | `<|eot_id|>`, `<|eom_id|>` |
| **Role tokens** | Encode message role | `system`, `user`, `assistant`, `ipython` |

### 2.3 Design Rationale

- Supports tool use requiring intermediate messages to execution environments
- Enables multi-step planning with interleaved tool calls and reasoning
- Header/termination tokens are special tokens added to the vocabulary, not composed from existing subwords

### 2.4 Failure Modes

- **Tail repetition:** Model repeatedly generates content after termination context
- **Premature termination:** Model abruptly generates termination tokens mid-response
- Both failure modes are exacerbated when formatting tokens contribute to contrastive losses (addressed in DPO modifications, §5)

---

## 3. Reward Modeling

### 3.1 Definition

A reward model $R_\phi: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$ is trained on top of the pre-trained checkpoint $\theta_0$ to predict scalar quality scores for response $y$ given prompt $x$.

### 3.2 Training Objective

The Bradley-Terry pairwise preference model:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\!\left( R_\phi(x, y_w) - R_\phi(x, y_l) \right) \right]$$

where $y_w$ denotes the chosen (winning) response and $y_l$ the rejected (losing) response.

**Key difference from Llama 2:** The margin term $m(r)$ present in the Llama 2 RM loss is removed:

$$\underbrace{\mathcal{L}_{\text{RM}}^{\text{Llama2}} = -\log \sigma\!\left( R_\phi(x, y_w) - R_\phi(x, y_l) - m(r) \right)}_{\text{margin } m(r) \text{ removed in Llama 3}}$$

**Justification:** Diminishing improvements observed after data scaling; the margin term provided negligible benefit at scale.

### 3.3 Three-Way Ranking Extension

For samples with edited responses, the ranking is:

$$R_\phi(x, y_{\text{edited}}) > R_\phi(x, y_{\text{chosen}}) > R_\phi(x, y_{\text{rejected}})$$

This yields three pairwise comparisons per sample:

$$\mathcal{L}_{\text{RM}}^{3\text{-way}} = -\sum_{(i,j) \in \{(e,c), (e,r), (c,r)\}} \log \sigma\!\left( R_\phi(x, y_i) - R_\phi(x, y_j) \right)$$

### 3.4 Training Data Construction

**Input format:** Prompt and multiple responses concatenated into a single row with responses randomly shuffled.

**Formal representation:**

$$\text{input}_k = [x \; \| \; \text{shuffle}(y_1, y_2, \ldots, y_m)]$$

where $m \in \{2, 3\}$ depending on whether an edited response exists.

**Filtering criteria:**
- Discard samples where chosen and rejected responses are rated as similar
- Use only samples labeled as "significantly better" or "better"
- All available preference data across rounds used for RM training

### 3.5 Complexity Analysis

For $N$ preference samples, each with average sequence length $L$:

$$\text{Training FLOPs} \approx 6 \cdot N \cdot L \cdot |\phi|$$

where $|\phi|$ denotes RM parameter count (initialized from 405B checkpoint).

### 3.6 Pseudo-Algorithm: Reward Model Training

```
ALGORITHM: RewardModelTraining
INPUT: Pre-trained checkpoint θ₀, preference dataset D_pref
OUTPUT: Reward model R_φ

1. Initialize φ ← θ₀
2. Filter D_pref: remove samples with similar chosen/rejected responses
3. Filter D_pref: retain only "significantly better" or "better" labels
4. FOR each epoch:
   5. FOR each batch B ⊂ D_pref:
      6. FOR each sample (x, y₁, ..., yₘ) ∈ B:
         7. Concatenate: input ← [x ∥ shuffle(y₁, ..., yₘ)]
         8. Compute R_φ(x, yᵢ) for all i
         9. Compute pairwise losses for all valid (i,j) pairs
      10. Aggregate loss L_RM over batch
      11. Update φ ← φ - η∇_φ L_RM
12. RETURN R_φ
```

### 3.7 Failure Modes

- **Score collapse:** All responses receive near-identical scores when training data lacks diversity
- **Reward hacking potential:** Downstream rejection sampling or RL may exploit RM artifacts
- **Calibration drift:** RM scores may not remain well-calibrated across iterative rounds as response distribution shifts

---

## 4. Supervised Finetuning (SFT)

### 4.1 Definition

Supervised finetuning adapts the pre-trained model $\pi_{\theta_0}$ using demonstration data $\mathcal{D}_{\text{SFT}} = \{(x_i, y_i)\}_{i=1}^N$ where $y_i$ are target responses (human-annotated, rejection-sampled, or synthetic).

### 4.2 Loss Formulation

Standard autoregressive cross-entropy loss on target tokens with prompt masking:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[ \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t}) \right]$$

**Prompt masking:** Loss is computed only over response tokens $y$; prompt tokens $x$ are masked:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|y|} \sum_{t=|x|+1}^{|x|+|y|} \log \pi_\theta(w_t \mid w_{<t})$$

where $w = [x \| y]$ is the full concatenated sequence.

### 4.3 Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | $1 \times 10^{-5}$ |
| Training steps | 8,500 – 9,000 |
| Model | Llama 3 405B |

### 4.4 Data Sources

The SFT data mix consists of three primary sources:

1. **Rejection-sampled responses** from human annotation prompts (§4.4.1)
2. **Synthetic data** targeting specific capabilities (§8)
3. **Human-curated data** in small quantities

### 4.4.1 Rejection Sampling (RS)

**Definition:** For each prompt $x$, sample $K$ candidate responses from the current best policy $\pi_{\theta_{\text{best}}}$ and select the response with the highest reward:

$$y^* = \arg\max_{y \in \{y^{(1)}, \ldots, y^{(K)}\}} R_\phi(x, y)$$

where $y^{(k)} \sim \pi_{\theta_{\text{best}}}(\cdot \mid x)$ for $k = 1, \ldots, K$ and typically $K \in [10, 30]$.

**System prompt steering:** In later rounds, system prompts are prepended to steer RS responses toward desirable tone, style, and formatting:

$$y^{(k)} \sim \pi_{\theta_{\text{best}}}(\cdot \mid [s_{\text{sys}} \| x])$$

where $s_{\text{sys}}$ is a capability-specific system prompt.

**Computational Optimization via PagedAttention:**

- Dynamic KV-cache allocation eliminates pre-allocated memory waste
- KV-cache pages for prompt are shared across all $K$ output candidates
- Maximum output length pre-defined; request issued only if sufficient memory available
- **Throughput improvement:** $> 2\times$ during rejection sampling
- Eliminates swap-out overhead by capacity-checking before request admission

### 4.4.2 SFT Data Statistics

| Category | % of Examples | Avg. Turns | Avg. Tokens | Avg. Context Tokens | Avg. Response Tokens |
|---|---|---|---|---|---|
| General English | 52.66% | 6.3 | 974.0 | 656.7 | 317.1 |
| Code | 14.89% | 2.7 | 753.3 | 378.8 | 374.5 |
| Multilingual | 3.01% | 2.7 | 520.5 | 230.8 | 289.7 |
| Exam-like | 8.14% | 2.3 | 297.8 | 124.4 | 173.4 |
| Reasoning & Tools | 21.19% | 3.1 | 661.6 | 359.8 | 301.9 |
| Long Context | 0.11% | 6.7 | 38,135.6 | 37,395.2 | 740.5 |
| **Total** | **100%** | **4.7** | **846.1** | **535.7** | **310.4** |

### 4.5 Pseudo-Algorithm: SFT Stage

```
ALGORITHM: SupervisedFinetuning
INPUT: Pre-trained checkpoint θ₀, SFT dataset D_SFT, reward model R_φ
OUTPUT: SFT model θ_SFT

1. Construct D_SFT:
   2. FOR each prompt x in human annotation set:
      3. Sample K responses: {y⁽¹⁾, ..., y⁽ᴷ⁾} ~ π_θ_best(· | [s_sys ∥ x])
         using PagedAttention with shared KV-cache
      4. Select y* = argmax_k R_φ(x, y⁽ᵏ⁾)
      5. Add (x, y*) to D_RS
   6. Merge D_RS with synthetic data D_synth and human-curated D_human
   7. Apply data processing pipeline (§6)
   8. Construct final mix with per-source epoch multipliers

9. Initialize θ ← θ₀
10. FOR step = 1 to 9000:
    11. Sample batch B ~ D_SFT
    12. Compute L_SFT = -(1/|B|) Σ_{(x,y)∈B} Σ_t log π_θ(yₜ | x, y_{<t})
    13. Update θ ← θ - η∇_θ L_SFT    [η = 1e-5]
14. RETURN θ_SFT
```

### 4.6 Convergence Dynamics

- Learning rate $\eta = 10^{-5}$ is intentionally low to prevent catastrophic forgetting of pre-trained knowledge
- 8.5K–9K steps found robust across different rounds and data mixes
- Final data mix epochs multiple times on high-quality sources and downsamples others, creating non-uniform sampling weights

### 4.7 Failure Modes

- **Catastrophic forgetting:** Excessive SFT steps or high learning rate degrades pre-training knowledge
- **Distribution mismatch:** SFT on model-generated data introduces feedback loops if quality control is insufficient
- **Length bias:** Rejection sampling may favor longer responses if RM exhibits length correlation

---

## 5. Direct Preference Optimization (DPO)

### 5.1 Definition

DPO directly optimizes the policy $\pi_\theta$ to satisfy human preferences without explicitly training a reward model at this stage, by reparameterizing the reward function as the log-ratio of the policy to a reference model.

### 5.2 Standard DPO Loss

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

where:
- $\pi_{\text{ref}}$ is the reference policy (SFT model)
- $\beta = 0.1$ controls the strength of the KL constraint
- $\sigma(\cdot)$ is the sigmoid function

**Implicit reward:**

$$r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

### 5.3 Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | $1 \times 10^{-5}$ |
| $\beta$ (KL penalty coefficient) | $0.1$ |
| NLL regularization coefficient | $0.2$ |
| Training data | Most recent preference batches only |

### 5.4 Algorithmic Modifications

#### 5.4.1 Formatting Token Masking

**Definition:** Special formatting tokens (header tokens, termination tokens) are masked from both chosen and rejected responses in the DPO loss computation.

Let $\mathcal{M} \subset \{1, \ldots, T\}$ be the set of positions corresponding to formatting tokens. The modified per-token log-probability is:

$$\log \pi_\theta^{\text{masked}}(y \mid x) = \sum_{t \notin \mathcal{M}} \log \pi_\theta(y_t \mid x, y_{<t})$$

**Modified DPO loss:**

$$\mathcal{L}_{\text{DPO}}^{\text{masked}}(\theta) = -\mathbb{E}\!\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta^{\text{masked}}(y_w \mid x)}{\pi_{\text{ref}}^{\text{masked}}(y_w \mid x)} - \beta \log \frac{\pi_\theta^{\text{masked}}(y_l \mid x)}{\pi_{\text{ref}}^{\text{masked}}(y_l \mid x)} \right) \right]$$

**Rationale:** The contrastive nature of DPO creates a conflicting learning objective for tokens appearing identically in both chosen and rejected responses. The model simultaneously receives gradients to:
- Increase log-likelihood of formatting tokens in $y_w$
- Decrease log-likelihood of identical formatting tokens in $y_l$

This conflict causes:
- Tail repetition (model continues generating after expected termination)
- Abrupt termination (model prematurely generates stop tokens)

#### 5.4.2 NLL Regularization

An additional negative log-likelihood loss on chosen responses is added:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{DPO}}^{\text{masked}}(\theta) + \alpha \cdot \mathcal{L}_{\text{NLL}}(\theta)$$

where:

$$\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x, y_w) \sim \mathcal{D}_{\text{pref}}} \left[ \sum_{t=1}^{|y_w|} \log \pi_\theta(y_{w,t} \mid x, y_{w,<t}) \right]$$

and $\alpha = 0.2$.

**Benefits:**
- Stabilizes DPO training by maintaining desired formatting for generation
- Prevents the decrease of log probability of chosen responses (a known DPO failure mode where both chosen and rejected log-probs decrease, with rejected decreasing faster)
- Acts as an anchor preventing excessive divergence from the data distribution

### 5.5 DPO Data Selection

**Key principle:** Only the most recent batches of preference data from each capability domain are used for DPO training.

**Rationale:** This ensures training data conforms to the distribution of the current policy being optimized, reducing:
- Off-policy distribution mismatch
- Stale preference artifacts from earlier, weaker models

**Filtering:** Same as RM — retain only "significantly better" or "better" labeled pairs; discard "similar" responses.

### 5.6 Comparison with PPO

| Criterion | DPO | PPO |
|---|---|---|
| Compute requirement | Lower for large-scale models | Higher (requires value model + rollouts) |
| IFEval performance | **Superior** | Inferior |
| Implementation complexity | Lower | Higher |
| On-policy data | Not required | Required |

DPO was preferred over PPO for Llama 3 405B due to lower compute cost and stronger performance on instruction following benchmarks.

### 5.7 Gradient Analysis

The DPO gradient with respect to $\theta$ for a single sample $(x, y_w, y_l)$:

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \sigma\!\left(-\hat{r}_\theta\right) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

where:

$$\hat{r}_\theta = \beta \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)$$

**Interpretation:**
- $\sigma(-\hat{r}_\theta)$: weighting factor that downweights already-learned preferences
- Gradient increases $\log \pi_\theta(y_w)$ and decreases $\log \pi_\theta(y_l)$
- When formatting tokens are not masked, gradients on shared tokens partially cancel, leading to noisy updates and the observed instabilities

### 5.8 Pseudo-Algorithm: DPO Stage

```
ALGORITHM: DirectPreferenceOptimization
INPUT: SFT checkpoint θ_SFT, recent preference data D_pref_recent
OUTPUT: DPO-aligned model θ_DPO

1. Set π_ref ← θ_SFT (frozen reference)
2. Initialize θ ← θ_SFT
3. Filter D_pref_recent:
   4. Retain samples with label ∈ {"significantly better", "better"}
   5. Discard samples with similar responses
6. Identify formatting token positions M for each sample
7. FOR each training step:
   8. Sample batch B ~ D_pref_recent
   9. FOR each (x, y_w, y_l) ∈ B:
      10. Compute log π_θ^masked(y_w | x) = Σ_{t∉M} log π_θ(y_{w,t} | x, y_{w,<t})
      11. Compute log π_ref^masked(y_w | x) similarly
      12. Compute log π_θ^masked(y_l | x) similarly
      13. Compute log π_ref^masked(y_l | x) similarly
      14. r̂ = β(log π_θ^masked(y_w)/π_ref^masked(y_w) - log π_θ^masked(y_l)/π_ref^masked(y_l))
      15. L_DPO = -log σ(r̂)
      16. L_NLL = -Σ_t log π_θ(y_{w,t} | x, y_{w,<t})
      17. L_total = L_DPO + 0.2 · L_NLL
   18. Aggregate L_total over batch
   19. Update θ ← θ - η∇_θ L_total    [η = 1e-5]
20. RETURN θ_DPO
```

### 5.9 Long Context DPO

**Critical finding:** Using only short-context training data in DPO does **not** negatively impact long-context performance, provided the SFT model has high-quality long-context capabilities.

**Hypothesis:** DPO uses fewer optimizer steps than SFT, insufficient to degrade the long-context representations established during SFT.

**Operational decision:** Standard short-context DPO recipe applied on top of long-context SFT checkpoints without modification.

### 5.10 Failure Modes

- **Chosen log-prob decrease:** Without NLL regularization, both chosen and rejected log-probs decrease; the NLL term with $\alpha = 0.2$ mitigates this
- **Reward hacking:** Over-optimization can exploit distributional artifacts in preference data
- **Formatting collapse:** Without formatting token masking, model learns contradictory objectives on shared tokens
- **Distribution shift:** Using stale preference data from earlier rounds leads to off-policy optimization degradation

---

## 6. Data Processing and Quality Control Pipeline

### 6.1 Overview

Given that most training data is model-generated, rigorous cleaning and quality control is required.

### 6.2 Data Cleaning (Rule-Based)

**Objective:** Remove or modify samples exhibiting undesirable patterns.

| Pattern | Mitigation |
|---|---|
| Excessive emoji usage | Rule-based filtering/removal |
| Excessive exclamation points | Frequency-based thresholding |
| Overly apologetic tone | Identify overused phrases ("I'm sorry", "I apologize"); balance proportion in dataset |

### 6.3 Data Pruning (Model-Based)

Four-stage model-based quality pipeline:

#### 6.3.1 Topic Classification

**Method:** Finetune Llama 3 8B as a topic classifier.

- **Coarse-grained buckets:** e.g., "mathematical reasoning"
- **Fine-grained buckets:** e.g., "geometry and trigonometry"
- Applied to all SFT data via inference

#### 6.3.2 Quality Scoring

Two independent scoring signals combined:

**Signal 1 — Reward Model Score:**

$$q_{\text{RM}}(x, y) = \mathbb{1}\!\left[R_\phi(x, y) \geq \text{P75}\!\left(\{R_\phi(x_i, y_i)\}_{i=1}^N\right)\right]$$

Samples in the top quartile of RM scores are labeled "high quality."

**Signal 2 — Llama-Based Score:**

For general English data, Llama 3 rates each sample on three criteria:
1. Accuracy
2. Instruction following
3. Tone/presentation

Each on a three-point scale. Maximum score = high quality.

For coding data, two criteria:
1. Bug identification
2. User intention

Each on a two-point scale. Maximum score = high quality.

**Combination rule:**

$$q_{\text{combined}}(x, y) = q_{\text{RM}}(x, y) \lor q_{\text{Llama}}(x, y)$$

**Justification:** RM and Llama-based scores exhibit high disagreement rates; the union (OR) yields best recall on internal test sets.

#### 6.3.3 Difficulty Scoring

Two measures:

**Measure 1 — InsTaG (Intention Tagging):**

Prompt Llama 3 70B to perform intention tagging of SFT prompts:

$$d_{\text{InsTaG}}(x) = |\text{intentions}(x)|$$

More intentions $\Rightarrow$ higher complexity.

**Measure 2 — Llama-Based Difficulty:**

Prompt Llama 3 to rate dialog difficulty on a three-point scale:

$$d_{\text{Llama}}(x, y) \in \{1, 2, 3\}$$

#### 6.3.4 Semantic Deduplication

**Step 1:** Cluster complete dialogs using RoBERTa embeddings.

$$\mathbf{e}_i = \text{RoBERTa}(x_i, y_i) \in \mathbb{R}^h$$

**Step 2:** Within each cluster, sort by composite score:

$$s_i = q_{\text{combined}}(x_i, y_i) \times d(x_i, y_i)$$

**Step 3:** Greedy selection with cosine similarity threshold $\tau$:

```
ALGORITHM: SemanticDeduplication
INPUT: Cluster C = {(x_i, y_i)}, threshold τ
OUTPUT: Deduplicated subset S

1. Sort C by s_i in descending order
2. Initialize S ← ∅
3. FOR each (x_i, y_i) in sorted C:
   4. Compute max_sim = max_{(x_j, y_j) ∈ S} cos(e_i, e_j)
   5. IF max_sim < τ:
      6. S ← S ∪ {(x_i, y_i)}
7. RETURN S
```

### 6.4 Complete Data Processing Pseudo-Algorithm

```
ALGORITHM: DataProcessingPipeline
INPUT: Raw SFT data D_raw
OUTPUT: Cleaned, pruned, deduplicated dataset D_final

1. RULE-BASED CLEANING:
   2. Remove samples with emoji density > threshold
   3. Remove samples with exclamation point density > threshold
   4. Balance apologetic phrase frequency to target proportion

5. TOPIC CLASSIFICATION:
   6. Run Llama 3 8B classifier on all samples
   7. Assign coarse and fine-grained topic labels

8. QUALITY SCORING:
   9. Compute R_φ(x, y) for all samples
   10. Compute Llama-based quality scores for all samples
   11. Label as high_quality if RM top-quartile OR Llama max-score

12. DIFFICULTY SCORING:
   13. Compute InsTaG scores via Llama 3 70B
   14. Compute Llama-based difficulty scores

15. SEMANTIC DEDUPLICATION:
   16. Compute RoBERTa embeddings for all samples
   17. Cluster using cosine similarity
   18. Within each cluster, sort by quality × difficulty
   19. Greedy select with cosine threshold τ

20. CONSTRUCT FINAL MIX:
   21. Adjust per-category proportions based on benchmark targeting
   22. Apply per-source epoch multipliers (some sources epoched multiple times)
   23. Downsample overrepresented categories

24. RETURN D_final
```

---

## 7. Model Averaging

### 7.1 Definition

At each stage (RM, SFT, DPO), multiple experiments are run with different data versions and hyperparameters. The final model is obtained via parameter averaging:

$$\theta_{\text{avg}} = \frac{1}{M} \sum_{m=1}^{M} \theta^{(m)}$$

where $\theta^{(m)}$ are parameters from the $m$-th experiment variant.

### 7.2 Theoretical Justification

Based on:
- **Stochastic Weight Averaging (SWA):** Averaging weights from different points in the loss landscape tends to find flatter minima with better generalization
- **Model soups:** Averaging fine-tuned models with different hyperparameters often improves over individual models without additional compute at inference time

### 7.3 Invariants

- All averaged models share the same base architecture (Llama 3 405B)
- Averaging is performed in parameter space, not prediction space
- Applied independently at RM, SFT, and DPO stages

### 7.4 Failure Modes

- **Weight interference:** If models are in different loss basins, averaging may produce a model in a high-loss region
- **Mitigated by:** Using models from similar training trajectories with overlapping data distributions

---

## 8. Capability-Specific Data Pipelines

### 8.1 Code

#### 8.1.1 Code Expert Training

**Pipeline:**

$$\pi_{\text{base}} \xrightarrow{\text{branch}} \pi_{\text{code-CPT}} \xrightarrow{\text{LCFT}} \pi_{\text{code-16K}} \xrightarrow{\text{SFT+DPO}} \pi_{\text{code-expert}}$$

- **Continued pre-training:** 1T tokens, >85% code data
- **Long-context finetuning (LCFT):** Last several thousand steps; extends context to 16K on repo-level code data
- **Post-training:** SFT and DPO with code-focused data mixes
- **Usage:** Human annotation collection + rejection sampling for code prompts

#### 8.1.2 Synthetic Code Data Generation

Three approaches generating >2.7M synthetic SFT examples:

**Approach 1: Execution Feedback**

~1M synthetic coding dialogues:

```
ALGORITHM: CodeSynthesis_ExecutionFeedback
INPUT: Code snippet corpus, Llama 3
OUTPUT: Verified (prompt, solution) pairs D_code

1. PROBLEM GENERATION:
   2. Sample random code snippets from diverse sources
   3. Prompt model to generate programming problems inspired by snippets
   4. Ensure long-tail topic coverage

5. SOLUTION GENERATION:
   6. FOR each problem p in target language:
      7. Prompt Llama 3 with p + general programming rules
      8. Require model to explain thought process in comments
      9. Obtain candidate solution s

10. CORRECTNESS VERIFICATION:
    11. STATIC ANALYSIS:
        12. Parse s through parser and linter
        13. Check: syntax errors, uninitialized variables, non-imported functions,
            code style, typing errors
    14. DYNAMIC ANALYSIS:
        15. Prompt model to generate unit tests for (p, s)
        16. Execute s + unit tests in containerized environment
        17. Capture stdout, stderr, return code

18. ITERATIVE SELF-CORRECTION:
    19. IF s fails any check:
        20. Construct feedback prompt: [p ∥ s ∥ error_feedback]
        21. Prompt model to revise solution
        22. Model may either fix code OR modify unit tests
        23. Re-run verification loop
        24. Maximum correction attempts: bounded

25. FILTERING:
    26. Include only dialogs passing all checks in D_code
    27. ~20% of solutions initially incorrect but self-corrected

28. ITERATIVE IMPROVEMENT:
    29. Finetune model on D_code
    30. Use improved model for next round of synthesis

31. RETURN D_code
```

**Key finding:** Training 405B on its own generated data without execution feedback degrades performance. Execution feedback as a source of truth is essential for self-improvement at the frontier scale.

**Approach 2: Programming Language Translation**

Addresses performance gap between common languages (Python, C++) and less common languages (TypeScript, PHP):

```
ALGORITHM: CodeTranslation
INPUT: Source language programs D_src, target languages L_target
OUTPUT: Translated programs D_trans

1. FOR each program p_src ∈ D_src:
   2. FOR each target language l ∈ L_target:
      3. Prompt Llama 3 to translate p_src to language l
      4. VERIFY via:
         5. Syntax parsing in target language
         6. Compilation (if applicable)
         7. Execution with equivalent test cases
      8. IF verification passes:
         9. Add (p_src, p_trans_l) to D_trans
10. RETURN D_trans
```

**Approach 3: Backtranslation**

~1.2M synthetic dialogs for code explanation, documentation, debugging:

```
ALGORITHM: CodeBacktranslation
INPUT: Code snippets C from pre-training data
OUTPUT: Verified synthetic dialogs D_bt

1. FOR each code snippet c ∈ C:
   2. GENERATE: Prompt Llama 3 for target capability
      (e.g., add documentation/comments, explain code)
      → produces auxiliary artifact a
   3. BACKTRANSLATE: Prompt Llama 3 to reconstruct code from a
      → produces c'
   4. FILTER: Prompt Llama 3 to judge faithfulness of c' to c
      → produces quality score q
   5. IF q ≥ threshold:
      6. Add (generate_prompt, a, backtranslate_prompt, c') to D_bt
7. RETURN D_bt (highest self-verification scores selected)
```

#### 8.1.3 Rejection Sampling System Prompt Steering for Code

Code-specific system prompts improve:
- Code readability
- Documentation quality
- Thoroughness and specificity
- Informative variable naming
- Memory efficiency

#### 8.1.4 Code Data Quality Filtering

**Challenge:** Rejection-sampled responses mix natural language and code; code may not always be executable (e.g., pseudocode, code snippets).

**Solution: Model-as-judge**

$$\text{score}(x, y) = \text{correctness\_score}(y) + \text{style\_score}(y) \in \{0, 1, 2\}$$

- Retain only samples with score = 2
- **Initial regression:** Stringent filtering disproportionately removed hard examples
- **Fix:** Strategically revise responses for the hardest coding problems until they meet model-as-judge criteria
- Final dataset achieves balance between quality and difficulty

### 8.2 Multilinguality

#### 8.2.1 Multilingual Expert

$$\pi_{\text{base}} \xrightarrow{\text{branch + CPT}} \pi_{\text{multilingual}} \xrightarrow{\text{post-training}} \pi_{\text{ml-expert}}$$

- Continued pre-training data mix: 90% multilingual tokens
- Used for higher-quality annotations in non-English languages until full pre-training completion

#### 8.2.2 Multilingual SFT Data Composition

| Source | Proportion |
|---|---|
| Human annotations | 2.4% |
| Data from other NLP tasks | 44.2% |
| Rejection sampled data | 18.8% |
| Translated reasoning data | 34.6% |

**Target languages:** German, French, Italian, Portuguese, Hindi, Spanish, Thai

**Detailed source descriptions:**

1. **Human annotations:** Open-ended prompts from linguists and native speakers representing real-world use cases

2. **NLP task data:** Rewritten into dialog format from:
   - Exams-QA, Conic10k
   - Parallel texts from GlobalVoices, Wikimedia
   - Filtered using LID and Blaser2.0 for quality
   - Parallel text reformatted using multilingual templates simulating real-life conversation scenarios (translation, language learning)

3. **Rejection sampled data:**
   - Temperature: explored random $T \in [0.2, 1.0]$ in early rounds; fixed $T = 0.6$ in final round
   - High temperature: creative but susceptible to unnatural code-switching
   - Specialized system prompts for format, structure, readability
   - Multilingual-specific pre-selection checks: language-match rate between prompt and response (e.g., romanized Hindi prompt should not receive Devanagari response)

4. **Translated reasoning data:**
   - Exception to general policy of avoiding machine translation (to prevent translationese, name bias, gender bias, cultural bias)
   - Only synthetic quantitative reasoning data translated
   - Simple mathematical language → minimal translation quality issues
   - Strong gains on MGSM benchmark

### 8.3 Math and Reasoning

#### 8.3.1 Challenge Formalization

Five core challenges:

| Challenge | Formal Description |
|---|---|
| Lack of prompts | $|\mathcal{D}_{\text{prompt}}|$ decreases as problem complexity $c$ increases: $|\mathcal{D}_{\text{prompt}}| \propto c^{-\alpha}$ |
| Lack of ground truth CoT | Step-by-step solutions $(s_1, \ldots, s_n)$ unavailable for most complex problems |
| Incorrect intermediate steps | $\exists\, t: s_t$ incorrect even when final answer $a$ correct (spurious correctness) |
| Tool integration | Need to interleave natural language reasoning $r_t$ with code execution $c_t$ |
| Train-inference discrepancy | SFT training assumes teacher-forcing; inference involves interactive feedback |

#### 8.3.2 Methodologies

**Method 1: Prompt Sourcing**

- Extract mathematical content from pre-training data → convert to QA format
- Build taxonomy of mathematical skills
- Identify underperforming skills → source human prompts for those skills

**Method 2: Step-wise Reasoning Trace Augmentation**

For prompt set $\mathcal{P}$:

$$\forall x \in \mathcal{P}: \quad \{(s_1^{(k)}, \ldots, s_{n_k}^{(k)}, a^{(k)})\}_{k=1}^{K} \sim \pi_\theta(\cdot \mid x)$$

Filter based on answer correctness:

$$\mathcal{D}_{\text{valid}} = \left\{(x, s_{1:n}^{(k)}) \;\middle|\; a^{(k)} = a^* \right\}$$

Self-verification via Llama 3: prompt model to verify whether each step-by-step solution is valid for the given question.

**Method 3: Reward Model Filtering**

- **Outcome Reward Model (ORM):** $R_{\text{ORM}}(x, s_{1:n}, a) \in \mathbb{R}$ scores complete solutions
- **Stepwise Reward Model (PRM):** $R_{\text{PRM}}(x, s_{1:t}) \in \mathbb{R}$ scores individual reasoning steps

$$\mathcal{D}_{\text{filtered}} = \left\{(x, s_{1:n}) \;\middle|\; \forall t: R_{\text{PRM}}(x, s_{1:t}) \geq \tau_{\text{step}} \right\}$$

**Monte Carlo Tree Search (MCTS):** For the most challenging prompts, MCTS with learned stepwise reward models generates valid reasoning traces:

$$\text{MCTS}(x, R_{\text{PRM}}) \rightarrow s_{1:n}^* = \arg\max_{s_{1:n}} \sum_{t=1}^{n} R_{\text{PRM}}(x, s_{1:t})$$

**Method 4: Interleaved Code and Text Reasoning**

$$\text{Reasoning trace} = (r_1, c_1, o_1, r_2, c_2, o_2, \ldots, r_m, a)$$

where $r_t$ = textual reasoning, $c_t$ = Python code, $o_t$ = execution output.

Code execution serves as a feedback signal to eliminate invalid reasoning chains.

**Method 5: Learning from Feedback and Mistakes**

```
ALGORITHM: ErrorCorrectionTraining
INPUT: Prompt set P, model π_θ
OUTPUT: Error-correction training data D_ec

1. FOR each x ∈ P:
   2. Generate K solutions: {(s₁:n⁽ᵏ⁾, a⁽ᵏ⁾)}
   3. Identify incorrect generations: I = {k : a⁽ᵏ⁾ ≠ a*}
   4. FOR each k ∈ I:
      5. Construct error-correction prompt:
         [x ∥ s₁:n⁽ᵏ⁾ ∥ "The above solution is incorrect. Please correct it."]
      6. Generate corrected solution s'₁:n'
      7. IF a' = a*:
         8. Add (error_correction_prompt, s'₁:n') to D_ec
9. RETURN D_ec
```

### 8.4 Long Context

#### 8.4.1 Problem Statement

Pre-training extends context from 8K to 128K tokens. SFT must preserve this capability while maintaining short-context performance.

**Critical finding:** Naive SFT with only short-context data causes significant regression in long-context capabilities.

#### 8.4.2 Synthetic Long-Context Data Generation

Three categories:

**Category 1: Question Answering**

```
ALGORITHM: LongContextQA
INPUT: Long documents D_long from pre-training, chunk size 8K
OUTPUT: Long-context QA pairs

1. FOR each document d ∈ D_long:
   2. Split d into chunks: [c₁, c₂, ..., cₙ] of 8K tokens each
   3. Randomly select subset of chunks
   4. Prompt earlier Llama 3 to generate QA pairs from selected chunks
   5. Training sample: (full document d as context, question, answer)
```

**Category 2: Hierarchical Summarization**

```
ALGORITHM: HierarchicalSummarization
INPUT: Long documents D_long
OUTPUT: Summarization training data

1. FOR each document d:
   2. Split into 8K chunks
   3. Summarize each chunk independently using best 8K-context Llama 3
   4. Summarize the chunk summaries to produce global summary
   5. Training sample: (full document, "Summarize preserving all important details", summary)
   6. Generate QA pairs from global summary requiring whole-document understanding
```

**Category 3: Long Context Code Reasoning**

```
ALGORITHM: LongContextCodeReasoning
INPUT: Python repositories
OUTPUT: Code reasoning training data

1. Parse Python files for import statements and dependencies
2. Identify key files referenced by ≥ 5 other files
3. Remove one key file from repository
4. Training sample: (repository - key file, 
   "Identify dependent files and generate missing code", 
   expected output)
```

#### 8.4.3 Length Categorization

Synthetic samples categorized by sequence length:
- 16K, 32K, 64K, 128K tokens
- Enables fine-grained targeting of input lengths in the data mix

#### 8.4.4 Mixing Ratio

**Optimal ratio found through ablation:**

$$\text{Long-context data fraction} = 0.1\% \text{ of total SFT data}$$

This ratio optimizes performance across both short-context and long-context benchmarks.

#### 8.4.5 DPO for Long Context

**Finding:** Short-context DPO on top of long-context SFT checkpoints does not degrade long-context performance.

**Hypothesis:** DPO has fewer optimizer steps than SFT, insufficient to override long-context representations.

**Decision:** No modification to standard short-context DPO recipe.

### 8.5 Tool Use

#### 8.5.1 Tool Taxonomy

| Tool | Purpose | Interface |
|---|---|---|
| Brave Search | Recent events, factual retrieval beyond knowledge cutoff | API call |
| Python interpreter | Complex computation, file processing, data analysis, visualization | Code execution |
| Wolfram Alpha | Precise math/science computation, database retrieval | API call |

#### 8.5.2 Implementation Architecture

- Core tools implemented as Python objects with methods
- Zero-shot tools: Python functions with descriptions and docstrings
- Function definitions and calls convertible to JSON format for web API calls
- All tool calls executed by Python interpreter (must be enabled in system prompt)
- Core tools individually enable/disable via system prompt

#### 8.5.3 Annotation Differences from Standard Pipeline

| Aspect | Standard Pipeline | Tool Use Pipeline |
|---|---|---|
| Annotation granularity | Turn-level | **Message-level** (multiple assistant messages per turn) |
| Rejection sampling | Applied | **Not applied** (no gains observed) |
| Annotation scope | Response quality | Tool call quality + reasoning about tool output |
| Tool output editing | N/A | Not permitted (annotators cannot edit tool outputs) |

#### 8.5.4 Progressive Annotation Protocol

$$\text{Round 1: single-turn tool use} \rightarrow \text{Round 2: tool use in dialogs} \rightarrow \text{Round 3: multi-step tool use + data analysis}$$

Bootstrapped via synthetic data from previous Llama 3 checkpoints to reduce annotator editing burden.

#### 8.5.5 Tool Dataset Construction

**Single-Step Tool Use:**

```
ALGORITHM: SingleStepToolData
INPUT: Core tools T, knowledge cutoff date
OUTPUT: Single-step tool use trajectories

1. Few-shot generate user prompts requiring tool call
   (e.g., questions exceeding knowledge cutoff)
2. Few-shot generate appropriate tool calls for each prompt
3. Execute tool calls → obtain tool outputs
4. Prompt model to generate final answer given tool output
5. Trajectory: [system_prompt, user_prompt, tool_call, tool_output, final_answer]
6. Filter ~30% for execution failures or formatting issues
```

**Multi-Step Tool Use:**

```
ALGORITHM: MultiStepToolData
INPUT: Core tools T
OUTPUT: Multi-step tool use trajectories

1. Prompt Llama 3 to generate user prompts requiring ≥ 2 tool calls
   (same or different tools from core set)
2. Few-shot prompt Llama 3 to generate ReAct-style solution:
   [reasoning₁, tool_call₁, tool_output₁, reasoning₂, tool_call₂, ...]
3. Execute all tool calls
4. Validate trajectory consistency
```

**File Upload Processing:**

Supported file types: `.txt`, `.docx`, `.pdf`, `.pptx`, `.xlsx`, `.csv`, `.tsv`, `.py`, `.json`, `.jsonl`, `.html`, `.xml`

Tasks: summarization, bug finding/fixing, code optimization, data analysis, visualization.

**Negative Examples for Tool Restraint:**

- Add easy math/QA queries (from WebQuestions, MAWPS, TriviaQA, AQuA) with responses **without** tool calls but **with** tools activated in system prompt
- Teaches model to avoid unnecessary tool calls for simple queries

#### 8.5.6 Zero-Shot Tool Use (Function Calling)

**Training data construction:**

```
ALGORITHM: ZeroShotToolData
INPUT: The Stack code corpus
OUTPUT: (function_definition, user_query, function_call) tuples

1. Mine The Stack for function calls and their definitions
2. Clean and filter:
   - Remove functions with missing docstrings
   - Remove non-executable functions
3. Prompt Llama 3 to generate natural language query
   corresponding to each function call
4. Types of calls:
   a. Simple: single function call
   b. Nested: function call as argument of another function
   c. Parallel: list of independent function calls

5. Multi-turn function calling:
   6. Use multiple Llama 3 agents with different roles:
      - Domain generator agent
      - API generator agent
      - User query generator agent
      - API call generator agent
      - Response generator agent
   7. Agents collaborate step-by-step
   8. Ensure diverse domain and realistic API coverage
```

### 8.6 Factuality

#### 8.6.1 Core Principle

**Alignment principle:** Post-training should align the model to "know what it knows" rather than add knowledge.

#### 8.6.2 Knowledge Probing Pipeline

```
ALGORITHM: KnowledgeProbing
INPUT: Pre-training data D_pre, Llama 3 model π
OUTPUT: Factuality training data D_fact

1. EXTRACT: Sample data snippet d from D_pre (context)
2. GENERATE QUESTION: Prompt π to generate factual question q about d
3. SAMPLE RESPONSES: Generate N responses {r₁, ..., rₙ} ~ π(· | q)
4. SCORE CORRECTNESS: For each rᵢ, use d as reference and π as judge
   → correctness(rᵢ) ∈ {correct, incorrect}
5. SCORE INFORMATIVENESS: Use π as judge
   → informativeness(rᵢ) ∈ {informative, uninformative}
6. CATEGORIZE:
   IF consistently informative AND consistently incorrect across generations:
      7. Generate refusal response using π
      8. Add (q, refusal) to D_fact    [teaches model to decline]
   ELSE IF consistently informative AND correct:
      9. Add (q, best_rᵢ) to D_fact   [reinforces correct knowledge]
10. RETURN D_fact
```

**Additional data:** Limited set of labeled factuality data covering sensitive topics with prevalent factual contradictions or incorrect statements.

#### 8.6.3 Failure Modes

- Pre-training data itself contains factual inconsistencies
- Model overconfidence in low-knowledge domains
- Refusal calibration: excessive refusal reduces helpfulness

### 8.7 Steerability

#### 8.7.1 Definition

Steerability: the ability to direct model actions and outcomes to meet developer and user specifications through natural language system prompts.

#### 8.7.2 Steerability Dimensions

- Response length
- Response format
- Tone
- Character/persona

#### 8.7.3 Data Collection

- Annotators design diverse system prompts for Llama 3
- Engage in multi-turn conversations evaluating consistency of instruction following over the conversation
- Preference samples collected within general English category

#### 8.7.4 Integration

Steerability preference data flows into:
1. Reward modeling
2. Rejection sampling
3. SFT
4. DPO

---

## 9. Preference Data Pipeline

### 9.1 Annotation Protocol

| Step | Description |
|---|---|
| Model deployment | Multiple models deployed after each round |
| Sampling | Two responses sampled from two different models per prompt |
| Model diversity | Models trained with different data mixes and alignment recipes |
| Preference strength | 4 levels: significantly better, better, slightly better, marginally better |
| Editing step | Annotators edit chosen response or prompt model to refine |
| Result | 2 or 3 ranked responses: (edited > chosen > rejected) |

### 9.2 Preference Data Statistics

| Category | % of Comparisons | Avg. Turns/Dialog | Avg. Tokens/Example | Avg. Prompt Tokens | Avg. Response Tokens |
|---|---|---|---|---|---|
| General English | 81.99% | 4.1 | 1,000.4 | 36.4 | 271.2 |
| Coding | 6.93% | 3.2 | 1,621.0 | 113.8 | 462.9 |
| Multilingual | 5.19% | 1.8 | 1,299.4 | 77.1 | 420.9 |
| Reasoning & Tools | 5.89% | 1.6 | 707.7 | 46.6 | 129.9 |
| **Total** | **100%** | **3.8** | **1,041.6** | **44.5** | **284.0** |

### 9.3 Data Routing

| Stage | Data Selection |
|---|---|
| Reward Modeling | All available preference data across all rounds |
| DPO | Only most recent batches from various capabilities |
| Both | Only "significantly better" or "better" labels; "similar" discarded |

### 9.4 Quality Assurance

- Systematic quality analysis and human evaluation process
- Actionable feedback loops to annotators
- Prompt complexity increased each round to target model weaknesses

---

## 10. Iterative Round Protocol

### 10.1 Formal Round Structure

For round $r \in \{1, \ldots, 6\}$:

$$\theta_r = \text{Avg}\!\left(\text{DPO}\!\left(\text{SFT}\!\left(\theta_0, \mathcal{D}_{\text{SFT}}^{(r)}\right), \mathcal{D}_{\text{pref}}^{(r)}\right)\right)$$

where:
- $\theta_0$ is the pre-trained base checkpoint (constant across rounds)
- $\mathcal{D}_{\text{SFT}}^{(r)}$ includes rejection-sampled data from $\pi_{\theta_{r-1}}$
- $\mathcal{D}_{\text{pref}}^{(r)}$ includes human preferences collected using $\pi_{\theta_{r-1}}$
- SFT always starts from the pre-trained checkpoint $\theta_0$

<img src="assets/llama_3_technical_synthesis_p07.png" alt="Six-round post-training alignment protocol iterated over progressively scaled prompt complexity" width="100%" />

*Figure. Iterative six-round alignment loop viewed from the round-structure perspective, corresponding to the cross-round dependencies and progressively increasing prompt complexity in Section 10.*

### 10.2 Cross-Round Dependencies

```
Round r-1 model → generates synthetic data for round r
Round r-1 model → deployed for preference annotation in round r
All preference data ∪_{i=1}^{r} D_pref^{(i)} → RM training in round r
Only D_pref^{(r)} (latest batch) → DPO training in round r
```

### 10.3 Progressive Complexity Scaling

As model quality improves across rounds:
- Prompt difficulty increases to target remaining weaknesses
- Annotation protocols complexify (e.g., single-turn → multi-turn → multi-step)
- Synthetic data quality ceiling rises with improved generator model
- Expert models (code, multilingual) become available as pre-training completes

---

## 11. Complete End-to-End Pseudo-Algorithm

```
ALGORITHM: Llama3PostTraining
INPUT: Pre-trained model θ₀, number of rounds R=6
OUTPUT: Aligned model θ_final

1. Initialize θ_current ← θ₀

2. TRAIN DOMAIN EXPERTS (parallel with main pipeline):
   3. Code expert: branch θ₀ → CPT on 1T code tokens → LCFT 16K → code post-training
   4. Multilingual expert: branch θ₀ → CPT on 90% multilingual tokens → ML post-training

5. FOR round r = 1 to R:

   === STAGE A: DATA COLLECTION ===
   6. Deploy θ_current and variants for annotation
   7. Collect human preference data D_pref^(r):
      8. Sample 2 responses per prompt from different models
      9. Annotators rank with 4-level strength + optional editing
   10. Collect SFT prompts from human annotation
   11. Generate synthetic data using θ_current:
       12. Code: execution feedback + translation + backtranslation (>2.7M)
       13. Math: step-wise traces + MCTS + interleaved code-text
       14. Long context: QA + summarization + code reasoning (0.1% of mix)
       15. Tool use: single-step + multi-step + file uploads
       16. Factuality: knowledge probing pipeline
       17. Steerability: system prompt preference samples
       18. Multilingual: RS + translated reasoning + NLP task reformatting

   === STAGE B: REWARD MODELING ===
   19. Aggregate D_RM = ∪_{i=1}^{r} D_pref^(i)
   20. Filter: remove similar responses, keep "significantly better"/"better"
   21. Train R_φ on θ₀ initialization with Bradley-Terry loss (no margin)
   22. Support 2-way and 3-way rankings (edited > chosen > rejected)

   === STAGE C: REJECTION SAMPLING ===
   23. FOR each prompt x in human annotation set:
       24. Sample K ∈ [10,30] responses from θ_current using PagedAttention
       25. With capability-specific system prompts in later rounds
       26. Select y* = argmax_k R_φ(x, y^(k))
   27. Apply code-specific model-as-judge filtering (score 2/2 required)

   === STAGE D: SUPERVISED FINETUNING ===
   28. Construct D_SFT^(r) from:
       - Rejection-sampled data
       - Synthetic data (code, math, long context, tools, factuality, steerability, multilingual)
       - Human-curated data
   29. Apply data processing pipeline:
       - Rule-based cleaning
       - Topic classification (Llama 3 8B)
       - Quality scoring (RM top-quartile ∨ Llama max-score)
       - Difficulty scoring (InsTaG + Llama)
       - Semantic deduplication (RoBERTa clustering + cosine threshold)
   30. Finetune θ₀ with cross-entropy loss, lr=1e-5, 8.5K-9K steps
       - Prompt tokens masked
       - Long context data = 0.1% of mix

   === STAGE E: DPO ===
   31. Use only most recent D_pref^(r) for DPO
   32. Filter: "significantly better"/"better" only
   33. Reference model π_ref ← θ_SFT (from Stage D)
   34. Train with modified DPO loss:
       - Mask formatting tokens (headers, terminators)
       - Add NLL regularization (α=0.2) on chosen responses
       - β=0.1, lr=1e-5
   35. Short-context DPO only (does not degrade long-context from SFT)

   === STAGE F: MODEL AVERAGING ===
   36. Run multiple experiments with data/hyperparameter variants
   37. Average parameters: θ_r = (1/M) Σ_m θ^(m)

   38. θ_current ← θ_r

39. θ_final ← θ_R
40. RETURN θ_final
```

---

## 12. Mathematical Summary of All Loss Functions

### 12.1 Reward Model Loss

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( R_\phi(x, y_w) - R_\phi(x, y_l) \right) \right]$$

### 12.2 SFT Loss

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

### 12.3 DPO Loss (Modified)

$$\mathcal{L}_{\text{DPO}}^{\text{total}}(\theta) = \underbrace{-\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta^{\neg\mathcal{M}}(y_w|x)}{\pi_{\text{ref}}^{\neg\mathcal{M}}(y_w|x)} - \beta \log \frac{\pi_\theta^{\neg\mathcal{M}}(y_l|x)}{\pi_{\text{ref}}^{\neg\mathcal{M}}(y_l|x)}\right)\right]}_{\text{Masked DPO}} + \underbrace{0.2 \cdot \left(-\mathbb{E}\!\left[\sum_t \log \pi_\theta(y_{w,t}|x,y_{w,<t})\right]\right)}_{\text{NLL Regularization}}$$

where $\pi^{\neg\mathcal{M}}$ denotes log-probability computed excluding positions in formatting mask set $\mathcal{M}$.

### 12.4 Rejection Sampling Selection

$$y^* = \arg\max_{y^{(k)} \sim \pi_\theta(\cdot|x), \; k=1,\ldots,K} R_\phi(x, y^{(k)})$$

### 12.5 Model Averaging

$$\theta_{\text{final}} = \frac{1}{M}\sum_{m=1}^M \theta^{(m)}$$

### 12.6 Semantic Deduplication Score

$$s_i = q_{\text{combined}}(x_i, y_i) \times d(x_i, y_i), \quad \text{admit iff } \max_{j \in S} \cos(\mathbf{e}_i, \mathbf{e}_j) < \tau$$

---

## 13. Tensor Transformations and Memory Flow

### 13.1 SFT Forward Pass

For input sequence $w = [x_1, \ldots, x_n, y_1, \ldots, y_m]$:

1. **Embedding:** $\mathbf{E} = \text{Embed}(w) \in \mathbb{R}^{(n+m) \times d}$
2. **Transformer layers:** $\mathbf{H}^{(l)} = \text{TransformerBlock}^{(l)}(\mathbf{H}^{(l-1)})$, with RoPE applied to Q,K
3. **Output logits:** $\mathbf{Z} = \mathbf{H}^{(L)} \mathbf{W}_{\text{out}} \in \mathbb{R}^{(n+m) \times V}$
4. **Loss computation:** Only on positions $[n+1, \ldots, n+m]$ (prompt masked)

### 13.2 DPO Memory Requirements

For each DPO sample, four forward passes required:
- $\log \pi_\theta(y_w \mid x)$: current model on chosen
- $\log \pi_\theta(y_l \mid x)$: current model on rejected
- $\log \pi_{\text{ref}}(y_w \mid x)$: reference model on chosen (no gradients)
- $\log \pi_{\text{ref}}(y_l \mid x)$: reference model on rejected (no gradients)

**Memory complexity per sample:**

$$\mathcal{O}\left(2 \cdot (L_w + L_l) \cdot d + \text{activations for gradient computation on } \pi_\theta \text{ passes}\right)$$

The reference model can share parameters with the policy model at initialization but must be frozen; in practice, reference log-probs can be pre-computed and cached:

$$\text{Cache: } \{\log \pi_{\text{ref}}(y_{w,t} \mid x, y_{w,<t})\}_{t=1}^{|y_w|}, \quad \{\log \pi_{\text{ref}}(y_{l,t} \mid x, y_{l,<t})\}_{t=1}^{|y_l|}$$

### 13.3 Rejection Sampling Memory Flow

With PagedAttention:
- **KV-cache sharing:** For prompt $x$ of length $n$, KV-cache pages for positions $[1, \ldots, n]$ shared across $K$ candidates
- **Memory per candidate:** Only incremental KV-cache for generated tokens
- **Total memory:** $\mathcal{O}(n \cdot d_{\text{kv}} + K \cdot L_{\max} \cdot d_{\text{kv}})$ instead of $\mathcal{O}(K \cdot (n + L_{\max}) \cdot d_{\text{kv}})$
- **Savings factor:** $\approx K$ when $n \gg L_{\max}$

---

## 14. Convergence and Stability Analysis

### 14.1 SFT Convergence

- Low learning rate ($10^{-5}$) ensures small parameter updates: $\|\Delta\theta\| \ll \|\theta_0\|$
- KL divergence from pre-trained model implicitly bounded by step count and learning rate
- Consistent across rounds: 8.5K–9K steps robust to data mix changes

### 14.2 DPO Stability

**Without modifications:**
- Formatting token conflicts create oscillating gradients → unstable loss landscape
- Chosen log-prob monotonically decreases → quality degradation

**With modifications (masking + NLL regularization):**
- Masking eliminates contradictory gradients on shared tokens
- NLL term $\alpha = 0.2$ anchors chosen response likelihood:

$$\frac{\partial \mathcal{L}_{\text{NLL}}}{\partial \theta} = -0.2 \sum_t \nabla_\theta \log \pi_\theta(y_{w,t} \mid x, y_{w,<t})$$

This directly counteracts the DPO tendency to decrease chosen log-probs.

### 14.3 Iterative Round Stability

- Each round starts SFT from the **pre-trained checkpoint** $\theta_0$ (not the previous round's aligned model)
- Prevents accumulation of alignment artifacts across rounds
- RM trained on cumulative data provides increasingly accurate reward signal
- DPO trained on latest data only prevents distribution shift from stale preferences

---

## 15. Comprehensive Failure Mode Taxonomy

| Failure Mode | Stage | Cause | Mitigation |
|---|---|---|---|
| Tail repetition | DPO | Formatting tokens in contrastive loss | Formatting token masking |
| Premature termination | DPO | Conflicting gradients on termination tokens | Formatting token masking |
| Chosen log-prob decrease | DPO | Contrastive loss pushes both probs down | NLL regularization ($\alpha=0.2$) |
| Reward hacking | RS/RM | RM exploitable by surface patterns | Multi-round data diversity + model averaging |
| Hallucination | SFT | Training on unverified model-generated data | Knowledge probing + factuality data |
| Excessive apologizing | SFT | Imbalanced safety/helpfulness data | Rule-based phrase frequency balancing |
| Code syntax errors | SFT | Low-quality synthetic code | Static analysis + dynamic testing + model-as-judge |
| Long-context regression | SFT | Insufficient long-context data | 0.1% synthetic long-context data in SFT mix |
| Code-switching | Multilingual RS | High temperature sampling | Fixed temperature ($T=0.6$) + language-match checks |
| Translationese | Multilingual SFT | Machine-translated training data | Avoid MT except for simple math problems |
| Self-improvement failure at scale | Code synthesis | 405B training on own data degrades performance | Execution feedback as external ground truth |
| Quality-difficulty imbalance | Code filtering | Stringent filtering removes hard examples | Strategic response revision for hardest problems |
| Over-refusal | Factuality | Aggressive refusal training | Informativeness scoring to preserve helpfulness |
| Unnecessary tool calls | Tool use | Model defaults to tool use for simple queries | Negative examples: easy queries without tool calls |

---

## 16. Information Preservation Invariants

### 16.1 Knowledge Preservation

$$D_{\text{KL}}\!\left[\pi_{\theta_{\text{aligned}}}(\cdot \mid x) \| \pi_{\theta_0}(\cdot \mid x)\right] \leq \epsilon_{\text{implicit}}$$

Maintained through:
- Low learning rate ($10^{-5}$)
- Limited training steps (8.5K–9K for SFT)
- Starting from pre-trained checkpoint each round
- DPO KL regularization via $\beta = 0.1$

### 16.2 Formatting Invariants

After DPO with modifications:
- Formatting token generation probability remains stable (masking prevents contrastive interference)
- Chosen response generation quality maintained (NLL regularization)

### 16.3 Long-Context Invariants

- 0.1% long-context SFT data preserves 128K context capability from pre-training
- Short-context DPO does not degrade long-context performance (verified empirically)

### 16.4 Capability Monotonicity

Each iterative round improves or maintains performance on target benchmarks:
- Ensured by progressive prompt complexity scaling
- Quality control pipeline filters low-quality synthetic data
- Model averaging reduces variance across hyperparameter choices

<img src="assets/llama_3_technical_synthesis_p15.png" alt="Invariant structure of Llama 3 covering scale and curation, dense stability, disciplined optimization, and compositional integrity" width="100%" />

*Figure. Llama 3 invariants, especially relevant here for disciplined optimization: the deterministic six-round loop, masked DPO, and NLL regularization discussed throughout this post-training report.*

---

## 17. Deployment-Relevant Specifications

### 17.1 Rejection Sampling Serving

| Parameter | Specification |
|---|---|
| Candidates per prompt ($K$) | 10–30 |
| Inference engine | PagedAttention with KV-cache sharing |
| Throughput gain | $>2\times$ vs. naive sampling |
| Memory management | Pre-check capacity before request admission |
| Swap overhead | Eliminated via max-output-length reservation |

<img src="assets/llama_3_405b_technical_blueprint_p13.png" alt="Deployment optimization and inference scaling for Llama 3 405B including pipeline parallelism, FP8 quantization, and KV-cache management" width="100%" />

*Figure. Serving and memory optimization view, corresponding to the rejection-sampling serving stack's dependence on KV-cache efficiency and high-throughput deployment configuration.*

### 17.2 System Prompt Configuration

- Tools individually enable/disable in system prompt
- Python interpreter must be explicitly enabled
- Capability-specific system prompts for RS in later rounds
- Steerability system prompts for length, format, tone, persona

### 17.3 Multi-Message Routing

Messages routed to:
- `user`: Human-facing responses
- `ipython`: Code execution environment
- `system`: System-level instructions

Header and termination tokens control routing and turn-taking.

### 17.4 Scale Parameters

| Component | Scale |
|---|---|
| Base model | Llama 3 405B |
| Post-training rounds | 6 |
| SFT steps per round | 8,500–9,000 |
| Synthetic code examples | >2.7M |
| Backtranslation dialogs | ~1.2M |
| Execution feedback dialogs | ~1M |
| Supported programming languages | 10 |
| Supported natural languages | 8 (including English) |
| Context length | 128K tokens |
| Long-context SFT ratio | 0.1% |
