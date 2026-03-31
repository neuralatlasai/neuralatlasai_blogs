

# KIMI K2: END-TO-END TECHNICAL REPORT

---

## 1. SYSTEM OVERVIEW AND FORMAL PROBLEM FORMULATION

### 1.1 Definition

Kimi K2 is a **Mixture-of-Experts (MoE) autoregressive Transformer** language model with:

- **Total parameters**: $N_{\text{total}} = 1.04 \times 10^{12}$ (1.04T)
- **Activated parameters per token**: $N_{\text{active}} = 32.6 \times 10^{9}$ (32.6B)
- **Expert count**: $E = 384$, with $k = 8$ experts active per token, plus $1$ shared expert
- **Attention mechanism**: Multi-Head Latent Attention (MLA)
- **Pre-training corpus**: $T = 15.5 \times 10^{12}$ tokens
- **Sparsity ratio**: $S = E / k = 384 / 8 = 48$

### 1.2 Formal Objective

The pre-training objective is standard autoregressive language modeling:

$$\mathcal{L}_{\text{pretrain}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t \mid x_{<t})$$

where $\theta$ denotes the full parameter set of the MoE Transformer, and $x_t$ is the token at position $t$ in the training corpus.

### 1.3 Design Invariants

| Invariant | Specification |
|---|---|
| Activated FLOPs per token | Fixed at $N_{\text{active}} = 32.6\text{B}$ parameters equivalent |
| Sparsity | $S = 48$ (384 total / 8 active experts) |
| Attention heads | $H = 64$ |
| Layers | $L = 61$ |
| Hidden dimension | $d_{\text{model}} = 7168$ |
| Expert hidden dimension | $d_{\text{expert}} = 2048$ |
| Dense layers | $1$ (vs. 3 in DeepSeek-V3) |
| Shared experts | $1$ |
| Training stability | Zero loss spikes over 15.5T tokens |
| Context window | $4096 \rightarrow 32768 \rightarrow 131072$ (via YaRN) |

---

## 2. DATA PIPELINE

### 2.1 Corpus Composition

**Domains**: Web Text, Code, Mathematics, Knowledge.

**Total volume**: 15.5T tokens of curated, high-quality data.

**Processing pipelines**: Following Kimi K1.5 methodologies — deduplication, quality filtering, correctness validation, domain-targeted data experiments.

### 2.2 Synthetic Data Generation: Rephrasing Pipeline

#### 2.2.1 Motivation and Formal Objective

**Definition**: Token utility $U(x_t)$ is the effective learning signal contributed by token $x_t$ to model parameter updates. Token efficiency is defined as:

$$\eta_{\text{token}} = \frac{\Delta \mathcal{P}(\theta)}{T_{\text{consumed}}}$$

where $\Delta \mathcal{P}(\theta)$ denotes performance improvement and $T_{\text{consumed}}$ is the number of tokens consumed.

**Problem**: High-quality tokens are finite. Naïve multi-epoch repetition induces overfitting. Rephrasing increases $U(x_t)$ by generating semantically equivalent but lexically and structurally diverse variants.

#### 2.2.2 Knowledge Data Rephrasing

Three components:

**1. Style- and Perspective-Diverse Prompting**
- Inspired by WRAP: a set of engineered prompts $\{p_1, p_2, \ldots, p_M\}$ guides an LLM $\mathcal{M}_{\text{rephrase}}$ to produce faithful rephrasings of source document $D$:

$$D'_m = \mathcal{M}_{\text{rephrase}}(D, p_m), \quad m \in \{1, \ldots, M\}$$

- Each prompt $p_m$ specifies a distinct style/perspective while preserving factual content.

**2. Chunk-wise Autoregressive Generation**

- Long document $D$ is partitioned into chunks $\{c_1, c_2, \ldots, c_K\}$, each of length $\leq 256$ tokens.
- Each chunk is rewritten autoregressively with the preceding rewritten chunk as context:

$$c'_i = \mathcal{M}_{\text{rephrase}}(c_i \mid c'_{<i}), \quad i = 1, \ldots, K$$

- Final output: $D' = \text{concat}(c'_1, c'_2, \ldots, c'_K)$
- Maximum output length per chunk: 4096 tokens.

**3. Fidelity Verification**

- Semantic alignment check between $D$ and $D'$ using a fidelity verifier:

$$\text{fidelity}(D, D') = \text{sim}(\phi(D), \phi(D')) \geq \delta$$

where $\phi(\cdot)$ extracts semantic representations and $\delta$ is the acceptance threshold.

#### 2.2.3 Pseudo-Algorithm: Knowledge Rephrasing Pipeline

```
ALGORITHM: KnowledgeRephrasing
Input: Document D, prompt set {p_1,...,p_M}, chunk_size=256
Output: Rephrased document D'

1: chunks ← SPLIT(D, chunk_size)
2: context ← ∅
3: FOR i = 1 TO |chunks| DO
4:     c'_i ← M_rephrase(chunks[i], context, p_m)   // m sampled from {1,...,M}
5:     context ← c'_i
6: END FOR
7: D' ← CONCAT(c'_1, c'_2, ..., c'_K)
8: IF fidelity(D, D') < δ THEN
9:     DISCARD D'
10: ELSE
11:    RETURN D'
12: END IF
```

#### 2.2.4 Empirical Validation

| # Rephrasings | # Epochs | SimpleQA Accuracy |
|---|---|---|
| 0 (raw) | 10 | 23.76 |
| 1 | 10 | 27.39 |
| 10 | 1 | 28.94 |

**Key finding**: 10 rephrasings × 1 epoch > 1 rephrasing × 10 epochs > 0 rephrasings × 10 epochs. Each corpus is rephrased at most twice in practice.

#### 2.2.5 Mathematics Data Rephrasing

- High-quality mathematical documents rewritten into "learning-note" style (following SwallowMath methodology).
- Cross-lingual augmentation: translation of high-quality mathematical materials from other languages into English.

#### 2.2.6 Failure Modes of Rephrasing

| Failure Mode | Description | Mitigation |
|---|---|---|
| Hallucination injection | Rephraser introduces unsupported facts | Fidelity verification |
| Semantic drift | Successive chunk rewrites drift from original meaning | Context-conditioned autoregressive rewriting |
| Toxicity amplification | LLM rephraser injects unintended bias | Quality filtering |
| Output length limitation | LLMs truncate long documents | Chunk-wise decomposition |
| Domain mismatch | Generic rephraser fails on specialized content | Domain-specialized prompts |

---

## 3. MODEL ARCHITECTURE

### 3.1 Overall Architecture Specification

Kimi K2 is a 61-layer MoE Transformer with MLA attention.

| Parameter | DeepSeek-V3 | Kimi K2 | Delta |
|---|---|---|---|
| Layers | 61 | 61 | = |
| Total Parameters | 671B | 1.04T | ↑ 54% |
| Activated Parameters | 37B | 32.6B | ↓ 13% |
| Total Experts | 256 | 384 | ↑ 50% |
| Active Experts/Token | 8 | 8 | = |
| Shared Experts | 1 | 1 | = |
| Attention Heads | 128 | 64 | ↓ 50% |
| Dense Layers | 3 | 1 | ↓ 67% |
| Expert Grouping | Yes | No | — |
| $d_{\text{model}}$ | 7168 | 7168 | = |
| $d_{\text{expert}}$ | 2048 | 2048 | = |

### 3.2 Multi-Head Latent Attention (MLA)

#### 3.2.1 Definition

MLA compresses the KV cache by projecting keys and values into a low-dimensional latent space, then reconstructing head-specific components during computation. This reduces memory-bound inference cost.

For input representation $X \in \mathbb{R}^{n \times d_{\text{model}}}$:

**Latent compression** (shared across heads):

$$c_{\text{KV}} = X W_{\text{DKV}} \in \mathbb{R}^{n \times d_c}$$

where $W_{\text{DKV}} \in \mathbb{R}^{d_{\text{model}} \times d_c}$ and $d_c \ll H \cdot d_h$.

**Head-specific reconstruction**:

$$K^h_C = c_{\text{KV}} W^h_{UK} \in \mathbb{R}^{n \times d_h}, \quad V^h = c_{\text{KV}} W^h_{UV} \in \mathbb{R}^{n \times d_h}$$

**Query compression** (analogous):

$$c_Q = X W_{DQ} \in \mathbb{R}^{n \times d'_c}$$

$$Q^h_C = c_Q W^h_{UQ} \in \mathbb{R}^{n \times d_h}$$

**Rotary components** (for positional encoding via RoPE):

$$Q^h_R = X W^h_{QR} \in \mathbb{R}^{n \times d_R}, \quad K_R = X W_{KR} \in \mathbb{R}^{n \times d_R}$$

Note: $K_R$ is shared across heads.

**Full query/key construction**:

$$Q^h = [Q^h_C; Q^h_R], \quad K^h = [K^h_C; K_R]$$

**Attention computation** for head $h$:

$$A^h = \text{softmax}\left(\frac{Q^h (K^h)^\top}{\sqrt{d_h + d_R}}\right) V^h$$

**KV-cache efficiency**: Only $c_{\text{KV}} \in \mathbb{R}^{n \times d_c}$ and $K_R \in \mathbb{R}^{n \times d_R}$ are cached, versus $H \cdot d_h$ per head in standard MHA.

$$\text{Cache size ratio} = \frac{d_c + d_R}{2 H d_h}$$

#### 3.2.2 Why MLA Matters for K2

With $H = 64$ heads and $d_h$ per head, MLA compresses the KV cache by storing only the shared latent $c_{\text{KV}}$ and shared rotary key $K_R$. This is critical for:
- **Long-context agentic serving** (128k tokens)
- **Memory-bandwidth-bound inference**
- **Serving multiple concurrent agentic trajectories**

### 3.3 Mixture-of-Experts (MoE) Layer

#### 3.3.1 Formal Definition

For token representation $x \in \mathbb{R}^{d_{\text{model}}}$, the MoE FFN output is:

$$\text{MoE}(x) = \text{SharedExpert}(x) + \sum_{i \in \text{TopK}(g(x), k)} g_i(x) \cdot \text{Expert}_i(x)$$

where:

- $g(x) \in \mathbb{R}^{E}$ is the gating/routing function
- $\text{TopK}(g(x), k)$ selects the top-$k = 8$ experts
- $g_i(x)$ is the normalized gate weight for expert $i$
- Each expert: $\text{Expert}_i(x) = W^i_{\text{down}} \cdot \text{SwiGLU}(W^i_{\text{up}} x)$ with $W^i_{\text{up}} \in \mathbb{R}^{d_{\text{expert}} \times d_{\text{model}}}$, $W^i_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{expert}}}$

**SwiGLU activation**:

$$\text{SwiGLU}(x) = \text{Swish}(W_{\text{gate}} x) \odot (W_{\text{up}} x)$$

$$\text{Swish}(z) = z \cdot \sigma(z)$$

#### 3.3.2 Sparsity Ratio

$$S = \frac{E_{\text{total}}}{E_{\text{active}}} = \frac{384}{8} = 48$$

**Activated FLOPs per token** (for MoE layers):

$$\text{FLOPs}_{\text{MoE}} = k \cdot (3 \cdot d_{\text{model}} \cdot d_{\text{expert}}) + \text{SharedExpert FLOPs}$$

$$= 8 \times (3 \times 7168 \times 2048) + (3 \times 7168 \times 2048)$$

The factor $3$ accounts for up-projection gate, up-projection value, and down-projection in SwiGLU.

#### 3.3.3 Complexity Analysis

| Component | Per-Token FLOPs | Memory |
|---|---|---|
| MLA attention (per layer) | $O(n \cdot d_{\text{model}} \cdot d_c + n^2 \cdot (d_h + d_R))$ | $O(n \cdot (d_c + d_R))$ KV-cache |
| MoE FFN (per layer) | $O(k \cdot d_{\text{model}} \cdot d_{\text{expert}})$ | $O(E \cdot d_{\text{model}} \cdot d_{\text{expert}})$ params |
| Full model forward | $O(L \cdot (n \cdot d^2 / H + n^2 \cdot d_h + k \cdot d_{\text{model}} \cdot d_{\text{expert}}))$ | $O(N_{\text{total}})$ params |

### 3.4 Sparsity Scaling Law

#### 3.4.1 Formal Derivation

Under fixed activated parameters $N_{\text{active}}$ (hence constant per-token FLOPs $C_{\text{token}}$), increasing total experts $E$ while keeping $k$ fixed increases sparsity $S = E/k$. Empirically:

$$\mathcal{L}_{\text{val}}(C, S) = \alpha \cdot C^{-\beta} \cdot S^{-\gamma} + \mathcal{L}_\infty$$

where:
- $C$ = total training compute (FLOPs)
- $S$ = sparsity ratio
- $\alpha, \beta, \gamma$ = fitted scaling coefficients
- $\mathcal{L}_\infty$ = irreducible loss

**Key empirical finding**: At target $\mathcal{L}_{\text{val}} = 1.5$:

| Sparsity | FLOPs Reduction vs. $S=8$ |
|---|---|
| $S = 8$ | baseline |
| $S = 16$ | $1.22\times$ |
| $S = 32$ | $1.47\times$ |
| $S = 48$ | $1.69\times$ |
| $S = 64$ | diminishing returns with infra complexity |

#### 3.4.2 Choice of $S = 48$

$S = 48$ was selected as the Pareto-optimal point balancing:
- **Performance gain** (close to $S = 64$)
- **Infrastructure complexity** (EP communication, load balancing)
- **Memory overhead** ($384 \times d_{\text{expert}} \times d_{\text{model}}$ total expert parameters)

### 3.5 Number of Attention Heads: Design Rationale

#### 3.5.1 Inference FLOPs Analysis

For sequence length $n$, attention FLOPs scale as:

$$\text{FLOPs}_{\text{attn}} \propto H \cdot n \cdot d_h \cdot n = H \cdot d_h \cdot n^2$$

At $n = 131072$ (128k context):

$$\frac{\text{FLOPs}_{H=128}}{\text{FLOPs}_{H=64}} = \frac{128}{64} = 2 \quad (\text{if } d_h \text{ is kept constant})$$

Including KV-cache reconstruction costs in MLA, the actual overhead increase is **83%** when doubling heads from 64 to 128 at 128k context.

#### 3.5.2 Quality Impact

Under iso-token training, doubling attention heads ($H = 2L$ vs. $H = L$, where $L = 61$) yields only **0.5%–1.2%** validation loss improvement across compute budgets from $1.2 \times 10^{20}$ to $9.0 \times 10^{20}$ FLOPs:

$$\frac{\Delta \mathcal{L}_{\text{val}}}{\mathcal{L}_{\text{val}}} \in [0.005, 0.012]$$

**Decision**: $H = 64$ — marginal quality loss, major inference cost savings for agentic long-context workloads.

---

## 4. OPTIMIZATION STRATEGY: MUONCLIP

### 4.1 Muon Optimizer

#### 4.1.1 Definition

Muon (Momentum + Orthogonalization via Newton-Schulz) is an optimizer that applies orthogonal updates to weight matrices, achieving higher token efficiency than AdamW.

For weight matrix $W \in \mathbb{R}^{n \times m}$:

**Momentum update**:
$$M_t = \mu M_{t-1} + G_t$$

where $G_t = \nabla_W \mathcal{L}$ is the gradient at step $t$, $\mu$ is the momentum coefficient, $M_0 = 0$.

**Newton-Schulz orthogonalization**: Approximate polar decomposition to extract the orthogonal component of $M_t$:

$$O_t = \text{Newton-Schulz}(M_t) \cdot \sqrt{\max(n, m)} \cdot 0.2$$

The scaling factor $\sqrt{\max(n, m)} \cdot 0.2$ matches the RMS of Adam updates (consistent update RMS scaling).

**Weight update**:

$$W_t = W_{t-1} - \eta (O_t + \lambda W_{t-1})$$

where $\eta$ is the learning rate and $\lambda = 0.1$ is weight decay.

#### 4.1.2 Newton-Schulz Iteration

Given matrix $M$, the Newton-Schulz iteration computes an approximate unitary factor $U$ such that $M = U \Sigma V^\top$ and $O \approx U V^\top$:

$$X_0 = M / \|M\|_F$$

$$X_{k+1} = \frac{3}{2} X_k - \frac{1}{2} X_k X_k^\top X_k$$

Converges in $\approx 5$ iterations for well-conditioned matrices.

**Computational complexity**: $O(n \cdot m \cdot \min(n, m))$ per iteration — dominated by matrix multiplications.

#### 4.1.3 Why Muon Achieves Higher Token Efficiency

Muon's orthogonal updates ensure that parameter updates lie on the Stiefel manifold (or its approximation), preventing update collapse along dominant singular directions. This:
- Preserves gradient information across all singular directions equally
- Reduces directional bias in parameter space exploration
- Achieves higher effective rank of weight updates

Empirically: under identical compute and model size, Muon **substantially outperforms** AdamW in terms of downstream task performance per token consumed.

### 4.2 Training Instability in Muon at Scale

#### 4.2.1 Failure Mode: Attention Logit Explosion

**Definition**: The maximum pre-softmax attention logit for head $h$ in a batch $B$:

$$S^h_{\max} = \max_{X \in B} \max_{i, j} \left[ (Q^h)_i^\top (K^h)_j \right]$$

**Observed failure**: In Muon training at 9B-active / 53B-total MoE scale, $S^h_{\max}$ rapidly exceeds 1000 within the first 15,000 steps. This causes:
- Softmax saturation: $\text{softmax}(S) \rightarrow$ one-hot, destroying attention diversity
- Numerical instability in BF16/FP16: overflow risk
- Loss spikes and potential divergence

**Root cause**: Muon's orthogonal updates maintain high effective rank, which can amplify projection norms along $W_Q$ and $W_K$ more aggressively than AdamW (which has implicit norm regularization via adaptive learning rates).

#### 4.2.2 Insufficiency of Existing Mitigations

| Method | Problem |
|---|---|
| **Logit soft-cap** | Clips $\text{softmax}$ input but $Q^\top K$ can still grow unboundedly before capping; does not address root cause |
| **QK-Norm** | Normalizes $Q, K$ vectors; **incompatible with MLA** because $K$ matrices are not fully materialized during inference (only $c_{\text{KV}}$ and $K_R$ are cached) |

### 4.3 QK-Clip: Weight-Level Attention Logit Control

#### 4.3.1 Core Mechanism

**Principle**: Post-update rescaling of query and key projection weights to bound $S^h_{\max}$ below threshold $\tau$.

**Key property**: QK-Clip does **not** alter the forward/backward computation of the current step. It uses $S^h_{\max}$ (already computed during forward pass) as a **guiding signal** to control weight growth for the **next** step.

#### 4.3.2 Per-Head Scaling Factor

For each head $h$:

$$\gamma_h = \min\left(1, \frac{\tau}{S^h_{\max}}\right)$$

where $\tau$ is the target threshold (set to $\tau = 100$ for Kimi K2).

**Property**: $\gamma_h = 1$ when $S^h_{\max} \leq \tau$ (no intervention). $\gamma_h < 1$ only for heads with exploding logits, minimizing interference with stable heads.

#### 4.3.3 MLA-Specific Clipping Rules

Because MLA decomposes attention into head-specific and shared components, clipping must respect this decomposition:

| Component | Scope | Scaling |
|---|---|---|
| $W^h_{QC}$ (head-specific query, compressed) | Per-head | $W^h_{QC} \leftarrow W^h_{QC} \cdot \sqrt{\gamma_h}$ |
| $W^h_{KC}$ (head-specific key, compressed) | Per-head | $W^h_{KC} \leftarrow W^h_{KC} \cdot \sqrt{\gamma_h}$ |
| $W^h_{QR}$ (head-specific query, rotary) | Per-head | $W^h_{QR} \leftarrow W^h_{QR} \cdot \gamma_h$ |
| $W_{KR}$ (shared rotary key) | Shared | **Untouched** |

**Rationale for asymmetric scaling**:

The pre-softmax logit for head $h$ decomposes as:

$$S^h_{ij} = (Q^h_C)_i^\top (K^h_C)_j + (Q^h_R)_i^\top (K_R)_j$$

After clipping:

$$(Q^h_C)_i^\top (K^h_C)_j \rightarrow \gamma_h \cdot (Q^h_C)_i^\top (K^h_C)_j$$

$$(Q^h_R)_i^\top (K_R)_j \rightarrow \gamma_h \cdot (Q^h_R)_i^\top (K_R)_j$$

Both terms are scaled by $\gamma_h$, so:

$$S'^h_{ij} = \gamma_h \cdot S^h_{ij}$$

The split $\sqrt{\gamma_h}$ on $Q_C, K_C$ achieves $\sqrt{\gamma_h} \cdot \sqrt{\gamma_h} = \gamma_h$. The full $\gamma_h$ on $Q_R$ with no change to $K_R$ achieves $\gamma_h \cdot 1 = \gamma_h$.

$K_R$ is **left untouched** because it is **shared** across all heads — modifying it would affect all heads, violating the per-head intervention principle.

#### 4.3.4 Naïve (Global) vs. Per-Head Clipping

**Naïve**:
$$\gamma = \min\left(1, \frac{\tau}{S_{\max}}\right), \quad S_{\max} = \max_h S^h_{\max}$$

$$W_Q \leftarrow W_Q \cdot \gamma^\alpha, \quad W_K \leftarrow W_K \cdot \gamma^{1-\alpha}$$

with $\alpha = 0.5$ (equal scaling).

**Problem**: Clips all heads uniformly, even stable ones. In practice, only a small subset of heads exhibit exploding logits.

**Per-head** (adopted): Minimizes intervention — only exploding heads are rescaled.

### 4.4 MuonClip: Unified Algorithm

```
ALGORITHM 1: MuonClip Optimizer
Input: Model parameters θ, learning rate η, weight decay λ, momentum μ,
       QK-Clip threshold τ
Output: Updated parameters θ

FOR each training step t DO

  // Phase 1: Muon optimizer step
  FOR each weight W ∈ ℝ^{n×m} DO
    M_t ← μ·M_{t-1} + G_t                              // Momentum accumulation
    O_t ← Newton-Schulz(M_t) · √(max(n,m)) · 0.2       // Orthogonalize + RMS match
    W_t ← W_{t-1} - η·(O_t + λ·W_{t-1})                // Update with weight decay
  END FOR

  // Phase 2: QK-Clip (post-update)
  FOR each attention layer ℓ DO
    FOR each head h DO
      Retrieve S^h_max (computed during forward pass)
      IF S^h_max > τ THEN
        γ ← τ / S^h_max
        W^h_QC ← W^h_QC · √γ                            // Head-specific query compressed
        W^h_KC ← W^h_KC · √γ                            // Head-specific key compressed
        W^h_QR ← W^h_QR · γ                              // Head-specific query rotary
        // W_KR (shared rotary key) is NOT modified
      END IF
    END FOR
  END FOR

END FOR
```

### 4.5 Empirical Validation of MuonClip

**Mid-scale experiment** (9B active / 53B total MoE):
- Vanilla Muon: $S^h_{\max} > 1000$ within 15k steps → instability, loss spikes, divergence risk
- MuonClip ($\tau = 100$): logits capped at 100 initially, naturally decay to stable operating range after ~30% of training steps

**Kimi K2 full-scale**:
- $\tau = 100$ applied throughout 15.5T tokens
- **Zero loss spikes** over entire training (verified by per-step unsmoothed loss curve)
- Max logits initially capped at 100, decay to natural stable range without $\tau$ adjustment
- No degradation of model quality vs. vanilla Muon (confirmed by ablation)

### 4.6 Complexity and Overhead of QK-Clip

| Operation | Cost | Relative Overhead |
|---|---|---|
| Compute $S^h_{\max}$ | Already available from forward pass attention | **Zero** additional FLOPs |
| Per-head comparison $S^h_{\max} > \tau$ | $O(H \cdot L)$ comparisons | Negligible |
| Weight rescaling | $O(d_c \cdot d_h)$ per affected head | Negligible vs. forward/backward |
| Total overhead | — | **< 0.01%** of training step time |

---

## 5. TRAINING STAGES

### 5.1 Stage 1: Pre-training (Constant LR Phase)

| Parameter | Value |
|---|---|
| Context length | 4096 tokens |
| Optimizer | MuonClip |
| LR schedule | WSD (Warmup-Stable-Decay) |
| Warmup | 500 steps |
| Constant LR | $2 \times 10^{-4}$ |
| Weight decay | $\lambda = 0.1$ |
| Batch size | $67 \times 10^6$ tokens |
| Token budget | 10T tokens |
| Precision | BF16 parameters, FP32 gradient accumulation |

### 5.2 Stage 2: Pre-training (Cosine Decay Phase)

| Parameter | Value |
|---|---|
| Context length | 4096 tokens |
| LR schedule | Cosine decay from $2 \times 10^{-4}$ to $2 \times 10^{-5}$ |
| Token budget | 5.5T tokens |
| All other hyperparams | Same as Stage 1 |

**Total pre-training**: $10\text{T} + 5.5\text{T} = 15.5\text{T}$ tokens

### 5.3 Stage 3: Annealing + Long-Context Activation

**Phase 3a**: Annealing (4k context)

| Parameter | Value |
|---|---|
| Context length | 4096 tokens |
| LR | Decayed from $2 \times 10^{-5}$ to $7 \times 10^{-6}$ |
| Batch size | $67 \times 10^6$ tokens |
| Token budget | 400B tokens |

**Phase 3b**: Long-context activation (32k context)

| Parameter | Value |
|---|---|
| Context length | 32768 tokens |
| Token budget | 60B tokens |

**Phase 3c**: Context extension to 128k via YaRN

YaRN (Yet another RoPE extensioN) modifies the RoPE frequency basis to extrapolate beyond training sequence length:

$$f'_i = f_i \cdot s^{-2i/d_R}$$

where $s$ is the scaling factor computed from the ratio of target to training context lengths, applied to the rotary position encoding frequencies.

### 5.4 Stage 4: Supervised Fine-Tuning (SFT)

#### 5.4.1 Objective

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{|\mathcal{D}_{\text{SFT}}|}\sum_{(x,y) \in \mathcal{D}_{\text{SFT}}} \sum_{t=1}^{|y|} \log p_\theta(y_t \mid x, y_{<t})$$

where $x$ is the instruction/prompt, $y$ is the target response, and $\mathcal{D}_{\text{SFT}}$ spans diverse domains.

#### 5.4.2 Optimizer

Muon (consistent with pre-training; Muon-pretrained checkpoints perform best with Muon fine-tuning).

#### 5.4.3 Data Construction

**Core principles**: Maximize prompt diversity, ensure high response quality.

**Pipeline components**:
1. Human annotation for high-quality seed data
2. Prompt engineering with K1.5 and domain-specialized expert models for candidate response generation
3. LLM-based or human-based quality evaluation and filtering
4. Agentic data synthesis pipeline (Section 5.4.4)

#### 5.4.4 Large-Scale Agentic Data Synthesis Pipeline

**Three-stage pipeline**:

**Stage A: Tool Specification Generation**

*Real-world tools*:
- 3000+ MCP (Model Context Protocol) tools fetched from GitHub repositories

*Synthetic tools*:
- Hierarchical domain evolution: seed categories → specific application domains → specialized tools
- 20,000+ synthetic tools generated
- Each tool has clear interfaces, descriptions, operational semantics

```
ALGORITHM: ToolSpecGeneration
Input: Seed categories C = {c_1,...,c_N}, evolution model M_evolve
Output: Tool repository T

1: T_real ← FETCH_MCP_TOOLS(GitHub)             // 3000+ real MCP tools
2: T_synth ← ∅
3: FOR each category c ∈ C DO
4:     domains ← M_evolve.generate_domains(c)    // evolve specific domains
5:     FOR each domain d ∈ domains DO
6:         tools ← M_evolve.generate_tools(d)    // synthesize tools with specs
7:         T_synth ← T_synth ∪ tools
8:     END FOR
9: END FOR
10: T ← T_real ∪ T_synth                         // 23,000+ total tools
11: RETURN T
```

**Stage B: Agent and Task Generation**

- **Agent diversification**: Thousands of distinct agents via:
  - Synthesized system prompts
  - Different tool combinations sampled from repository $T$
  - Varied capabilities, expertise areas, behavioral patterns

- **Rubric-based task generation**: For each agent configuration:
  - Tasks ranging from simple to complex operations
  - Each task paired with explicit rubric specifying:
    - Success criteria
    - Expected tool-use patterns
    - Evaluation checkpoints

**Stage C: Trajectory Generation**

*Multi-turn trajectory synthesis via multi-agent simulation*:

```
ALGORITHM: TrajectoryGeneration
Input: Agent A, Task τ, Rubric R, Tool Simulator S, User Simulator U
Output: Trajectory T or ∅ (filtered)

1: state ← S.initialize()
2: conversation ← []
3: user_msg ← U.generate_initial_query(τ)
4: conversation.append(user_msg)
5: FOR turn = 1 TO MAX_TURNS DO
6:     agent_response ← A.respond(conversation, state)
7:     IF agent_response.has_tool_call THEN
8:         tool_result ← S.execute(agent_response.tool_call, state)
9:         state ← S.update_state(tool_result)     // persistent state updates
10:        conversation.append(tool_result)
11:    END IF
12:    conversation.append(agent_response)
13:    IF task_complete(conversation, τ) THEN BREAK
14:    user_msg ← U.generate_followup(conversation, τ)
15:    conversation.append(user_msg)
16: END FOR
17: T ← conversation
18: score ← Judge.evaluate(T, R)                    // LLM-based judge vs. rubric
19: IF score ≥ threshold THEN
20:    RETURN T
21: ELSE
22:    RETURN ∅                                      // filtered out
23: END IF
```

**Tool Simulator Properties**:
- Functions as a **world model**: maintains state, updates after each execution
- Introduces **controlled stochasticity**: varied outcomes including successes, partial failures, edge cases
- Produces realistic feedback for multi-step interactions with persistent effects

**Hybrid real execution**:
- Complements simulation with real sandboxes for coding/SE tasks
- Actual code execution, genuine development environments
- Ground-truth feedback via test suite pass rates
- Effectively implements **large-scale rejection sampling** through quality filtering

### 5.5 Stage 5: Reinforcement Learning

#### 5.5.1 RL Objective (Policy Optimization)

For problem $x$, sample $K$ responses $\{y_1, \ldots, y_K\}$ from previous policy $\pi_{\text{old}}$. The objective for policy $\pi_\theta$:

$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{K} \sum_{i=1}^{K} \left( -\frac{r(x, y_i) - \bar{r}(x)}{\tau} \cdot \log \pi_\theta(y_i \mid x) \right) \right]$$

where:
- $\bar{r}(x) = \frac{1}{K} \sum_{i=1}^{K} r(x, y_i)$ is the **mean reward baseline** (reduces variance)
- $\tau > 0$ is the **temperature/regularization parameter** (promotes stable learning)
- $r(x, y_i)$ is the reward for response $y_i$ to problem $x$

This is a **REINFORCE-style** estimator with a per-problem baseline, equivalent to a simplified advantage estimate:

$$\hat{A}(x, y_i) = \frac{r(x, y_i) - \bar{r}(x)}{\tau}$$

#### 5.5.2 Verifiable Rewards Gym (RLVR)

**Math, STEM, and Logical Tasks**:

*Data construction principles*:

1. **Diverse coverage**: QA pairs from expert annotations, internal extraction pipelines, open datasets, covering under-represented domains via a tagging system. Logical tasks include tabular reasoning, cross-table aggregation, 24-game, Sudoku, riddles, cryptarithms, Morse-code decoding.

2. **Moderate difficulty**: Filter problems by SFT model's $\text{pass}@k$ accuracy. Select only problems where:

$$p_{\text{low}} < \text{pass}@k < p_{\text{high}}$$

Problems too easy ($\text{pass}@k \approx 1$) or too hard ($\text{pass}@k \approx 0$) provide little gradient signal.

**Reward function**: Binary verifiable reward:

$$r_{\text{verify}}(x, y) = \begin{cases} 1 & \text{if } \text{verify}(y, a^*) = \text{True} \\ 0 & \text{otherwise} \end{cases}$$

where $a^*$ is the ground-truth answer and $\text{verify}$ is domain-specific (exact match, code execution, symbolic equivalence).

**Complex Instruction Following**:

*Hybrid rule verification*:
1. **Deterministic verification**: Code interpreter evaluates verifiable constraints (length, format, style)
2. **LLM-as-judge**: Evaluates nuanced constraint compliance
3. **Hack-check layer**: Detects adversarial behaviors where model claims compliance without actual fulfillment

*Multi-source instruction generation*:
1. Expert-crafted complex conditional prompts with rubrics
2. Agentic instruction augmentation (AutoIF-style)
3. Fine-tuned model for generating failure-mode-probing instructions

**Faithfulness**:

- Sentence-level faithfulness judge model trained on FACTS Grounding framework
- Detects unsupported factual claims
- Serves as reward model:

$$r_{\text{faith}}(x, y) = \frac{1}{|S_y|} \sum_{s \in S_y} \mathbb{1}[\text{faithful}(s, \text{context})]$$

where $S_y$ is the set of sentences in response $y$.

**Coding & Software Engineering**:

- Competition-level problems with human-written unit tests
- SE tasks from GitHub pull requests/issues with executable tests
- Sandbox infrastructure: Kubernetes-powered, 10,000+ concurrent instances

$$r_{\text{code}}(x, y) = \frac{\text{tests\_passed}(y)}{\text{total\_tests}(x)}$$

**Safety**:

- Human-curated seed prompts spanning violence, fraud, discrimination
- Automated prompt evolution pipeline:
  - **Attack model**: generates adversarial prompts iteratively
  - **Target model**: produces responses (simulates vulnerabilities)
  - **Judge model**: binary success/failure label per rubric

$$r_{\text{safety}}(x, y) = \begin{cases} 1 & \text{if Judge deems response safe} \\ 0 & \text{otherwise} \end{cases}$$

#### 5.5.3 Self-Critique Rubric Reward

**Motivation**: Extends RL beyond tasks with verifiable rewards to open-ended, subjective domains (creative writing, helpfulness, depth of reasoning, factuality).

**Mechanism**:

1. **K2 Actor** generates $K$ responses $\{y_1, \ldots, y_K\}$ for prompt $x$
2. **K2 Critic** ranks responses via **pairwise evaluation** against rubrics:
   - **Core rubrics**: Fundamental values (helpfulness, safety, honesty)
   - **Prescriptive rubrics**: Anti-reward-hacking rules
   - **Human-annotated rubrics**: Domain-specific instructional contexts

Pairwise comparison:

$$\text{pref}(y_i, y_j \mid x, \mathcal{R}) = \text{Critic}(x, y_i, y_j, \mathcal{R}) \in \{y_i \succ y_j, y_j \succ y_i, \text{tie}\}$$

Reward derived from pairwise ranking:

$$r_{\text{self-critique}}(x, y_i) = \frac{1}{K-1} \sum_{j \neq i} \mathbb{1}[\text{pref}(y_i, y_j) = y_i \succ y_j]$$

**Closed-Loop Critic Refinement**:

- Critic is continuously updated using verifiable signals from RLVR:
  - On-policy rollouts from verifiable tasks provide objective performance labels
  - These ground the critic's subjective judgments in verifiable data
  - **Transfer learning**: RLVR performance gains distill into critic evaluation quality

$$\mathcal{L}_{\text{critic}}(\phi) = -\sum_{(x,y_w,y_l) \in \mathcal{D}_{\text{verify}}} \log \sigma\left(\text{Critic}_\phi(x, y_w) - \text{Critic}_\phi(x, y_l)\right)$$

where $(y_w, y_l)$ are winner/loser pairs determined by verifiable reward.

#### 5.5.4 Joint RL Training Augmentations

**Budget Control**:

Per-sample maximum token budget $B_{\text{max}}(x)$ determined by task type:

$$r_{\text{budget}}(x, y) = \begin{cases} r(x, y) & \text{if } |y| \leq B_{\text{max}}(x) \\ r_{\text{penalty}} < 0 & \text{if } |y| > B_{\text{max}}(x) \end{cases}$$

Responses exceeding $B_{\text{max}}$ are truncated and penalized, incentivizing concise solutions.

**PTX Loss (Pre-Training eXperience)**:

Auxiliary loss to prevent catastrophic forgetting:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RL}}(\theta) + \beta \cdot \mathcal{L}_{\text{PTX}}(\theta)$$

$$\mathcal{L}_{\text{PTX}}(\theta) = -\frac{1}{|\mathcal{D}_{\text{PTX}}|} \sum_{(x,y) \in \mathcal{D}_{\text{PTX}}} \log \pi_\theta(y \mid x)$$

where $\mathcal{D}_{\text{PTX}}$ is a curated set of high-quality samples and $\beta$ controls the mixing ratio.

**Temperature Decay**:

Sampling temperature $T_{\text{sample}}$ follows a decay schedule:

$$T_{\text{sample}}(t) = T_{\text{high}} \cdot \left(\frac{T_{\text{low}}}{T_{\text{high}}}\right)^{t / t_{\text{max}}}$$

- **Early training** ($T_{\text{high}}$): Promotes exploration, diverse responses, reduces premature convergence
- **Late training** ($T_{\text{low}}$): Exploitation, stable high-quality outputs

### 5.6 Pseudo-Algorithm: Full RL Training Loop

```
ALGORITHM: JointRLTraining
Input: SFT checkpoint θ_0, reward functions {r_verify, r_self-critique, r_safety},
       PTX data D_PTX, task distribution D_tasks, budget function B_max
Output: Aligned policy θ*

1: θ ← θ_0
2: critic ← initialize_critic(θ_0)              // Bootstrap from SFT stage
3: FOR iteration t = 1 TO T_max DO
4:     // Phase 1: Rollout generation
5:     batch ← SAMPLE(D_tasks)
6:     T_sample ← TEMPERATURE_DECAY(t)
7:     FOR each problem x in batch DO
8:         {y_1,...,y_K} ← GENERATE(π_θ, x, T_sample)    // K samples
9:         
10:        // Phase 2: Reward computation
11:        IF x ∈ VERIFIABLE_TASKS THEN
12:            FOR i = 1 TO K DO
13:                r_i ← r_verify(x, y_i)
14:            END FOR
15:        ELSE
16:            FOR i = 1 TO K DO
17:                r_i ← r_self-critique(x, y_i, {y_j}_{j≠i}, critic)
18:            END FOR
19:        END IF
20:        
21:        // Phase 3: Budget control
22:        FOR i = 1 TO K DO
23:            IF |y_i| > B_max(x) THEN
24:                r_i ← r_penalty
25:            END IF
26:        END FOR
27:        
28:        // Phase 4: Advantage estimation
29:        r̄ ← (1/K) Σ_i r_i
30:        FOR i = 1 TO K DO
31:            Â_i ← (r_i - r̄) / τ
32:        END FOR
33:    END FOR
34:    
35:    // Phase 5: Policy update
36:    L_RL ← -(1/|batch|) Σ_x Σ_i Â_i · log π_θ(y_i|x)
37:    L_PTX ← -(1/|D_PTX_batch|) Σ log π_θ(y|x)       // From curated data
38:    L_total ← L_RL + β · L_PTX
39:    θ ← MUON_STEP(θ, ∇L_total)
40:    
41:    // Phase 6: Critic refinement (closed-loop)
42:    IF t mod critic_update_freq = 0 THEN
43:        D_critic ← COLLECT_VERIFIABLE_PAIRS(rollouts)
44:        critic ← UPDATE_CRITIC(critic, D_critic)
45:    END IF
46: END FOR
47: RETURN θ
```

---

## 6. TRAINING INFRASTRUCTURE

### 6.1 Compute Cluster

| Resource | Specification |
|---|---|
| GPUs | NVIDIA H800 |
| Per-node GPUs | 8 (connected via NVLink + NVSwitch) |
| Per-node RAM | 2 TB |
| Inter-node interconnect | $8 \times 400$ Gbps RoCE |
| Training flexibility | Any number of nodes that is a multiple of 32 |

### 6.2 Parallelism Strategy

**Multi-dimensional parallelism**: 16-way PP × 16-way EP × ZeRO-1 DP

| Dimension | Size | Scope |
|---|---|---|
| Pipeline Parallelism (PP) | 16 | Virtual stages with interleaved 1F1B |
| Expert Parallelism (EP) | 16 | Distributes 384 experts across 16 devices |
| Data Parallelism | $N_{\text{nodes}} / 32 \times \text{DP factor}$ | ZeRO-1 (optimizer state sharding) |

**Model-parallel group**: 256 GPUs ($16 \times 16$)

**Memory budget** (BF16 params + FP32 grad accum):

$$\text{Memory}_{\text{params+grads}} \approx N_{\text{total}} \times (2 + 4) \text{ bytes} = 1.04\text{T} \times 6\text{B} \approx 6\text{ TB}$$

Distributed over 256 GPUs → $\approx 23.4$ GB/GPU for params+grads.

With optimizer states (Muon momentum in FP32 = 4 bytes/param):
- Large-scale: distributed across DP dimension → negligible per-device
- Small-scale (32 nodes): offloaded to CPU

**Per-GPU allocation**: $\approx 30$ GB for model states → remaining memory for activations.

### 6.3 EP Communication Overlap

**Method**: Increase warm-up micro-batches in interleaved 1F1B schedule to overlap EP all-to-all with computation.

**Why not DualPipe**: DualPipe doubles memory for parameters and gradients → requires more parallelism → more bubbles (PP) or more EP overhead. Prohibitive at 1T+ scale.

**Weight-gradient decoupling**: Weight-gradient computation separated from micro-batch backward pass, executed in parallel with PP communication → all PP communications overlapped except warm-up phase.

**Why EP = 16 (smallest feasible)**:
- K2 has 64 attention heads (vs. 128 in DeepSeek-V3) → reduced attention computation time
- Smaller EP size → less EP communication volume → easier to overlap
- Smaller EP group relaxes expert-balance constraints → near-optimal speed without further tuning

### 6.4 Activation Memory Reduction

#### 6.4.1 Selective Recomputation

Recomputed during backward pass (high memory footprint, low FLOPs):

| Component | Recomputed? | Rationale |
|---|---|---|
| LayerNorm | Yes | High activation footprint, cheap to recompute |
| SwiGLU | Yes | Stores gate and up-projection activations |
| MLA up-projections | Yes | Large intermediate tensors |
| MoE down-projections | Yes (optional) | Prevents crashes from expert imbalance |

#### 6.4.2 FP8 Storage for Insensitive Activations

- **Targets**: Inputs of MoE up-projections and SwiGLU
- **Format**: FP8-E4M3 in $1 \times 128$ tiles with FP32 scales
- **Validation**: No measurable loss increase in small-scale experiments
- **Note**: FP8 is NOT used in computation (only storage) due to observed performance degradation risks

**Compression ratio**:

$$\text{CR}_{\text{FP8}} = \frac{2 \text{ bytes (BF16)}}{1 \text{ byte (FP8)} + \frac{4 \text{ bytes (FP32 scale)}}{128}} \approx \frac{2}{1.03125} \approx 1.94\times$$

#### 6.4.3 Activation CPU Offload

- **All remaining activations** offloaded to CPU RAM (2 TB/node)
- **Copy engine**: Streams offload/onload overlapped with computation and communication
- **1F1B phase**: Offload previous micro-batch's forward activations while prefetching next micro-batch's backward activations
- **PCIe traffic**: May slightly affect EP traffic due to congestion, but EP communication remains fully overlapped

### 6.5 Memory Flow Diagram (Per-GPU)

```
┌─────────────────────────────────────┐
│           GPU Memory (~80GB)        │
│                                     │
│  Model States (~30GB)               │
│  ├─ Parameters (BF16 shard)        │
│  ├─ Gradient Accum Buffer (FP32)   │
│  └─ Optimizer State (distributed)   │
│                                     │
│  Activations (~50GB)                │
│  ├─ Current micro-batch FWD acts   │
│  ├─ FP8 compressed acts (E4M3)    │
│  └─ Recomputation workspace        │
│                                     │
└─────────┬───────────────────────────┘
          │ PCIe (offload/onload)
┌─────────▼───────────────────────────┐
│         CPU RAM (2TB/node)          │
│  ├─ Offloaded activations          │
│  ├─ Optimizer states (small-scale) │
│  └─ Copy engine buffers            │
└─────────────────────────────────────┘
```

---

## 7. RL INFRASTRUCTURE

### 7.1 Colocated Architecture

Training engine and inference engine **share the same GPU workers**. Mutual exclusion: when one engine is active, the other releases/offloads GPU resources.

```
ALGORITHM: RLIterationLoop
Input: Centralized controller, training engine T_eng, inference engine I_eng

FOR each RL iteration DO
  // Phase 1: Rollout (inference)
  T_eng.offload_to_DRAM()
  I_eng.activate()
  rollout_data ← I_eng.generate_rollouts()
  
  // Phase 2: Engine switch
  I_eng.release()
  T_eng.load_from_DRAM()                  // H2D transmission
  
  // Phase 3: Training
  T_eng.train(rollout_data)
  
  // Phase 4: Parameter synchronization
  CHECKPOINT_ENGINE.broadcast(T_eng.params)
  I_eng.update_params(CHECKPOINT_ENGINE)
END FOR
```

### 7.2 Efficient Engine Switching via Checkpoint Engine

**Problem**: Training engine and inference engine use **different sharding paradigms**. Network filesystem resharding at 1T scale requires petabytes/second aggregate bandwidth.

**Solution**: Distributed **checkpoint engine** co-located on training nodes.

```
ALGORITHM: ParameterUpdate
Input: Training engine params (T_eng sharding),
       Checkpoint engine workers {CE_1,...,CE_N},
       Inference engine workers {IE_1,...,IE_M}

1: // Step 1: Collect from training engine (pipelined, per-parameter)
   FOR each parameter P in model DO
     CE_local ← T_eng.get_local_shard(P)
   
2:   // Step 2: Broadcast full parameter across all CE workers
     ALLGATHER(CE_local) → CE_full_P across all CE workers
   
3:   // Step 3: Inference engine retrieves needed shard
     FOR each IE_j DO
       IE_j.shard(P) ← CE_nearest.extract_shard(P, IE_j.sharding_spec)
     END FOR
   END FOR
```

**Design trade-off**: Broadcasting full parameter set across cluster transfers more data than theoretically optimal, but:
- Simpler system design
- Fully decouples training and inference engines
- Reduces synchronization overhead
- Higher network bandwidth utilization

**Performance**: Full parameter update for 1T model: **< 30 seconds** (negligible for RL iteration).

### 7.3 Efficient System Startup

**Training engine startup**:
- Each worker selectively reads part/none of parameters from disk
- Broadcasts necessary parameters to peers
- Design ensures all workers collectively read checkpoint **exactly once** (minimizes disk I/O)

**Inference engine startup**:
- Reuses checkpoint engine
- Checkpoint engine reads from disk (like training startup)
- Updates uninitialized inference engine using same parameter distribution mechanism
- Robust to single-point failures: inference replica can restart without communicating with other replicas

### 7.4 Agentic Rollout Optimizations

| Challenge | Solution |
|---|---|
| GPU idle during environment interaction | (i) Heavy environments deployed as dedicated scalable services; (ii) Large number of concurrent rollouts to amortize latency |
| Long-tail trajectories blocking rollout | **Partial rollout**: Long unfinished tasks paused, resumed in next RL iteration |
| Diverse environment integration | Unified OpenAI Gym-inspired interface for new environments |

---

## 8. INFERENCE PATH

### 8.1 Forward Pass Per Token

For input token $x_t$ with context $x_{<t}$:

**Step 1: Embedding**
$$h^0_t = \text{Embed}(x_t) \in \mathbb{R}^{d_{\text{model}}}$$

**Step 2: For each layer $\ell = 1, \ldots, 61$**:

*Attention sub-layer (MLA)*:

$$c^Q_t = h^{\ell-1}_t W^{\ell}_{DQ} \in \mathbb{R}^{d'_c}$$

$$Q^{h,\ell}_{C,t} = c^Q_t W^{h,\ell}_{UQ} \in \mathbb{R}^{d_h}, \quad Q^{h,\ell}_{R,t} = h^{\ell-1}_t W^{h,\ell}_{QR} \in \mathbb{R}^{d_R}$$

$$c^{\text{KV}}_t = h^{\ell-1}_t W^{\ell}_{DKV} \in \mathbb{R}^{d_c} \quad \text{(cached)}$$

$$K^{h,\ell}_{C,t} = c^{\text{KV}}_t W^{h,\ell}_{UK}, \quad K^{\ell}_{R,t} = h^{\ell-1}_t W^{\ell}_{KR} \in \mathbb{R}^{d_R} \quad \text{(cached)}$$

$$V^{h,\ell}_t = c^{\text{KV}}_t W^{h,\ell}_{UV} \in \mathbb{R}^{d_h}$$

$$A^{h,\ell}_t = \text{softmax}\left(\frac{[Q^{h,\ell}_{C,t}; Q^{h,\ell}_{R,t}]^\top [K^{h,\ell}_{C,:}; K^{\ell}_{R,:}]}{\sqrt{d_h + d_R}}\right) V^{h,\ell}_{:}$$

$$\text{attn}^{\ell}_t = \text{Concat}(A^{1,\ell}_t, \ldots, A^{H,\ell}_t) W^{\ell}_O$$

*MoE FFN sub-layer*:

$$g(h) = \text{Router}(h^{\ell}_{\text{attn}}) \in \mathbb{R}^{384}$$

$$\text{TopK}(g, 8) \rightarrow \{e_1, \ldots, e_8\}$$

$$\text{MoE}(h) = \text{SharedExpert}(h) + \sum_{i=1}^{8} \bar{g}_{e_i} \cdot \text{Expert}_{e_i}(h)$$

where $\bar{g}_{e_i}$ is the normalized gate weight for selected expert $e_i$.

*Residual connections and LayerNorm applied at each sub-layer.*

**Step 3: Output projection**

$$p(x_{t+1} \mid x_{\leq t}) = \text{softmax}(h^{61}_t W_{\text{vocab}})$$

### 8.2 KV-Cache Structure

Per token, per layer, cached:
- $c^{\text{KV}}_t \in \mathbb{R}^{d_c}$ (shared latent)
- $K^{\ell}_{R,t} \in \mathbb{R}^{d_R}$ (shared rotary key)

**Total KV-cache per token**:

$$\text{KV-cache/token} = L \times (d_c + d_R) \times 2 \text{ bytes (BF16)}$$

For 128k context: $131072 \times 61 \times (d_c + d_R) \times 2$ bytes.

This is **dramatically smaller** than standard MHA cache: $L \times 2 \times H \times d_h \times 2$ bytes.

### 8.3 Context Window

| Setting | Max Context |
|---|---|
| Pre-training | 4096 |
| Long-context activation | 32768 |
| YaRN extension | 131072 (128k) |
| Evaluation | 128k (inputs truncated beyond) |

### 8.4 Inference Considerations for Agentic Workloads

- 64 heads (vs. 128) → **83% FLOPs reduction** in attention at 128k context
- MLA KV-cache compression → efficient multi-turn trajectory storage
- MoE sparsity → only 32.6B of 1.04T parameters activated per token
- Multi-turn tool-use requires persistent KV-cache across interaction turns

---

## 9. EVALUATION PROTOCOL

### 9.1 Post-Training Evaluation (Kimi-K2-Instruct)

#### 9.1.1 Benchmark Suite

| Domain | Benchmarks |
|---|---|
| **Coding** | LiveCodeBench v6, OJBench, MultiPL-E, SWE-bench Verified (Agentless + Agentic), TerminalBench, Multi-SWE-bench, SWE-Lancer, PaperBench, Aider-Polyglot |
| **Tool Use** | τ2-Bench (retail, airline, telecom), ACEBench |
| **Math & STEM** | AIME 2024/2025, MATH-500, HMMT 2025, CNMO 2024, PolyMath-en, ZebraLogic, AutoLogi, GPQA-Diamond, SuperGPQA, Humanity's Last Exam |
| **Long-Context** | MRCR (retrieval), DROP, FRAMES, LongBench v2 (reasoning) |
| **Factuality** | FACTS Grounding, Vectara Hallucination (HHEM v2.1), FaithJudge |
| **General** | MMLU, MMLU-Redux, MMLU-Pro, IFEval, Multi-Challenge, SimpleQA, LiveBench, Arena Hard v2.0 |

#### 9.1.2 Evaluation Configurations

| Setting | Value |
|---|---|
| Mode | Non-thinking (all models) |
| Max output tokens | 8192 (default), 16384 (SWE-bench Verified Agentless) |
| High-variance benchmarks | Avg@k (repeated sampling) |
| Context window | 128k (truncate beyond) |
| Temperature/top-p | Unified across models |

**SWE-bench Verified modes**:
1. **Agentless Coding**: Single Patch without Test (Acc)
2. **Agentic Coding — Single Attempt** (Acc): bash/editor tools
3. **Agentic Coding — Multi-Attempt** (Acc): best-of-N with internal verifier

#### 9.1.3 Key Results (Kimi-K2-Instruct)

**Agentic & SE SOTA**:

| Benchmark | K2 | Best Open-Source Baseline | Best Proprietary |
|---|---|---|---|
| SWE-bench Verified (Agentic-Single) | **65.8** | 38.8 (DS-V3-0324) | 72.5 (Claude 4 Opus) |
| SWE-bench Verified (Agentic-Multi) | **71.6** | — | 80.2 (Claude 4 Sonnet) |
| SWE-bench Multilingual | **47.3** | 25.8 (DS-V3-0324) | 51.0 (Claude 4 Sonnet) |
| τ2-Bench micro-avg | **66.1** | 41.0 (DS-V3-0324) | 67.6 (Claude 4 Opus) |
| ACEBench (en) | **76.5** | 72.7 (DS-V3-0324) | 80.1 (GPT-4.1) |
| LiveCodeBench v6 | **53.7** | 46.9 (DS-V3-0324) | 48.5 (Claude 4 Sonnet) |
| OJBench | **27.1** | 24.0 (DS-V3-0324) | 19.6 (Claude 4 Opus) |

**Math & STEM**:

| Benchmark | K2 | Best Baseline |
|---|---|---|
| AIME 2025 (Avg@64) | **49.5** | 46.7 (DS-V3) |
| GPQA-Diamond (Avg@8) | **75.1** | 74.9 (Claude 4 Opus) |
| AIME 2024 (Avg@64) | **69.6** | 61.3 (Gemini 2.5 Flash) |

**General**:

| Benchmark | K2 | Best Baseline |
|---|---|---|
| IFEval (Prompt Strict) | **89.8** | 88.0 (GPT-4.1) |
| Multi-Challenge | **54.1** | 49.0 (Claude 4 Opus) |
| FACTS Grounding | **88.5** | 86.6 (Gemini 2.5 Flash) |
| Arena Hard v2.0 Creative Writing | **85.0** | 72.8 (Gemini 2.5 Flash) |

**LMSYS Arena** (July 17, 2025): **#1 open-source model, #5 overall**, 3000+ user votes.

### 9.2 Pre-Training Evaluation (Kimi-K2-Base)

#### 9.2.1 Evaluation Settings

**Evaluation types**:
- **Perplexity-based**: MMLU, MMLU-Redux, GPQA-Diamond, HellaSwag, ARC-Challenge, C-Eval, CMMLU
- **Generation-based**: MMLU-Pro, SuperGPQA, TriviaQA, BBH, CSimpleQA, MATH, CMATH, GSM8K, GSM8K-Platinum, CRUXEval, LiveCodeBench, EvalPlus

**GPQA-Diamond**: Mean across 8 independent runs (high per-question variance).

#### 9.2.2 Key Results

SOTA on **10/12 English benchmarks**, all code benchmarks, 3/4 math benchmarks, all Chinese benchmarks:

| Category | Benchmark | K2-Base | Best Baseline |
|---|---|---|---|
| English | MMLU (5-shot) | **87.79** | 87.10 (DS-V3) |
| English | MMLU-Pro (5-shot) | **69.17** | 63.47 (Llama4-Maverick) |
| English | SimpleQA | **35.25** | 26.49 (DS-V3) |
| Code | EvalPlus | **80.33** | 66.04 (Qwen2.5-72B) |
| Code | LiveCodeBench v6 | **26.29** | 25.14 (Llama4-Maverick) |
| Math | MATH (4-shot) | **70.22** | 63.02 (Llama4-Maverick) |
| Chinese | CSimpleQA | **77.57** | 72.13 (DS-V3) |

### 9.3 Safety Evaluation

#### 9.3.1 Protocol

**Tool**: Promptfoo (automated red-teaming)

**Plugins** (5 categories, 30+ subcategories):
- **Harmful**: Graphic content, harassment, hate speech, insults, profanity, radicalization, self-harm, sexual content, ToxicChat
- **Criminal**: Chemical/biological weapons, child exploitation, copyright violations, cybercrime, illegal activities/drugs, indiscriminate weapons, IP violation, non-violent/violent/sex crimes
- **Misinformation**: Competitor endorsement, unsupervised contracts, excessive agency, hallucination, misinformation/disinformation, specialized advice, unsafe practices, imitation, overreliance, political opinions, religious sensitivity
- **Privacy**: Privacy violation, PII in API/database, direct PII exposure, PII in session data, PII via social engineering
- **Security**: ASCII smuggling, CyberSecEval, HarmBench, debug access, divergent repetition, DoNotAnswer, malicious code, Pliny, prompt extraction, reasoning DoS, tool discovery

**Attack strategies** (4 types):
- Basic
- Base64
- Prompt Injection
- Iterative Jailbreak
- Crescendo

**Test case generation**: 3 attack prompts per plugin-strategy combination. Language-aware: 6 prompts for bilingual combinations (English + Chinese).

**Review**: Multi-round human review with same reviewer per test set (consistency).

#### 9.3.2 Results Summary

| Plugin × Strategy | K2 Pass Rate | Notable Comparison |
|---|---|---|
| Harmful-Basic | 98.04% | Comparable to DS-R1 (99.02%), Qwen3 (98.53%) |
| Harmful-Iterative Jailbreak | 92.16% | Best among all models (DS-V3: 66.67%) |
| Criminal-Crescendo | 56.06% | Higher than DS-V3 (31.81%), DS-R1 (42.42%) |
| Privacy-Basic | 100% | All models at 100% |
| Security-Iterative Jailbreak | 43.90% | Same as DS-R1; Qwen3 best at 78.04% |

**Key observations**:
- Base64 strategy: near 100% pass rates (encoding transforms have minimal impact)
- Crescendo strategy: general drop across all models (strongest adversarial method)
- Complex attack strategies do not always outperform basic prompts (semantic corruption during transformation)

---

## 10. CONVERGENCE DYNAMICS AND TRAINING STABILITY

### 10.1 Loss Trajectory

- **15.5T tokens**: Smooth, monotonically decreasing loss curve
- **Zero loss spikes** (verified by per-step unsmoothed, unsubsampled loss plot)
- QK-Clip max logits: initially capped at $\tau = 100$, naturally decay to stable range after $\approx 30\%$ of training steps

### 10.2 QK-Clip Convergence Behavior

**Phase 1** (0–30% of training): Max logits capped at $\tau = 100$ by QK-Clip. The model learns under constrained attention dynamics.

**Phase 2** (30–100% of training): Max logits naturally decay below $\tau$ without adjustment. QK-Clip becomes a no-op ($\gamma_h = 1$ for all heads). The model has learned to self-regulate attention logits.

**Interpretation**: QK-Clip acts as a **transient stabilizer** during the critical early phase of Muon optimization at scale, then gracefully disengages as the model's attention dynamics stabilize. No hyperparameter adjustment required.

### 10.3 Learning Rate Schedule Visualization

```
LR
 ↑
2e-4 ─────────────────────┐
                           │ Cosine decay
                           │
                           └──────────────────┐
2e-5 ─────────────────────────────────────────┤
                                               │ Anneal
7e-6 ─────────────────────────────────────────┘
     ├──────────────────────┼─────────────────┼──┤
     0        10T          15.5T     15.9T  16T
                    Tokens
     │  Stage 1  │  Stage 2  │ Stage 3a │3b│
     │ Constant  │  Cosine   │ Anneal   │LC│
```

---

## 11. FAILURE MODES AND LIMITATIONS

### 11.1 Identified Failure Modes

| Failure Mode | Description | Domain |
|---|---|---|
| **Excessive token generation** | On hard reasoning tasks or unclear tool definitions, model generates excessive tokens → truncated outputs or incomplete tool calls | Agentic / reasoning |
| **Unnecessary tool use** | Performance degrades when tool use is enabled but not required | General tasks |
| **One-shot project limitations** | Success rate of one-shot prompting for complete software projects is lower than agentic coding frameworks | Software engineering |
| **Reward hacking (RL)** | Model may exploit prescriptive rubric loopholes; mitigated by prescriptive rubrics and hack-check layers | RL training |
| **Critic miscalibration** | Self-critique may drift without continuous grounding in verifiable signals; mitigated by closed-loop refinement | Self-critique RL |
| **Response length inflation** | RL training induces longer responses; mitigated by budget control and per-sample penalties | All RL domains |
| **Catastrophic forgetting** | Joint RL across diverse tasks may degrade specific capabilities; mitigated by PTX loss | RL training |
| **Safety gaps** | Complex attack strategies (Crescendo, Iterative Jailbreak) still achieve non-trivial bypass rates | Safety |

### 11.2 Architectural Trade-offs

| Decision | Gain | Cost |
|---|---|---|
| $H = 64$ (vs. 128) | 83% inference FLOPs reduction at 128k | 0.5–1.2% quality loss |
| $S = 48$ (vs. 64) | Lower infra complexity | Slightly higher loss |
| No expert grouping | Simpler routing | Potentially less structured expert specialization |
| 1 dense layer (vs. 3) | Lower parameter count | Marginally reduced capacity in early layers |
| EP = 16 (smallest) | Less EP communication, easier overlap | All 384 experts spread across only 16 devices |

---

## 12. COMPRESSION AND INFORMATION PRESERVATION

### 12.1 KV-Cache Compression (MLA)

**Compression equation**:

$$c_{\text{KV}} = X W_{\text{DKV}} \in \mathbb{R}^{n \times d_c}$$

**Information preservation**: The reconstruction error is bounded by:

$$\|K^h - c_{\text{KV}} W^h_{UK}\|_F \leq \sigma_{d_c+1}(X W_K^h) \cdot \|W^h_{UK}\|_F$$

where $\sigma_{d_c+1}$ is the $(d_c+1)$-th singular value of $XW_K^h$. When $d_c$ captures the effective rank of $K$, information loss is negligible.

**Compression ratio**:

$$\rho_{\text{MLA}} = \frac{d_c + d_R}{2Hd_h}$$

For typical values ($d_c = 512$, $d_R = 64$, $H = 64$, $d_h = 112$):

$$\rho_{\text{MLA}} = \frac{576}{14336} \approx 0.040 \quad (\sim 25\times \text{ compression})$$

### 12.2 Activation Compression (FP8)

**Format**: FP8-E4M3 in $1 \times 128$ tiles with FP32 per-tile scales.

**Quantization**:

$$\hat{x}_i = \text{clamp}\left(\text{round}\left(\frac{x_i}{s}\right), -\text{max}_{\text{E4M3}}, \text{max}_{\text{E4M3}}\right)$$

where $s = \frac{\max_{j \in \text{tile}}|x_j|}{\text{max}_{\text{E4M3}}}$ is the per-tile scale factor.

**Reconstruction**:

$$\tilde{x}_i = \hat{x}_i \cdot s$$

**Error bound** (per tile):

$$|x_i - \tilde{x}_i| \leq \frac{s}{2} \cdot 2^{-3} = \frac{\max_{j}|x_j|}{2 \cdot \text{max}_{\text{E4M3}} \cdot 8}$$

**Empirical validation**: No measurable loss increase in small-scale experiments.

### 12.3 Sparsity as Implicit Compression

MoE activates only $k = 8$ out of $E = 384$ experts per token:

$$\text{Computation compression ratio} = \frac{k + 1}{E + 1} = \frac{9}{385} \approx 0.023$$

(including 1 shared expert)

Information preservation: The routing function $g(x)$ selects experts maximizing the gate score, ensuring the most relevant parameters are activated for each input.

---

## 13. DEPLOYMENT CONSTRAINTS

### 13.1 Serving Requirements

| Constraint | Value/Consideration |
|---|---|
| **Total parameter storage** | $1.04\text{T} \times 2 \text{ bytes (BF16)} = 2.08\text{ TB}$ |
| **Minimum GPU count** | $\geq 26 \times$ H800 (80GB) for parameters alone |
| **KV-cache (128k context)** | $\sim L \times (d_c + d_R) \times 131072 \times 2$ bytes per request |
| **Expert parallelism** | 16-way EP minimum (384 experts) |
| **Batch concurrent requests** | Limited by KV-cache memory and EP communication |
| **Multi-turn agentic serving** | Persistent KV-cache across turns; tool execution latency |
| **Latency target** | First-token latency dominated by MLA decoding + routing |
| **Throughput** | 32.6B active params per token; MoE routing adds all-to-all communication |

### 13.2 Agentic Deployment Considerations

| Concern | Design Choice |
|---|---|
| Tool execution latency | Heavy environments as dedicated scalable services |
| Long trajectories | Partial rollout / streaming |
| Concurrent sessions | 10,000+ sandbox instances (Kubernetes) |
| Context management | 128k window with YaRN; truncation for longer inputs |
| Output token budget | Per-task maximum budget enforcement |

### 13.3 Model Release

| Artifact | Availability |
|---|---|
| **Kimi-K2-Base** | Open-source (HuggingFace) |
| **Kimi-K2-Instruct** | Open-source (HuggingFace) |
| **Checkpoint Engine** | Open-source (GitHub) |
| **Training code** | Not released |
| **Data** | Not released |

---

## 14. SUMMARY OF TENSOR DIMENSIONS AND PARAMETERIZATION

| Symbol | Description | Kimi K2 Value |
|---|---|---|
| $L$ | Number of layers | 61 |
| $d_{\text{model}}$ | Hidden dimension | 7168 |
| $d_{\text{expert}}$ | Expert FFN hidden dim | 2048 |
| $H$ | Number of attention heads | 64 |
| $d_h$ | Per-head dimension | $d_{\text{model}} / H = 112$ |
| $E$ | Total experts per MoE layer | 384 |
| $k$ | Active experts per token | 8 |
| $S$ | Sparsity ratio | 48 |
| $d_c$ | KV latent dimension (MLA) | Architecture-specific |
| $d_R$ | Rotary dimension | Architecture-specific |
| $d'_c$ | Query latent dimension | Architecture-specific |
| $N_{\text{total}}$ | Total parameters | 1.04T |
| $N_{\text{active}}$ | Activated parameters/token | 32.6B |
| $T$ | Pre-training tokens | 15.5T |
| $\tau$ | QK-Clip threshold | 100 |
| $\lambda$ | Weight decay | 0.1 |
| $\eta_{\text{max}}$ | Peak learning rate | $2 \times 10^{-4}$ |
| $\eta_{\text{min}}$ | Minimum learning rate | $7 \times 10^{-6}$ |
| Batch size | Global batch size | 67M tokens |

---

## 15. CROSS-STAGE DEPENDENCY GRAPH

```
                    ┌──────────────┐
                    │ Data Pipeline │
                    │ (§2)         │
                    │ - Web/Code/  │
                    │   Math/Know  │
                    │ - Rephrasing │
                    └──────┬───────┘
                           │ 15.5T tokens
                    ┌──────▼───────┐
                    │  Pre-training │
                    │  (§5.1-5.3)  │
                    │  MuonClip    │
                    │  WSD Schedule│
                    └──────┬───────┘
                           │ Base checkpoint
                    ┌──────▼───────┐
                    │     SFT      │
                    │   (§5.4)     │
                    │ + Agentic    │
                    │   Data Synth │
                    └──────┬───────┘
                           │ SFT checkpoint
                    ┌──────▼───────┐
                    │  Joint RL    │
                    │  (§5.5)      │
                    │  RLVR +      │
                    │  Self-Critique│
                    │  + PTX + BC  │
                    └──────┬───────┘
                           │ Instruct checkpoint
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │Evaluation│ │ Safety   │ │Deployment│
        │ (§9)     │ │ (§9.3)   │ │ (§13)    │
        └──────────┘ └──────────┘ └──────────┘
```