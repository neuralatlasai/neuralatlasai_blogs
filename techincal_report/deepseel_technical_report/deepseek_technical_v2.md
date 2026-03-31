# DeepSeek-V3: End-to-End Technical Report

---

## 1. Model Architecture

### 1.1 Definition and Problem Formulation

**Formal Definition.** DeepSeek-V3 is a Mixture-of-Experts (MoE) autoregressive Transformer language model with $671\text{B}$ total parameters, of which $37\text{B}$ are activated per token. The model maps an input sequence of discrete tokens $\mathbf{x} = (x_1, x_2, \ldots, x_T)$ drawn from vocabulary $\mathcal{V}$ with $|\mathcal{V}| = 128\text{K}$ to a probability distribution over subsequent tokens.

**Objective.** Minimize the expected negative log-likelihood over a corpus $\mathcal{D}$:

$$
\mathcal{L}_{\text{NTP}} = -\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})
$$

augmented by a Multi-Token Prediction (MTP) auxiliary objective and a complementary sequence-wise balance loss.

**Boundary Conditions.**
- Sequence length: $T \leq 4096$ during pre-training, extended to $T \leq 128\text{K}$ post-extension.
- Per-token activated parameters: $37\text{B}$ out of $671\text{B}$.
- Routed experts per token: $K_r = 8$ out of $N_r = 256$.
- Node-limited routing: each token dispatched to at most $M = 4$ nodes.
- Shared experts: $N_s = 1$.

**Invariants.**
- Causal masking is strictly enforced across all attention and MTP prediction depths.
- No token dropping during training or inference due to effective load balancing.
- KV cache during inference stores only compressed latent vectors $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ and decoupled RoPE keys $\mathbf{k}_t^R \in \mathbb{R}^{d_h^R}$.

**Assumptions.**
- Data is independently sampled and tokenized using Byte-level BPE with $128\text{K}$ vocabulary.
- FP8 mixed precision training preserves model quality within $0.25\%$ relative loss error compared to BF16.
- Expert load is globally balanced at the batch level via an auxiliary-loss-free bias mechanism.

<img src="assets/deepseek_v3_technical_architecture-03.png" alt="Formal DeepSeek-V3 system specification showing the RMSNorm-MLA-RMSNorm-MoE stack together with the optimization objective and auxiliary balance term" width="100%" />

*Figure. Formal system specification for DeepSeek-V3, linking the report's objective, routing regularization, and stacked RMSNorm-MLA-MoE structure before the component-level derivations.*

---

### 1.2 Multi-Head Latent Attention (MLA)

<img src="assets/deepseek_v3_technical_architecture-04.png" alt="Comparison of standard multi-head attention KV cache storage against DeepSeek-V3 multi-head latent attention with compressed latent cache and decoupled RoPE keys" width="100%" />

*Figure. MLA compresses the KV pathway into a compact latent cache plus a small RoPE branch, making the memory-reduction argument visually explicit before the tensor equations.*

#### 1.2.1 Core Mechanism

MLA performs low-rank joint compression of keys and values into a shared latent space, dramatically reducing KV cache memory while preserving MHA-equivalent representational capacity.

**Notation.**
- $d = 7168$: embedding dimension
- $n_h = 128$: number of attention heads
- $d_h = 128$: per-head dimension
- $d_c = 512$: KV compression dimension ($d_c \ll d_h \cdot n_h = 16384$)
- $d_c' = 1536$: query compression dimension ($d_c' \ll d_h \cdot n_h$)
- $d_h^R = 64$: per-head dimension for decoupled RoPE keys/queries
- $\mathbf{h}_t \in \mathbb{R}^d$: attention input for token $t$

#### 1.2.2 KV Compression

**Down-projection into latent space:**

$$
\mathbf{c}_t^{KV} = W^{DKV}\mathbf{h}_t, \quad W^{DKV} \in \mathbb{R}^{d_c \times d}
$$

**Up-projection for keys and values:**

$$
\mathbf{k}_t^{C} = W^{UK}\mathbf{c}_t^{KV}, \quad W^{UK} \in \mathbb{R}^{d_h n_h \times d_c}
$$

$$
\mathbf{v}_t^{C} = W^{UV}\mathbf{c}_t^{KV}, \quad W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}
$$

**Decoupled RoPE key:**

$$
\mathbf{k}_t^R = \text{RoPE}(W^{KR}\mathbf{h}_t), \quad W^{KR} \in \mathbb{R}^{d_h^R \times d}
$$

**Composite key per head $i$:**

$$
\mathbf{k}_{t,i} = [\mathbf{k}_{t,i}^{C};\; \mathbf{k}_t^R]
$$

**KV Cache Invariant.** During inference, only $\mathbf{c}_t^{KV} \in \mathbb{R}^{512}$ and $\mathbf{k}_t^R \in \mathbb{R}^{64}$ are cached per token, yielding cache size $576$ elements per token versus $2 \times d_h \times n_h = 32768$ for standard MHA — a compression ratio of approximately $\mathbf{56.9\times}$.

**KV Cache Memory per Token:**

$$
\text{MLA: } (d_c + d_h^R) \cdot \text{bytes} = (512 + 64) \cdot 2 = 1152 \text{ bytes (BF16)}
$$

$$
\text{MHA: } 2 \cdot d_h \cdot n_h \cdot 2 = 65536 \text{ bytes (BF16)}
$$

#### 1.2.3 Query Compression

**Down-projection:**

$$
\mathbf{c}_t^Q = W^{DQ}\mathbf{h}_t, \quad W^{DQ} \in \mathbb{R}^{d_c' \times d}
$$

**Up-projection for content queries:**

$$
\mathbf{q}_{t}^{C} = W^{UQ}\mathbf{c}_t^Q, \quad W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}
$$

**Decoupled RoPE queries:**

$$
\mathbf{q}_{t}^{R} = \text{RoPE}(W^{QR}\mathbf{c}_t^Q), \quad W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}
$$

**Composite query per head $i$:**

$$
\mathbf{q}_{t,i} = [\mathbf{q}_{t,i}^{C};\; \mathbf{q}_{t,i}^{R}]
$$

#### 1.2.4 Attention Computation

$$
o_{t,i} = \sum_{j=1}^{t} \text{Softmax}_j\left(\frac{\mathbf{q}_{t,i}^\top \mathbf{k}_{j,i}}{\sqrt{d_h + d_h^R}}\right) \mathbf{v}_{j,i}^{C}
$$

$$
\mathbf{u}_t = W^O [\mathbf{o}_{t,1};\; \mathbf{o}_{t,2};\; \ldots;\; \mathbf{o}_{t,n_h}], \quad W^O \in \mathbb{R}^{d \times d_h n_h}
$$

**Design Rationale — Decoupled RoPE.**
- RoPE is applied to separate, low-dimensional projections $\mathbf{k}_t^R$ and $\mathbf{q}_{t,i}^R$ rather than being mixed into the compressed latent.
- This decoupling is essential because RoPE is position-dependent and non-linear; applying it after up-projection from a shared compressed latent would entangle positional information with the compression, preventing efficient KV caching (the cached $\mathbf{c}_t^{KV}$ must remain position-independent for the content pathway).

**Additional Architectural Details.**
- Additional RMSNorm layers are applied after the compressed latent vectors $\mathbf{c}_t^{KV}$ and $\mathbf{c}_t^Q$.
- Additional learned scaling factors are applied at width bottlenecks (compression/decompression boundaries).

**Complexity Analysis.**
- Attention computation: $O(T^2 \cdot (d_h + d_h^R) \cdot n_h)$ per layer, identical asymptotically to MHA.
- KV cache memory: $O(T \cdot (d_c + d_h^R) \cdot L)$ across $L$ layers versus $O(T \cdot 2 d_h n_h \cdot L)$ for MHA.
- Query/KV down-projection reduces activation memory during training by factor $\approx d_h n_h / d_c$ and $d_h n_h / d_c'$.

---

### 1.3 DeepSeekMoE with Auxiliary-Loss-Free Load Balancing

<img src="assets/deepseek_v3_technical_architecture-05.png" alt="DeepSeekMoE sparse activation diagram with 256 routed experts, one shared expert, sigmoid gating, and node-limited routing constraints" width="100%" />

*Figure. DeepSeekMoE routing structure with sparse expert activation, shared expertise, and node-limited dispatch, matching the gating and load-balancing mechanics defined in this section.*

#### 1.3.1 Basic DeepSeekMoE Architecture

**FFN input:** $\mathbf{u}_t \in \mathbb{R}^d$ (output of MLA + residual + RMSNorm).

**FFN output:**

$$
\mathbf{h}_t' = \sum_{i=1}^{N_s} \text{FFN}_i^{(s)}(\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \cdot \text{FFN}_i^{(r)}(\mathbf{u}_t)
$$

where $N_s = 1$ shared expert, $N_r = 256$ routed experts, and each expert FFN has intermediate hidden dimension $2048$.

**Routing (Gating).**

Token-to-expert affinity using sigmoid:

$$
s_{i,t} = \sigma(\mathbf{e}_i^\top \mathbf{u}_t)
$$

where $\mathbf{e}_i$ is the centroid vector (routing weight) of the $i$-th routed expert.

Top-$K_r$ selection ($K_r = 8$):

$$
g_{i,t} = \begin{cases} \dfrac{s_{i,t}}{\sum_{j \in \text{TopK}(s_{\cdot,t}, K_r)} s_{j,t}} & \text{if } i \in \text{TopK}(s_{\cdot,t}, K_r) \\[6pt] 0 & \text{otherwise} \end{cases}
$$

**Key Difference from DeepSeek-V2.** DeepSeek-V3 uses **sigmoid** activation for affinity scores (rather than softmax) and normalizes only among the selected top-$K_r$ experts.

**Node-Limited Routing.** Each token is dispatched to at most $M = 4$ nodes. The target nodes are selected by ranking the sum of the highest $\frac{K_r}{M}$ affinity scores among experts on each node. This bounds inter-node communication cost.

**Individual Expert Architecture.** Each FFN expert (shared or routed) is a SwiGLU FFN:

$$
\text{FFN}(\mathbf{x}) = (W_{\text{up}}\mathbf{x} \odot \text{SiLU}(W_{\text{gate}}\mathbf{x})) W_{\text{down}}
$$

with $W_{\text{up}}, W_{\text{gate}} \in \mathbb{R}^{2048 \times d}$, $W_{\text{down}} \in \mathbb{R}^{d \times 2048}$.

**MoE Layer Placement.** All FFN layers except the first 3 (layers 0, 1, 2) are replaced by MoE layers. The first 3 layers use dense FFNs.

#### 1.3.2 Auxiliary-Loss-Free Load Balancing

**Problem.** Conventional auxiliary losses to encourage load balance degrade model quality when made sufficiently large to ensure balance.

**Mechanism.** A per-expert bias term $b_i$ is introduced and added to affinity scores for routing decisions only:

$$
g_{i,t}' = \begin{cases} g_{i,t} & \text{(gating value computation unchanged)} \end{cases}
$$

$$
\text{Routing decision: TopK}\bigl(\{s_{i,t} + b_i\}_{i=1}^{N_r},\; K_r\bigr)
$$

**Critical invariant:** The bias $b_i$ affects only the top-K selection, not the gating value $g_{i,t}$. The gating value is computed from the original affinity score $s_{i,t}$ without bias.

**Bias Update Rule (per training step):**
- Monitor expert load across the entire batch.
- If expert $i$ is overloaded: $b_i \leftarrow b_i - \gamma$
- If expert $i$ is underloaded: $b_i \leftarrow b_i + \gamma$
- $\gamma = 0.001$ (bias update speed) for first 14.3T tokens; $\gamma = 0.0$ for final 500B tokens.

**Balancing Scope:** Batch-wise rather than sequence-wise. This permits experts to specialize per domain (e.g., code, math, language) since balance is not enforced within individual sequences.

#### 1.3.3 Complementary Sequence-Wise Auxiliary Loss

Despite the primary auxiliary-loss-free strategy, a very small sequence-wise balance loss prevents extreme per-sequence imbalance:

$$
\mathcal{L}_{\text{Bal}} = \alpha \sum_{i=1}^{N_r} f_i \cdot P_i
$$

where:

$$
f_i = \frac{N_r}{K_r \cdot T} \sum_{t=1}^{T} \mathbb{1}(i \in \text{TopK}(s_{\cdot,t}, K_r))
$$

$$
P_i = \frac{1}{T}\sum_{t=1}^{T} s_{i,t}
$$

- $\alpha = 0.0001$ (extremely small to avoid quality degradation)
- $T$: number of tokens in a sequence
- $f_i$: fraction of tokens routed to expert $i$ (rescaled)
- $P_i$: mean affinity to expert $i$

**No Token Dropping.** Due to effective load balancing, no tokens are dropped during training or inference.

#### 1.3.4 Expert Specialization Analysis

Batch-wise balancing (auxiliary-loss-free) produces significantly greater expert specialization patterns compared to sequence-wise auxiliary losses. Ablation experiments show:
- 1B MoE: validation loss 2.258 (sequence-wise aux loss) vs. 2.253 (aux-loss-free) vs. 2.253 (batch-wise aux loss)
- 3B MoE: validation loss 2.085 (sequence-wise) vs. 2.080 (aux-loss-free) vs. 2.080 (batch-wise)

This confirms that the performance advantage stems from the batch-wise balancing scope, not specifically from the bias mechanism.

**Potential Challenges of Batch-Wise Balancing:**
1. Load imbalance within individual sequences or small batches — mitigated by large-scale expert parallelism and data parallelism guaranteeing large micro-batch sizes.
2. Domain-shift-induced load imbalance during inference — mitigated by redundant expert deployment strategy.

---

### 1.4 Multi-Token Prediction (MTP)

<img src="assets/deepseek_v3_technical_architecture-07.png" alt="Multi-token prediction module showing concatenation of previous-depth hidden state with future token embedding followed by a transformer block and shared output head" width="100%" />

*Figure. Sequential MTP module used in DeepSeek-V3, clarifying how future-token embeddings and previous-depth representations are fused before the auxiliary prediction head.*

#### 1.4.1 Definition

MTP extends the training objective from next-token prediction to predicting $D$ additional future tokens at each position, while maintaining the complete causal chain at each prediction depth.

**Key Design Difference from Gloeckle et al. (2024):** Instead of parallel independent output heads, DeepSeek-V3 uses **sequential** MTP modules that maintain causal chain dependencies.

#### 1.4.2 MTP Module Architecture

The $k$-th MTP module ($k = 1, \ldots, D$) consists of:
- Shared embedding layer $\text{Emb}(\cdot)$ (shared with main model)
- Shared output head $\text{OutHead}(\cdot)$ (shared with main model)
- Transformer block $\text{TRM}_k(\cdot)$ (unique per depth)
- Projection matrix $M_k \in \mathbb{R}^{d \times 2d}$ (unique per depth)

**Forward pass for $k$-th depth, $i$-th token:**

**Step 1: Combine previous-depth representation with current-depth token embedding:**

$$
\mathbf{h}_i'^{(k)} = M_k [\mathbf{h}_i^{(k-1)};\; \text{Emb}(t_{i+k})]
$$

where:
- $\mathbf{h}_i^{(k-1)} \in \mathbb{R}^d$: representation at depth $k-1$ (for $k=1$, this is the main model's output representation)
- $\text{Emb}(t_{i+k}) \in \mathbb{R}^d$: embedding of the $(i+k)$-th token
- $[\cdot;\cdot]$: concatenation yielding $\mathbb{R}^{2d}$
- $M_k$: linear projection back to $\mathbb{R}^d$

**Step 2: Apply Transformer block with causal masking:**

$$
\mathbf{h}_{1:T}^{(k)} = \text{TRM}_k(\mathbf{h}_{1:T}'^{(k)})
$$

**Step 3: Compute prediction distribution:**

$$
P_i^{(k)} = \text{OutHead}(\mathbf{h}_i^{(k)}) \in \mathbb{R}^{|\mathcal{V}|}
$$

where $\text{OutHead}(\cdot)$ applies a linear projection to logits followed by $\text{Softmax}(\cdot)$.

#### 1.4.3 MTP Training Objective

Per-depth cross-entropy loss:

$$
\mathcal{L}_{\text{MTP}}^k = -\frac{1}{T}\sum_{i=1}^{T} \log P_i^{(k)}[t_{i+k}]
$$

Overall MTP loss:

$$
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D}\sum_{k=1}^{D}\mathcal{L}_{\text{MTP}}^k
$$

**Hyper-parameters:**
- $D = 1$ (predict one additional token beyond next-token)
- $\lambda = 0.3$ for first 10T tokens; $\lambda = 0.1$ for remaining 4.8T tokens

#### 1.4.4 Total Training Objective

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{MTP}} + \mathcal{L}_{\text{Bal}}
$$

$$
= -\frac{1}{T}\sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}) + \frac{\lambda}{D}\sum_{k=1}^D \mathcal{L}_{\text{MTP}}^k + \alpha\sum_{i=1}^{N_r} f_i P_i
$$

#### 1.4.5 MTP During Inference

- MTP modules are **discarded** during inference; the main model functions independently.
- Optionally, MTP modules can be repurposed for **speculative decoding** to reduce generation latency.
- The MTP Transformer blocks share a design similar to EAGLE for speculative decoding.

#### 1.4.6 Ablation Results

| Scale | Total Params | Tokens | Benchmark Examples (HumanEval Pass@1) | Without MTP | With MTP |
|---|---|---|---|---|---|
| Small MoE | 15.7B | 1.33T | HumanEval | 20.7 | 26.8 |
| Large MoE | 228.7B | 540B | HumanEval | 44.5 | 53.7 |

MTP consistently improves performance across benchmarks, with no additional inference cost (modules discarded).

---

### 1.5 Overall Model Configuration Summary

<img src="assets/deepseek_v3_engineering_blueprint-03.png" alt="DeepSeek-V3 61-layer macro architecture pipeline with embeddings, dense prefix layers, DeepSeekMoE stack, and output layers" width="100%" />

*Figure. The 61-layer macro architecture pipeline grounds the configuration table in a concrete stage-by-stage view of embeddings, dense early layers, MoE depth, and output projection.*

| Parameter | Value |
|---|---|
| Transformer layers $L$ | 61 |
| Hidden dimension $d$ | 7168 |
| Attention heads $n_h$ | 128 |
| Per-head dimension $d_h$ | 128 |
| KV compression dim $d_c$ | 512 |
| Query compression dim $d_c'$ | 1536 |
| Decoupled RoPE per-head dim $d_h^R$ | 64 |
| Shared experts $N_s$ | 1 |
| Routed experts $N_r$ | 256 |
| Activated routed experts $K_r$ | 8 |
| Expert intermediate dim | 2048 |
| MTP depth $D$ | 1 |
| Max nodes per token $M$ | 4 |
| Vocabulary size $|\mathcal{V}|$ | 128K |
| Total parameters | 671B |
| Activated parameters per token | 37B |
| MoE layer placement | All FFN layers except first 3 |
| Dense FFN layers | Layers 0, 1, 2 |
| Initialization std | 0.006 |
| Normalization | RMSNorm (additional norms after compressed latents) |

---

## 2. Data Pipeline

### 2.1 Data Construction

<img src="assets/deepseek_v3_architecture-10.png" alt="Pre-training lifecycle showing learning-rate schedule, batch-size ramp, context extension phases, and the raw-document to packed-token data preprocessing flow" width="100%" />

*Figure. Pre-training schedule and data preprocessing flowchart, aligning corpus construction, FIM injection, tokenization, packing, and context-extension timing in one view.*

#### 2.1.1 Corpus Composition

- **Total tokens:** 14.8T diverse, high-quality tokens
- **Language distribution:** Majority English and Chinese, with expanded multilingual coverage
- **Domain enhancements over DeepSeek-V2:** Increased ratio of mathematical and programming samples

#### 2.1.2 Processing Pipeline

**Objectives:**
- Minimize redundancy while maintaining corpus diversity
- Preserve document integrity via document packing
- Support Fill-in-Middle (FIM) capability

**Deduplication:** Refined pipeline to reduce redundancy (exact methods inherited from prior work, optimized for scale).

**Document Packing:** Following Ding et al. (2024), documents are packed into training sequences for data integrity. **No cross-sample attention masking** is applied during pre-training (attention masking is not used across packed documents).

#### 2.1.3 Fill-in-Middle (FIM) Strategy

FIM is applied using the Prefix-Suffix-Middle (PSM) framework at document level as part of pre-packing:

$$
\text{PSM}(\text{prefix}, \text{suffix}, \text{middle}) = \langle \text{PRE}\rangle\; \text{prefix}\; \langle \text{SUF}\rangle\; \text{suffix}\; \langle \text{MID}\rangle\; \text{middle}
$$

- Applied at rate $0.1$
- Applied at document level before packing
- Validated in DeepSeekCoder-V2: FIM does not compromise next-token prediction capability

#### 2.1.4 Tokenizer

- **Type:** Byte-level BPE
- **Vocabulary size:** 128K tokens
- **Modifications:**
  - Pre-tokenizer optimized for multilingual compression efficiency
  - Combined punctuation + line break tokens introduced
  - **Token boundary bias mitigation:** Random splitting of combined tokens (punctuation + line breaks) during training at a certain proportion to expose the model to edge cases (e.g., multi-line prompts without terminal line breaks, common in few-shot evaluation)

### 2.2 Pseudo-Algorithm: Data Preprocessing Pipeline

```
ALGORITHM: DataPreprocessing
INPUT: Raw document corpus D_raw
OUTPUT: Packed tokenized training sequences

1. DEDUPLICATE(D_raw) → D_dedup
   - Apply deduplication pipeline to minimize redundancy
   - Maintain corpus diversity

2. For each document d in D_dedup:
   a. With probability 0.1:
      - Split d into (prefix, middle, suffix) at random position
      - d_formatted ← PSM(prefix, suffix, middle)
   b. Else:
      - d_formatted ← d

3. TOKENIZE(d_formatted) using Byte-level BPE (128K vocab)
   - With probability p_split, randomly split combined punctuation+linebreak tokens

4. PACK tokenized documents into sequences of length T_max = 4096
   - Document packing method (Ding et al., 2024)
   - No cross-sample attention masking

5. RETURN packed sequences
```

---

## 3. Infrastructure and Training Framework

### 3.1 Compute Cluster

<img src="assets/deepseek_v3_engineering_blueprint-10.png" alt="Parallelism topology for 2048 H800 GPUs showing expert parallelism across nodes, ZeRO-1 data parallelism, and 16-way pipeline parallelism without tensor parallelism" width="100%" />

*Figure. Training-parallelism topology across 2048 H800 GPUs, showing why DeepSeek-V3 combines pipeline, expert, and ZeRO-style data parallelism while avoiding tensor parallelism.*

- **GPUs:** 2048 × NVIDIA H800
- **Intra-node:** 8 GPUs per node, connected via NVLink + NVSwitch (160 GB/s bandwidth)
- **Inter-node:** InfiniBand (IB) interconnects (50 GB/s bandwidth)
- **Bandwidth ratio:** NVLink/IB $\approx 3.2\times$

### 3.2 Parallelism Strategy

| Parallelism Dimension | Configuration |
|---|---|
| Pipeline Parallelism (PP) | 16-way |
| Expert Parallelism (EP) | 64-way across 8 nodes |
| Data Parallelism (DP) | ZeRO-1 |
| Tensor Parallelism (TP) | **Not used** during training (memory optimized to avoid TP) |

**No Tensor Parallelism** is used during training, eliminating TP communication overhead through meticulous memory optimization.

### 3.3 DualPipe: Efficient Pipeline Parallelism

<img src="assets/deepseek_v3_engineering_blueprint-11.png" alt="DualPipe bidirectional scheduling diagram comparing standard pipeline bubbles against overlapped forward backward and all-to-all communication" width="100%" />

*Figure. DualPipe overlaps forward, backward, and all-to-all phases from both pipeline ends to suppress idle bubbles and preserve throughput at large MoE scale.*

#### 3.3.1 Problem

Cross-node expert parallelism introduces computation-to-communication ratio of approximately $1:1$, making naive scheduling highly inefficient.

#### 3.3.2 Core Mechanism

Each chunk (forward or backward) is decomposed into four components:
1. **Attention** computation
2. **All-to-all dispatch** (send tokens to expert-hosting GPUs)
3. **MLP** (expert) computation
4. **All-to-all combine** (gather expert outputs back)

For backward chunks, attention and MLP are further split into:
- **Backward for input** (Dgrad)
- **Backward for weights** (Wgrad)

Plus a **PP communication** component for pipeline stage boundary transfers.

#### 3.3.3 Overlapping Strategy

For a pair of forward and backward chunks, components are rearranged with manually adjusted SM allocation ratios for communication vs. computation, ensuring:
- **All-to-all communication is fully hidden** behind computation
- **PP communication is fully hidden** behind computation
- Bidirectional pipeline scheduling feeds micro-batches from both ends simultaneously

#### 3.3.4 Complexity Comparison

| Method | Pipeline Bubble | Parameter Memory | Activation Memory |
|---|---|---|---|
| 1F1B | $(PP - 1)(F + B)$ | $1 \times$ | $PP \times$ |
| ZB1P | $(PP - 1)(F + B - 2W)$ | $1 \times$ | $PP \times$ |
| **DualPipe** | $\left(\frac{PP}{2} - 1\right)(F\&B + B - 3W)$ | $2 \times$ | $(PP + 1) \times$ |

where:
- $F$: forward chunk time
- $B$: full backward chunk time
- $W$: backward-for-weights time
- $F\&B$: overlapped forward+backward time

**DualPipe Advantages:**
- Significantly fewer pipeline bubbles
- Only $+1\times$ peak activation memory over ZB1P
- Two copies of model parameters required, but minimal impact due to large EP size
- Requires only micro-batches and pipeline stages divisible by 2 (unlike Chimera which requires micro-batches divisible by pipeline stages)
- Neither bubbles nor activation memory increase with more micro-batches

### 3.4 Cross-Node All-to-All Communication

<img src="assets/deepseek_v3_architecture-08.png" alt="Cross-node communication kernel showing NVLink and InfiniBand dispatch combine paths with specialized warp groups and overlap strategy" width="100%" />

*Figure. Cross-node communication kernel for expert dispatch and combine, highlighting the warp-specialized IB and NVLink path that keeps all-to-all overhead near zero.*

#### 3.4.1 Design Constraints

- Each token dispatched to at most 4 nodes (limiting IB traffic)
- Average 3.2 experts selected per node (matching NVLink/IB bandwidth ratio)
- Can scale to maximum 13 experts per token ($4 \times 3.2$) at same communication cost

#### 3.4.2 Implementation

- **20 SMs** allocated for communication (out of 132 available on H800)
- **10 communication channels** using warp specialization
- **Dispatching:** 3 specialized warp groups for (1) IB sending, (2) IB-to-NVLink forwarding, (3) NVLink receiving
- **Combining:** 3 specialized warp groups for (1) NVLink sending, (2) NVLink-to-IB forwarding + accumulation, (3) IB receiving + accumulation
- Dynamic warp allocation based on actual workload
- Custom PTX instructions with auto-tuned communication chunk sizes to minimize L2 cache interference

**Overlap Guarantee:** As model scales, maintaining constant computation-to-communication ratio ensures near-zero all-to-all overhead with fine-grained cross-node experts.

### 3.5 Memory Optimization

| Technique | Mechanism | Impact |
|---|---|---|
| RMSNorm recomputation | Recompute all RMSNorm during backprop | Eliminates RMSNorm output activation storage |
| MLA up-projection recomputation | Recompute MLA up-projections during backprop | Eliminates up-projection activation storage |
| EMA in CPU | Store Exponential Moving Average params in CPU, async update | Zero GPU memory/time overhead for EMA |
| Shared Embedding + Output Head | DualPipe places shallowest + deepest layers on same PP rank | Physical sharing of parameters/gradients for MTP modules |

---

## 4. FP8 Mixed Precision Training

### 4.1 Framework Overview

<img src="assets/deepseek_v3_architecture-09.png" alt="FP8 mixed precision execution stack showing preserved BF16 modules, FP8 GEMMs, tile-wise and block-wise quantization, and promoted accumulation" width="100%" />

*Figure. FP8 training overview with the exact split between FP8 GEMMs and higher-precision operators, plus the promoted-accumulation path used to control numerical error.*

**Principle:** Most compute-dense operations (GEMMs) executed in FP8; numerically sensitive operations retained in BF16/FP32.

**FP8 Operations (3 GEMMs per Linear):**
1. **Fprop:** Forward pass GEMM — FP8 inputs → BF16/FP32 output
2. **Dgrad:** Activation backward — FP8 inputs → BF16/FP32 output
3. **Wgrad:** Weight backward — FP8 inputs → BF16/FP32 output

**Higher-Precision Operations (retained at BF16/FP32):**
- Embedding module
- Output head
- MoE gating modules
- All normalization operators (RMSNorm)
- Attention operators (softmax, scaling)
- Master weights, weight gradients, optimizer states (FP32)

**Validation:** Relative loss error of FP8 vs. BF16 training remains consistently below $0.25\%$.

### 4.2 Fine-Grained Quantization

<img src="assets/deepseek_v3_technical_architecture-10.png" alt="Fine-grained quantization diagram contrasting activation tile scaling and weight block scaling with tensor-core and CUDA-core promotion path" width="100%" />

*Figure. Fine-grained quantization at tile and block granularity, showing how DeepSeek-V3 localizes scale factors and uses promoted accumulation to stabilize FP8 execution.*

#### 4.2.1 Problem with Standard Per-Tensor Scaling

Standard FP8 quantization scales by the global maximum absolute value, making it highly sensitive to outliers and wasting dynamic range for the majority of values.

#### 4.2.2 Tile-Wise and Block-Wise Quantization

**Activations:** Group and scale on a $1 \times 128$ tile basis (per token per 128 channels):

$$
\mathbf{x}_{\text{FP8}}^{(t, g)} = \text{Quantize}\left(\mathbf{x}^{(t, g)}, \; s^{(t,g)} = \frac{\max|\mathbf{x}^{(t,g)}|}{\text{FP8}_{\max}}\right)
$$

where $g$ indexes the group of 128 channels for token $t$.

**Weights:** Group and scale on a $128 \times 128$ block basis (per 128 input channels × 128 output channels):

$$
\mathbf{W}_{\text{FP8}}^{(b_r, b_c)} = \text{Quantize}\left(\mathbf{W}^{(b_r, b_c)}, \; s^{(b_r, b_c)} = \frac{\max|\mathbf{W}^{(b_r, b_c)}|}{\text{FP8}_{\max}}\right)
$$

**Consistency with microscaling formats** (Rouhani et al., 2023b); anticipated native hardware support in NVIDIA Blackwell GPUs.

### 4.3 Increased Accumulation Precision

#### 4.3.1 Problem

H800 Tensor Core FP8 GEMM accumulation retains only $\sim$14 bits, far below FP32 precision. For large inner dimension $K$ (e.g., $K = 4096$), this causes up to $\sim 2\%$ maximum relative error.

#### 4.3.2 Solution: Promotion to CUDA Cores

During MMA (Matrix Multiply-Accumulate):
1. Accumulate partial results on Tensor Cores using limited bit width
2. At interval $N_C = 128$ elements (4 WGMMAs), copy partial results to **FP32 registers on CUDA Cores**
3. Perform full-precision FP32 accumulation on CUDA Cores
4. Apply per-group dequantization scaling factors during CUDA Core accumulation (negligible overhead)

**Concurrent WGMMA Overlap:** On H800, two warpgroups persist concurrently — while one performs promotion, the other executes MMA, maintaining high Tensor Core utilization.

### 4.4 FP8 Format Choice

**All tensors use E4M3** (4-bit exponent, 3-bit mantissa) — both Fprop and backward passes.

**Justification:** Fine-grained tile/block-wise scaling effectively shares exponent bits among grouped elements, compensating for E4M3's reduced dynamic range compared to E5M2.

### 4.5 Online Quantization

- Maximum absolute value computed **online** for each $1 \times 128$ activation tile or $128 \times 128$ weight block
- No delayed quantization or historical maximum tracking
- Ensures accurate scaling factors with simplified framework

### 4.6 Low-Precision Storage and Communication

#### 4.6.1 Optimizer States

- AdamW first and second moments stored in **BF16** (not FP32)
- Master weights and gradients (for accumulation) remain in **FP32**

#### 4.6.2 Activation Storage

- Wgrad inputs cached in **FP8** for backward pass
- **Special cases:**
  - Activations after attention operator (inputs to post-attention Linear): stored in custom **E5M6** format (higher precision, sensitivity to attention backward)
  - These activations use **round scaling** (integral power of 2) to avoid quantization error during $1 \times 128 \rightarrow 128 \times 1$ tile transposition
  - SwiGLU inputs in MoE: cached in FP8 with fine-grained quantization; SwiGLU output recomputed in backward

#### 4.6.3 Communication Compression

- Activations before MoE up-projections: quantized to **FP8** before dispatch (compatible with FP8 Fprop)
- Activation gradients before MoE down-projections: quantized to **FP8**
- Scaling factors: integral powers of 2
- Forward and backward **combine** operations: retained in **BF16**

### 4.7 Pseudo-Algorithm: FP8 GEMM with Fine-Grained Quantization

```
ALGORITHM: FP8_GEMM_FinGrained
INPUT: Activation A ∈ ℝ^{M×K} (BF16), Weight W ∈ ℝ^{K×N} (BF16)
OUTPUT: C ∈ ℝ^{M×N} (BF16/FP32)

1. TILE_QUANTIZE_ACTIVATION:
   For each row m, group channels into tiles of size 128:
     s_A[m, g] ← max|A[m, g*128:(g+1)*128]| / FP8_MAX
     A_fp8[m, g*128:(g+1)*128] ← Round(A[m, ...] / s_A[m, g])

2. BLOCK_QUANTIZE_WEIGHT:
   For each 128×128 block (b_r, b_c):
     s_W[b_r, b_c] ← max|W[b_r*128:(b_r+1)*128, b_c*128:(b_c+1)*128]| / FP8_MAX
     W_fp8[block] ← Round(W[block] / s_W[b_r, b_c])

3. GEMM_WITH_PROMOTED_ACCUMULATION:
   Initialize C_fp32 ← 0
   For k_group = 0 to K/128 - 1:
     partial ← TensorCore_MMA(A_fp8[:, k_group*128:(k_group+1)*128],
                                W_fp8[k_group*128:(k_group+1)*128, :])
     // Promote to CUDA Cores every N_C = 128 elements
     C_fp32 += partial * s_A[:, k_group] * s_W[k_group, :]   // dequantize on CUDA Cores

4. CAST C_fp32 to BF16 → C
5. RETURN C
```

---

## 5. Training Pipeline

### 5.1 Pre-Training Stage

#### 5.1.1 Hyper-Parameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| $\beta_1, \beta_2$ | 0.9, 0.95 |
| Weight decay | 0.1 |
| Max sequence length | 4096 |
| Total training tokens | 14.8T |
| Gradient clipping norm | 1.0 |
| Initial batch size | 3072 sequences |
| Final batch size | 15360 sequences |
| Batch size ramp | Linear increase over first 469B tokens |

#### 5.1.2 Learning Rate Schedule

**Phase 1 — Warmup (0 → 2K steps):**

$$
\eta_t = \frac{t}{2000} \times 2.2 \times 10^{-4}
$$

**Phase 2 — Constant (2K steps → 10T tokens):**

$$
\eta = 2.2 \times 10^{-4}
$$

**Phase 3 — Cosine Decay (10T → 14.3T tokens, spanning 4.3T tokens):**

$$
\eta_t = 2.2 \times 10^{-5} + \frac{1}{2}(2.2 \times 10^{-4} - 2.2 \times 10^{-5})\left(1 + \cos\left(\frac{\pi \cdot (t - t_{10T})}{t_{14.3T} - t_{10T}}\right)\right)
$$

**Phase 4 — Final constant (14.3T → 14.633T tokens):**

$$
\eta = 2.2 \times 10^{-5} \quad \text{(333B tokens)}
$$

**Phase 5 — Final constant (14.633T → 14.8T tokens):**

$$
\eta = 7.3 \times 10^{-6} \quad \text{(167B tokens)}
$$

#### 5.1.3 Load Balancing Schedule

- Bias update speed $\gamma = 0.001$ for first 14.3T tokens
- $\gamma = 0.0$ for final 500B tokens (bias frozen)
- Balance loss coefficient $\alpha = 0.0001$ throughout

#### 5.1.4 MTP Loss Weight Schedule

- $\lambda = 0.3$ for first 10T tokens
- $\lambda = 0.1$ for remaining 4.8T tokens

#### 5.1.5 Training Stability

- **No irrecoverable loss spikes** throughout entire pre-training
- **No rollbacks** required
- Attributed to: auxiliary-loss-free balancing, careful initialization (std = 0.006), gradient clipping, and stable FP8 framework

#### 5.1.6 Training Cost

| Stage | H800 GPU Hours | USD (at \$2/GPU-hr) |
|---|---|---|
| Pre-Training | 2,664K | \$5.328M |
| Context Extension | 119K | \$0.238M |
| Post-Training | 5K | \$0.01M |
| **Total** | **2,788K** | **\$5.576M** |

**Efficiency:** 180K H800 GPU hours per trillion tokens = 3.7 days on 2048 H800 GPUs.

### 5.2 Long Context Extension

#### 5.2.1 Method: YaRN

Two-phase progressive context extension using YaRN (Yet another RoPE extensioN), applied exclusively to the **decoupled shared key** $\mathbf{k}_t^R$.

**YaRN Configuration:**
- Scale $s = 40$
- $\alpha = 1$, $\beta = 32$
- Scaling factor $\sqrt{t} = 0.1 \ln s + 1$

#### 5.2.2 Phase 1: 4K → 32K

- Duration: 1000 steps
- Sequence length: 32K
- Batch size: 1920
- Learning rate: $7.3 \times 10^{-6}$

#### 5.2.3 Phase 2: 32K → 128K

- Duration: 1000 steps
- Sequence length: 128K
- Batch size: 480
- Learning rate: $7.3 \times 10^{-6}$

**Validation:** Needle In A Haystack (NIAH) test shows consistent performance across all context lengths up to 128K.

### 5.3 Pseudo-Algorithm: Pre-Training

```
ALGORITHM: PreTraining
INPUT: Tokenized corpus D (14.8T tokens), Model θ (671B params)
OUTPUT: Pre-trained base model θ*

1. INITIALIZE θ with random weights, std = 0.006
2. INITIALIZE expert biases b_i ← 0 for all i ∈ {1,...,N_r}
3. INITIALIZE optimizer: AdamW(β₁=0.9, β₂=0.95, wd=0.1) with BF16 moments, FP32 master weights
4. SET learning rate schedule as defined in §5.1.2
5. SET batch_size ← 3072

6. FOR step = 1 to total_steps:
   a. SAMPLE mini-batch B of packed sequences (length 4096)
   
   b. IF tokens_consumed < 469B:
        batch_size ← linear_ramp(3072, 15360, tokens_consumed/469B)
   
   c. FORWARD PASS (FP8 mixed precision):
      - Compute MLA attention (BF16 for attention ops)
      - Compute MoE routing with bias: TopK(s_{i,t} + b_i, K_r) per token
      - Compute gating values from original s_{i,t} (no bias)
      - Execute selected expert FFNs (FP8 GEMMs)
      - Compute MTP predictions (depth D=1)
   
   d. COMPUTE LOSSES:
      L_NTP ← standard next-token cross-entropy
      L_MTP ← MTP cross-entropy (weight λ)
      L_Bal ← sequence-wise balance loss (weight α=0.0001)
      L_total ← L_NTP + L_MTP + L_Bal
   
   e. BACKWARD PASS (FP8 mixed precision):
      - Recompute RMSNorm and MLA up-projections
      - Use cached FP8 activations for Wgrad
      - Gradient clipping: max_norm = 1.0
   
   f. OPTIMIZER STEP:
      - Update FP32 master weights
      - Async update EMA in CPU
   
   g. LOAD BALANCE UPDATE:
      - Monitor expert load across batch
      - For each expert i:
        IF overloaded: b_i ← b_i - γ
        IF underloaded: b_i ← b_i + γ
   
   h. Adjust λ, γ according to token consumption thresholds

7. CONTEXT EXTENSION:
   a. Phase 1: YaRN to 32K, 1000 steps, lr=7.3e-6, batch=1920
   b. Phase 2: YaRN to 128K, 1000 steps, lr=7.3e-6, batch=480

8. RETURN θ*
```

---

## 6. Post-Training

<img src="assets/deepseek_v3_engineering_blueprint-14.png" alt="Integrated pre-training and post-training orchestration showing the 14.8T token schedule and the two-lane SFT and RL data pipeline" width="100%" />

*Figure. End-to-end orchestration from long-horizon pre-training into the two-lane SFT and RL post-training pipeline, making the transition between stages explicit.*

### 6.1 Supervised Fine-Tuning (SFT)

#### 6.1.1 Dataset

- **Total instances:** 1.5M spanning multiple domains
- **Two data categories:**

**Reasoning Data (math, code competition, logic puzzles):**
1. Generate data using internal DeepSeek-R1 model
2. Train domain-specific expert model via SFT + RL pipeline
3. Generate two SFT sample types per instance:
   - $\langle\text{problem}, \text{original response}\rangle$ — concise format
   - $\langle\text{system prompt}, \text{problem}, \text{R1 response}\rangle$ — enriched with reflection/verification patterns
4. RL phase: model generates responses integrating both R1 and original patterns via high-temperature sampling (no system prompt)
5. After RL convergence: rejection sampling to curate high-quality SFT data

**Non-Reasoning Data (creative writing, role-play, simple QA):**
- Generated by DeepSeek-V2.5
- Verified by human annotators for accuracy and correctness

#### 6.1.2 Distillation from DeepSeek-R1

**Objective:** Transfer reasoning capabilities from long-CoT R1 model into standard LLM while controlling output length and style.

**Pipeline:**
1. R1-generated data has high accuracy but suffers from overthinking, poor formatting, excessive length
2. System prompt designed to guide reflection and verification mechanisms
3. RL phase learns to incorporate R1 patterns without explicit system prompts
4. Rejection sampling from RL-converged expert model produces final training data
5. Final data retains R1's reasoning strengths with concise, well-formatted responses

**Ablation (on DeepSeek-V2.5):**

| Setting | LiveCodeBench-CoT Pass@1 | Length | MATH-500 Pass@1 | Length |
|---|---|---|---|---|
| Baseline (short CoT) | 31.1 | 718 | 74.6 | 769 |
| + R1 Distill | 37.4 | 783 | 83.2 | 1510 |

#### 6.1.3 SFT Settings

- **Epochs:** 2
- **Learning rate:** Cosine decay from $5 \times 10^{-6}$ to $1 \times 10^{-6}$
- **Sequence packing:** Multiple samples per sequence
- **Sample masking:** Packed samples are isolated and mutually invisible (attention masking between samples within packed sequences — note this differs from pre-training where no cross-sample masking is applied)

### 6.2 Reinforcement Learning

<img src="assets/deepseek_v3_technical_architecture-12.png" alt="Post-training diagram combining R1 distillation data construction with group relative policy optimization and critic-free advantage normalization" width="100%" />

*Figure. Post-training synthesis of R1 distillation and GRPO, showing how supervised reasoning transfer feeds into critic-free reinforcement learning.*

#### 6.2.1 Reward Models

**Rule-Based RM:**
- Applied to problems with deterministic answers (math with boxed answers, LeetCode with compiler test cases)
- Higher reliability; resistant to manipulation/exploitation

**Model-Based RM:**
- Trained from DeepSeek-V3 SFT checkpoints
- For free-form ground-truth answers: verify response matches expected answer
- For open-ended questions (no ground-truth): provide feedback based on question + answer
- **Chain-of-thought rewards:** Preference data includes reasoning chain leading to reward (not just final scalar), mitigating reward hacking

#### 6.2.2 Group Relative Policy Optimization (GRPO)

**Definition.** GRPO eliminates the critic model and estimates baselines from group scores.

**Procedure.** For each question $q$, sample $G$ outputs $\{o_1, o_2, \ldots, o_G\}$ from old policy $\pi_{\theta_{\text{old}}}$.

**Objective:**

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\; \{o_i\} \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i, \; \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i\right) - \beta \cdot D_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right)\right]
$$

**KL divergence (per-token):**

$$
D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} - \log\frac{\pi_{\text{ref}}(o_i|q)}{\pi_\theta(o_i|q)} - 1
$$

**Advantage estimation (group-normalized):**

$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}
$$

where $r_i$ is the reward for output $o_i$, and $\varepsilon$, $\beta$ are hyper-parameters.

**Key advantage over PPO:** No critic model required (which would be same size as policy), reducing memory and compute by $\sim 50\%$.

#### 6.2.3 RL Prompt Domains

- Coding, math, writing, role-playing, question answering
- Aligns model with human preferences and improves benchmark performance, especially where SFT data is limited

#### 6.2.4 Self-Rewarding

- For open-ended questions without deterministic answers
- DeepSeek-V3 serves as generative reward model with voting (maj@6 achieves 89.6 on RewardBench)
- Constitutional AI approach: voting evaluation results of DeepSeek-V3 itself as feedback
- Enables self-improvement loop

### 6.3 Pseudo-Algorithm: Post-Training Pipeline

```
ALGORITHM: PostTraining
INPUT: Pre-trained base model θ_base, SFT dataset D_SFT (1.5M), RL prompts D_RL
OUTPUT: Aligned chat model θ_final

STAGE 1: SUPERVISED FINE-TUNING
1. θ_SFT ← θ_base
2. FOR epoch = 1 to 2:
   FOR each packed batch B from D_SFT:
     a. Apply sample masking (isolate packed samples)
     b. Compute cross-entropy loss on response tokens
     c. Update θ_SFT with AdamW, lr: cosine decay 5e-6 → 1e-6

STAGE 2: REINFORCEMENT LEARNING
3. θ_policy ← θ_SFT
4. θ_ref ← θ_SFT (frozen reference)
5. FOR each RL iteration:
   FOR each prompt q from D_RL:
     a. Sample G outputs {o_1,...,o_G} from π_{θ_old}
     b. COMPUTE REWARDS:
        - Rule-based RM for deterministic problems
        - Model-based RM for free-form/open-ended
        - Self-reward (DeepSeek-V3 voting) for subjective tasks
     c. Compute advantages A_i via group normalization
     d. Update θ_policy via GRPO objective
     e. Apply KL penalty w.r.t. θ_ref

6. θ_final ← θ_policy
7. RETURN θ_final
```

---

## 7. Inference and Deployment

### 7.1 Architecture: Disaggregated Prefill-Decode

<img src="assets/deepseek_v3_architecture-13.png" alt="Disaggregated inference architecture comparing compute-bound prefill clusters and memory-bound decode clusters with their distinct parallelism layouts" width="100%" />

*Figure. Disaggregated inference layout, separating compute-bound prefilling from memory-bound decoding so the serving system can optimize each bottleneck independently.*

Prefilling and decoding stages are separated to independently optimize for SLO and throughput.

### 7.2 Prefilling Stage

**Minimum deployment unit:** 4 nodes = 32 GPUs

| Component | Parallelism |
|---|---|
| Attention | TP4 + SP + DP8 |
| MoE | EP32 |
| Dense MLPs (shallow layers) | TP1 (saves TP communication) |

**Key Techniques:**
- **Redundant experts:** 32 redundant experts deployed (each GPU hosts 8 original + 1 redundant expert)
- High-load expert detection: statistics-based, updated every ~10 minutes
- Expert rearrangement within nodes to balance load without increasing cross-node communication
- **Micro-batch overlap:** 2 micro-batches processed simultaneously — attention/MoE of one overlaps with dispatch/combine of another
- **Dynamic redundancy (exploration):** Each GPU hosts 16 experts, 9 activated per step; globally optimal routing computed on-the-fly (negligible overhead relative to prefill computation)

### 7.3 Decoding Stage

**Minimum deployment unit:** 40 nodes = 320 GPUs

| Component | Parallelism |
|---|---|
| Attention | TP4 + SP + DP80 |
| MoE | EP320 (1 expert per GPU) |

**Key Design Decisions:**
- Shared expert treated as always-selected routed expert → 9 experts per token
- 64 GPUs dedicated to redundant + shared experts
- **All-to-all:** Direct point-to-point IB transfers + IBGDA for low latency
- **Memory-bound regime:** Batch size per expert typically ≤256 tokens; bottleneck is memory access, not computation
- Small SM allocation to dispatch+MoE+combine avoids impacting attention computation
- **Micro-batch overlap (exploration):** Attention of one micro-batch overlapped with dispatch+MoE+combine of another

### 7.4 MTP for Speculative Decoding

MTP modules can be repurposed for speculative decoding:
- MTP module predicts $D$ additional draft tokens
- Main model verifies drafts in parallel
- Accepted tokens skip individual forward passes → reduced generation latency
- Design is architecturally analogous to EAGLE speculative decoding

### 7.5 Pseudo-Algorithm: Inference

```
ALGORITHM: Inference_Prefill
INPUT: Token sequence x = (x_1,...,x_T), Model θ on 32 GPUs
OUTPUT: KV cache, logits for last position

1. TOKENIZE input using Byte-level BPE (128K vocab)
2. DISTRIBUTE across TP4 + SP for attention, EP32 for MoE
3. FOR each layer l = 1 to L:
   a. RMSNorm → MLA attention (TP4 with SP)
      - Compute compressed queries: c_t^Q = W^{DQ}h_t
      - Up-project + RoPE for queries
      - For KV: c_t^{KV} = W^{DKV}h_t (cache this + k_t^R)
      - Attention computation
   b. RMSNorm → MoE/Dense FFN
      - If l ≤ 3: Dense FFN (TP1)
      - Else: MoE routing (EP32)
        - Compute affinities s_{i,t} = σ(e_i^T u_t)
        - Select TopK with bias: TopK(s_{i,t} + b_i, 8)
        - All-to-all dispatch (FP8, ≤4 nodes)
        - Expert computation (FP8 GEMMs)
        - All-to-all combine (BF16)
      - Overlap: pipeline 2 micro-batches
4. RETURN KV cache (c_t^{KV}, k_t^R per layer), final logits

ALGORITHM: Inference_Decode
INPUT: KV cache, previously generated tokens, Model θ on 320 GPUs
OUTPUT: Next token (or speculative draft)

1. FOR each new token generation step:
   a. Embed token, compute MLA queries from compressed latent
   b. Attend to cached (c_j^{KV}, k_j^R) for all prior positions
   c. MoE routing: 9 experts (8 routed + 1 shared treated as always-on)
   d. Point-to-point IB dispatch to expert-hosting GPU
   e. Single expert computation per GPU
   f. Combine results via IB
   g. Sample next token from logits

2. OPTIONAL SPECULATIVE DECODING:
   a. Use MTP module to draft D additional tokens
   b. Verify drafts with main model in parallel
   c. Accept verified tokens, reject and resample from divergence point
```

---

## 8. Evaluation Protocol

<img src="assets/deepseek_v3_architecture-14.png" alt="Comparative evaluation chart showing DeepSeek-V3 active-parameter efficiency against frontier closed and open models across benchmark suites" width="100%" />

*Figure. Comparative evaluation summary emphasizing how 37B activated parameters reach frontier-level performance across knowledge, coding, and open-ended benchmarks.*

### 8.1 Pre-Training Evaluation (Base Model)

#### 8.1.1 Benchmark Categories

**Multi-subject multiple-choice:** MMLU, MMLU-Redux, MMLU-Pro, MMMLU, C-Eval, CMMLU

**Language understanding/reasoning:** HellaSwag, PIQA, ARC, BBH

**Closed-book QA:** TriviaQA, NaturalQuestions

**Reading comprehension:** RACE, DROP, C3, CMRC

**Reference disambiguation:** CLUEWSC, WinoGrande

**Language modeling:** Pile-test (Bits-Per-Byte metric for tokenizer-fair comparison)

**Chinese understanding:** CCPM

**Math:** GSM8K, MATH, MGSM, CMath

**Code:** HumanEval, LiveCodeBench-Base, MBPP, CRUXEval

**Standardized exams:** AGIEval

**Multilingual:** MMMLU-non-English

#### 8.1.2 Evaluation Methods

| Method | Benchmarks |
|---|---|
| Perplexity-based | HellaSwag, PIQA, WinoGrande, RACE, MMLU variants, ARC, C-Eval, CMMLU, C3, CCPM |
| Generation-based | TriviaQA, NaturalQuestions, DROP, MATH, GSM8K, MGSM, HumanEval, MBPP, LiveCodeBench, CRUXEval, BBH, AGIEval, CLUEWSC, CMRC, CMath |
| Language-modeling (BPB) | Pile-test |

#### 8.1.3 Key Base Model Results

| Benchmark | DeepSeek-V3-Base | Qwen2.5-72B | LLaMA-3.1-405B |
|---|---|---|---|
| MMLU | **87.1** | 85.0 | 84.4 |
| MMLU-Pro | **64.4** | 58.3 | 52.8 |
| HumanEval | **65.2** | 53.0 | 54.9 |
| MATH | **61.6** | 54.4 | 49.0 |
| BBH | **87.5** | 79.8 | 82.9 |

DeepSeek-V3-Base is the strongest open-source base model with only 37B activated parameters.

### 8.2 Post-Training Evaluation (Chat Model)

#### 8.2.1 Additional Benchmarks

IFEval, FRAMES, LongBench v2, GPQA, SimpleQA, CSimpleQA, SWE-Bench Verified, Aider, LiveCodeBench (Aug-Nov 2024), Codeforces, CNMO 2024, AIME 2024

#### 8.2.2 Evaluation Settings

- Maximum output length: 8192 tokens for all benchmarks
- MMLU, DROP, GPQA, SimpleQA: simple-evals framework prompts
- MMLU-Redux: Zero-Eval prompt format (zero-shot)
- AIME, CNMO 2024: temperature 0.7, averaged over 16 runs
- MATH-500: greedy decoding
- SWE-Bench: agentless framework
- Aider: 'diff' format
- LiveCodeBench: both CoT and non-CoT evaluation
- Codeforces: percentage of competitors metric

#### 8.2.3 Key Chat Model Results

| Benchmark | DeepSeek-V3 | GPT-4o | Claude-3.5 |
|---|---|---|---|
| MMLU | 88.5 | 87.2 | 88.3 |
| GPQA-Diamond | 59.1 | 49.9 | 65.0 |
| MATH-500 | **90.2** | 74.6 | 78.3 |
| AIME 2024 | **39.2** | 9.3 | 16.0 |
| Codeforces | **51.6** | 23.6 | 20.3 |
| LiveCodeBench-CoT | **40.5** | 33.4 | 36.3 |
| SWE-Bench | 42.0 | 38.8 | **50.8** |
| CSimpleQA | **64.8** | 59.3 | 51.3 |

#### 8.2.4 Open-Ended Evaluation

| Model | Arena-Hard | AlpacaEval 2.0 (LC) |
|---|---|---|
| DeepSeek-V3 | **85.5** | **70.0** |
| Claude-3.5 | 85.2 | 52.0 |
| GPT-4o | 80.4 | 51.1 |

DeepSeek-V3: first open-source model to exceed 85% on Arena-Hard.

#### 8.2.5 Generative Reward Model Evaluation (RewardBench)

| Model | Chat | Chat-Hard | Safety | Reasoning | Average |
|---|---|---|---|---|---|
| DeepSeek-V3 | 96.9 | 79.8 | 87.0 | 84.3 | 87.0 |
| DeepSeek-V3 (maj@6) | 96.9 | 82.6 | 89.5 | 89.2 | **89.6** |
| GPT-4o-0806 | 96.1 | 76.1 | 88.1 | 86.6 | 86.7 |
| Claude-3.5-1022 | 96.4 | 79.7 | 91.1 | 87.6 | 88.7 |

---

## 9. Failure Modes and Limitations

### 9.1 Identified Failure Modes

| Failure Mode | Description | Mitigation |
|---|---|---|
| **Expert load imbalance (per-sequence)** | Batch-wise balancing permits intra-sequence imbalance | Complementary sequence-wise loss ($\alpha = 0.0001$) |
| **Domain-shift load imbalance (inference)** | Distribution shift from training → serving changes expert demand | Redundant expert deployment with periodic re-detection |
| **FP8 accumulation precision loss** | H800 Tensor Core retains only ~14 bits in FP8 GEMM | Promotion to CUDA Cores at $N_C = 128$ intervals |
| **Activation outliers in quantization** | Per-tensor FP8 scaling wastes dynamic range | Fine-grained tile/block-wise quantization |
| **Token boundary bias** | Combined punctuation+linebreak tokens cause bias in few-shot evaluation | Random token splitting during training |
| **R1 distillation length inflation** | Distilled reasoning data increases average response length | Careful output length control; trade-off between accuracy and computational cost |
| **Reward hacking** | Model-based RM can be exploited | Chain-of-thought reward data; rule-based RM where possible; self-rewarding with voting |
| **English factual knowledge gap** | SimpleQA trails GPT-4o and Claude | Resource allocation prioritized Chinese knowledge; acknowledged as limitation |
| **FP8 block-wise activation quantization instability** | Applying $128 \times 128$ block quantization to activations (same as weights) causes training instability | Use $1 \times 128$ tile-wise quantization for activations only |

### 9.2 Training Stability

- **No irrecoverable loss spikes** across entire 14.8T token training
- **No rollbacks** required
- This is attributed to:
  - Auxiliary-loss-free load balancing (smooth expert utilization)
  - Careful initialization (std = 0.006)
  - Gradient clipping (max norm = 1.0)
  - FP8 training framework with fine-grained quantization and precise accumulation

---

## 10. Hardware Design Recommendations

<img src="assets/deepseek_v3_efficiency_blueprint-14.png" alt="Hardware design recommendations for communication co-processors, native block-wise scaling, FP32 accumulation support, and transposed GEMM primitives" width="100%" />

*Figure. Hardware co-design recommendations distilled from the training stack, covering communication offload, native block scaling, higher-precision accumulation, and transposed GEMM support.*

### 10.1 Communication Hardware

**Problem:** 20 SMs (out of 132) dedicated to communication — Tensor Cores entirely unused during communication tasks.

**Recommendations:**
- Dedicated GPU co-processor or network co-processor for all-to-all communication (offload from SMs)
- Unified IB (scale-out) + NVLink (scale-up) interface from compute unit perspective
- Simple primitives: `read`, `write`, `multicast`, `reduce` across unified IB-NVLink domain

### 10.2 Compute Hardware

**Higher FP8 GEMM Accumulation Precision:**
- Current: 14-bit truncation in Tensor Core accumulation
- Required: Full FP32 accumulation natively in Tensor Cores

**Native Tile/Block-Wise Quantization Support:**
- Enable Tensor Cores to receive per-group scaling factors
- Implement MMA with group scaling directly inside Tensor Cores
- Eliminate frequent data movement between Tensor Cores and CUDA Cores

**Online Quantization Support:**
- Fuse FP8 cast with TMA access (quantize during global→shared memory transfer)
- Warp-level cast instruction for fused layer norm + FP8 cast
- Near-memory computing: BF16→FP8 cast at HBM interface → ~50% off-chip memory reduction

**Transposed GEMM Support:**
- Direct transposed reads from shared memory before MMA
- Eliminates: read → dequantize → transpose → re-quantize → store → read cycle
- Combined with fused FP8/TMA access: significantly streamlined quantization workflow

---

## 11. Compression Pipeline: Information Preservation Analysis

### 11.1 MLA Compression

**Input dimensionality (full MHA):** $2 \times d_h \times n_h = 2 \times 128 \times 128 = 32768$ per token (K+V)

**Compressed dimensionality (MLA):** $d_c + d_h^R = 512 + 64 = 576$ per token

**Compression ratio:** $32768 / 576 \approx 56.9\times$

**Information preservation guarantee:** The up-projection matrices $W^{UK}, W^{UV}$ are learned jointly with the down-projection $W^{DKV}$, ensuring that the compressed latent $\mathbf{c}_t^{KV}$ captures the maximal mutual information with the full KV representation under the rank-$d_c$ constraint:

$$
\max_{W^{DKV}, W^{UK}, W^{UV}} I\left(\mathbf{c}_t^{KV};\; (\mathbf{K}_t, \mathbf{V}_t)\right) \quad \text{s.t.} \quad \mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}
$$

This is implicitly optimized through the end-to-end training loss.

**Reconstruction error bound:** For learned linear compression, the reconstruction error in the attention output is bounded by:

$$
\|\mathbf{o}_t^{\text{MHA}} - \mathbf{o}_t^{\text{MLA}}\|^2 \leq \sum_{j=d_c+1}^{d_h n_h} \sigma_j^2
$$

where $\sigma_j$ are the singular values of the KV representation matrix, ordered descending. With $d_c = 512$ and total dimension $d_h n_h = 16384$, the error comprises the tail singular values.

### 11.2 FP8 Compression

**Per-element information loss from BF16 → FP8 (E4M3):**
- BF16: 1 sign + 8 exponent + 7 mantissa = 16 bits
- FP8 E4M3: 1 sign + 4 exponent + 3 mantissa = 8 bits
- **Information loss:** 8 bits per element, but fine-grained quantization limits relative error

**Quantization error bound (per tile/block):**

$$
\|\mathbf{x} - \hat{\mathbf{x}}\|_\infty \leq \frac{s}{2^{m+1}} \cdot \max|\mathbf{x}_{\text{group}}|
$$

where $s$ is the scaling factor and $m = 3$ is the mantissa bits. With tile-wise scaling on 128 elements, the error is bounded relative to the local group maximum, not the global tensor maximum.

### 11.3 Activation Compression for Communication

**Dispatch:** Activations quantized to FP8 with power-of-2 scaling → compatible with FP8 Fprop, no additional dequantization error from non-round scales.

**Combine (forward + backward):** Retained in BF16 to preserve critical gradient information.

**Net communication reduction:** $\sim 50\%$ bandwidth savings on dispatch operations.

---

## 12. Convergence Dynamics

### 12.1 Training Dynamics

**Loss behavior:** Monotonically decreasing with no irrecoverable spikes across 14.8T tokens.

**Factors contributing to stable convergence:**

1. **Auxiliary-loss-free load balancing:** Prevents routing collapse without quality-degrading auxiliary losses
2. **Sigmoid gating with top-K normalization:** Smoother gradient flow compared to softmax gating
3. **Small initialization (std = 0.006):** Prevents early-training instability
4. **Gradient clipping (norm = 1.0):** Bounds gradient magnitude
5. **Extended constant learning rate phase:** 10T tokens at peak LR before any decay
6. **Staged LR decay:** Cosine decay over 4.3T tokens, then two constant phases
7. **Batch size warmup:** Gradual increase from 3072 to 15360 over 469B tokens
8. **FP8 with fine-grained quantization:** Maintains $< 0.25\%$ relative loss error vs. BF16

### 12.2 Scaling Efficiency

$$
\text{Cost per trillion tokens} = 180\text{K H800 GPU hours}
$$

$$
\text{Wall-clock per trillion tokens} = 3.7 \text{ days on 2048 H800 GPUs}
$$

Compared to training 72B or 405B dense models, DeepSeek-V3 achieves significantly lower cost per quality-adjusted token due to:
- MoE activation sparsity ($37\text{B}/671\text{B} = 5.5\%$ density)
- FP8 training ($\sim 2\times$ compute speedup)
- DualPipe ($\sim 0$ pipeline bubbles with communication overlap)
- No tensor parallelism (eliminating TP communication overhead)

---

## 13. Multi-Token Prediction: Speculative Decoding Application

### 13.1 Mechanism

At inference time, the MTP module predicts $D$ draft tokens autoregressively. These drafts are verified by the main model in a single forward pass over the draft sequence.

**Acceptance criterion:** Standard speculative decoding acceptance probability:

$$
P_{\text{accept}}(x_{\text{draft}}) = \min\left(1, \frac{P_{\text{main}}(x_{\text{draft}})}{P_{\text{draft}}(x_{\text{draft}})}\right)
$$

**Expected speedup:** Depends on draft acceptance rate, which is enhanced by the fact that the MTP module is trained jointly with the main model on the same data, producing well-calibrated draft distributions.

### 13.2 Architecture Alignment with EAGLE

The sequential MTP design preserves the causal chain at each depth, making it architecturally compatible with EAGLE-style speculative decoding. The single-Transformer-block MTP module is lightweight relative to the main model, ensuring minimal overhead for draft generation.

---

## 14. Summary of Tensor Transformations and Memory Flow

### 14.1 Per-Layer Tensor Flow (Single Token)

```
Input: h_t ∈ ℝ^7168 (BF16)

MLA ATTENTION:
  h_t → W^{DQ} → c_t^Q ∈ ℝ^1536 → W^{UQ} → q_t^C ∈ ℝ^16384 (128 heads × 128)
                                    → W^{QR} → RoPE → q_t^R ∈ ℝ^8192 (128 heads × 64)
  h_t → W^{DKV} → c_t^{KV} ∈ ℝ^512 [CACHED] → W^{UK} → k_t^C ∈ ℝ^16384
                                                → W^{UV} → v_t^C ∈ ℝ^16384
  h_t → W^{KR} → RoPE → k_t^R ∈ ℝ^64 [CACHED]
  
  Attention: q_t ∈ ℝ^{128×(128+64)}, K ∈ ℝ^{T×(128+64)}, V ∈ ℝ^{T×128}
  → o_t ∈ ℝ^16384 → W^O → u_t ∈ ℝ^7168
  
  Residual: u_t ← h_t + u_t

MoE FFN (layers 3-60):
  RMSNorm(u_t) → routing: σ(e_i^T u_t) + b_i → TopK(8)
  → Dispatch to ≤4 nodes (FP8)
  → 8 expert FFNs: SwiGLU(u_t) with dim 2048 each (FP8 GEMMs)
  → 1 shared expert FFN: SwiGLU(u_t) with dim 2048 (FP8 GEMM)
  → Combine (BF16)
  → h_t' = Σ shared + Σ gated routed
  
  Residual: h_{t+1} ← u_t + h_t'

Dense FFN (layers 0-2):
  RMSNorm(u_t) → SwiGLU FFN → h_t'
  Residual: h_{t+1} ← u_t + h_t'
```

### 14.2 Memory Budget Per Token (KV Cache)

$$
\text{KV cache per token per layer} = (d_c + d_h^R) \times \text{precision\_bytes} = (512 + 64) \times 2 = 1152 \text{ bytes (BF16)}
$$

$$
\text{Total KV cache per token (61 layers)} = 576 \times 61 \times 2 = 70,272 \text{ bytes} \approx 68.6 \text{ KB}
$$

For 128K context:

$$
\text{Total KV cache} = 128000 \times 68.6 \text{ KB} \approx 8.35 \text{ GB}
$$

Compare to standard MHA ($2 \times 128 \times 128 \times 2 \times 61 = 4.0$ MB per token → 485 GB for 128K context).

---

## 15. Parameterization Summary

| Component | Dimensions | Precision | Count |
|---|---|---|---|
| $W^{DKV}$ | $512 \times 7168$ | FP8 (train), BF16 (master) | 61 layers |
| $W^{UK}$ | $16384 \times 512$ | FP8 (train) | 61 layers |
| $W^{UV}$ | $16384 \times 512$ | FP8 (train) | 61 layers |
| $W^{KR}$ | $64 \times 7168$ | BF16 | 61 layers |
| $W^{DQ}$ | $1536 \times 7168$ | FP8 (train) | 61 layers |
| $W^{UQ}$ | $16384 \times 1536$ | FP8 (train) | 61 layers |
| $W^{QR}$ | $8192 \times 1536$ | BF16 | 61 layers |
| $W^O$ | $7168 \times 16384$ | FP8 (train) | 61 layers |
| Expert FFN (each) | $W_{\text{up}}, W_{\text{gate}}: 2048 \times 7168$; $W_{\text{down}}: 7168 \times 2048$ | FP8 (train) | 257 per MoE layer × 58 layers |
| Dense FFN (each) | Full-width SwiGLU | FP8 (train) | 3 layers |
| MTP module | $M_1 \in \mathbb{R}^{7168 \times 14336}$, 1 Transformer block | Mixed | 1 |
| Embedding | $128\text{K} \times 7168$ | BF16 | Shared |
| Output head | $7168 \times 128\text{K}$ | BF16 | Shared |

<img src="assets/deepseek_v3_efficiency_blueprint-15.png" alt="Final DeepSeek-V3 invariants summarizing no token dropping, no irrecoverable loss spikes, MLA KV compression, FP8 accuracy retention, and causal masking guarantees" width="100%" />

*Figure. Final invariant summary for DeepSeek-V3, closing the report with the algorithmic and systems properties that remain stable across training, inference, and deployment.*
