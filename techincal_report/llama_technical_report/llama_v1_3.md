

# Llama 3 Evaluation, Safety, and Inference: End-to-End Technical Report

---

## 1. Evaluation Data Pipeline

### 1.1 Definition and Objective

The evaluation data pipeline is the deterministic system that maps raw benchmark datasets to final model quality estimates. Given a model $\mathcal{M}$ with parameters $\theta$, and a benchmark dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, the pipeline computes a scalar performance estimate $\hat{S}(\theta, \mathcal{D})$ that approximates the true population-level capability $S^*(\theta)$.

**Formal objective:**

$$\hat{S}(\theta, \mathcal{D}) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\mathcal{M}_\theta(x_i) \equiv y_i]$$

where $\equiv$ denotes task-specific matching (exact match, accuracy, pass@k, F1, etc.).

### 1.2 Benchmark Taxonomy

Eight top-level capability categories partition the evaluation space:

| **Category** | **Benchmarks** | **Metric** |
|---|---|---|
| Commonsense Reasoning | CommonSenseQA, PiQA, SiQA, OpenBookQA, WinoGrande | Accuracy |
| Knowledge | MMLU, MMLU-Pro, AGIEval | Accuracy / Macro-avg |
| Reading Comprehension | SQuAD V2, QuAC, RACE | EM / F1 |
| Math & Reasoning | GSM8K, MATH, ARC-C, DROP, WorldSense | Accuracy / EM |
| Long Context | QuALITY, Many-shot GSM8K, Needle-in-a-Haystack, ZeroSCROLLS, InfiniteBench | Accuracy / EM / F1 |
| Code | HumanEval, MBPP, HumanEval+, MBPP EvalPlus, MultiPL-E | pass@1 |
| Adversarial | Adv SQuAD, Dynabench SQuAD, GSM-Plus, PAWS | Accuracy / EM |
| Aggregate | MMLU, MMLU-Pro, AGIEval, BIG-Bench Hard | Accuracy |

### 1.3 Inputs, Outputs, and Invariants

- **Inputs:** Raw benchmark datasets $\mathcal{D}_b$ for each benchmark $b$; model $\mathcal{M}_\theta$; few-shot exemplars $\mathcal{E}_b$; prompt template $\mathcal{T}_b$
- **Outputs:** Point estimate $\hat{S}_b$ with 95% confidence interval $\text{CI}_b$
- **Invariants:**
  - Identical evaluation pipeline applied across all competing models
  - Best-of (self-computed, externally-reported) score selection for non-Llama models
  - Few-shot count, metric, and hyperparameters fixed per benchmark

### 1.4 Confidence Interval Formulation

Under the assumption that benchmark scores are approximately Gaussian-distributed (validated via preliminary bootstrap), the 95% confidence interval is:

$$\text{CI}_{95\%} = \hat{S} \pm 1.96 \sqrt{\frac{\hat{S}(1 - \hat{S})}{N}}$$

where $\hat{S}$ is the observed benchmark score (accuracy or EM) and $N$ is the benchmark sample size. This follows the normal approximation to the binomial proportion confidence interval. This CI **lower-bounds** the actual variation since subsampling is not the only source of variation in capability estimation.

**Failure modes:**
- Gaussian assumption is incorrect (scores are bounded $[0,1]$), but bootstrap experiments confirm CI approximation quality for discrete metrics
- CI is omitted for benchmark scores that are not simple averages (e.g., macro-averaged metrics)

### 1.5 Pseudo-Algorithm: Benchmark Evaluation Pipeline

```
ALGORITHM: EvaluateBenchmark
INPUT: Model M_θ, benchmark D_b = {(x_i, y_i)}_{i=1}^N, template T_b, exemplars E_b, metric_fn
OUTPUT: Score S_hat, CI_lower, CI_upper

1. FOR i = 1 TO N:
   a. prompt_i ← FormatPrompt(T_b, E_b, x_i)
   b. response_i ← M_θ.generate(prompt_i, decoding_params)
   c. score_i ← metric_fn(response_i, y_i)
2. S_hat ← (1/N) * SUM(score_i)
3. std_err ← SQRT(S_hat * (1 - S_hat) / N)
4. CI_lower ← S_hat - 1.96 * std_err
5. CI_upper ← S_hat + 1.96 * std_err
6. RETURN S_hat, CI_lower, CI_upper
```

---

## 2. Pre-trained Model Evaluation

### 2.1 Standard Benchmark Results

#### 2.1.1 Performance Summary: 8B and 70B Models

Results are aggregated by capability category via arithmetic mean of accuracies across constituent benchmarks.

**Key findings by model size:**

**Llama 3 8B vs. competitors (Mistral 7B, Gemma 7B, Llama 2 7B):**

| **Benchmark** | **Llama 3 8B** | **Mistral 7B** | **Gemma 7B** |
|---|---|---|---|
| SQuAD | $77.0 \pm 0.8$ | $73.2 \pm 0.8$ | $81.8 \pm 0.7$ |
| QuAC | $44.9 \pm 1.1$ | $44.7 \pm 1.1$ | $42.4 \pm 1.1$ |
| RACE | $54.3 \pm 1.4$ | $53.0 \pm 1.4$ | $48.8 \pm 1.4$ |
| HumanEval | $37.2 \pm 7.4$ | $30.5 \pm 7.0$ | $32.3 \pm 7.2$ |
| MBPP | $47.6 \pm 4.4$ | $47.5 \pm 4.4$ | $44.4 \pm 4.4$ |
| CommonSenseQA | $75.0 \pm 2.5$ | $71.2 \pm 2.6$ | $74.4 \pm 2.5$ |
| PiQA | $81.0 \pm 1.8$ | $83.0 \pm 1.7$ | $81.5 \pm 1.8$ |
| SiQA | $49.5 \pm 2.2$ | $48.2 \pm 2.2$ | $51.8 \pm 2.2$ |
| OpenBookQA | $45.0 \pm 4.4$ | $47.8 \pm 4.4$ | $52.8 \pm 4.4$ |
| WinoGrande | $75.7 \pm 2.0$ | $78.1 \pm 1.9$ | $74.7 \pm 2.0$ |
| GSM8K | $57.2 \pm 2.7$ | $52.5 \pm 2.7$ | $46.4 \pm 2.7$ |
| MATH | $20.3 \pm 1.1$ | $13.1 \pm 0.9$ | $24.3 \pm 1.2$ |
| ARC-C | $79.7 \pm 2.3$ | $78.2 \pm 2.4$ | $78.6 \pm 2.4$ |
| DROP | $59.5 \pm 1.0$ | $53.0 \pm 1.0$ | $56.3 \pm 1.0$ |
| WorldSense | $45.5 \pm 0.3$ | $44.9 \pm 0.3$ | $46.0 \pm 0.3$ |
| MMLU | $66.7$ | $63.6$ | $64.3$ |
| MMLU-Pro | $37.1$ | $32.5$ | $35.1$ |
| AGIEval | $47.8 \pm 1.9$ | $42.7 \pm 1.9$ | $46.0 \pm 1.9$ |
| BB Hard | $64.2 \pm 1.2$ | $56.8 \pm 1.2$ | $57.7 \pm 1.2$ |

**Conclusion:** Llama 3 8B outperforms competing models in virtually every category, both in per-category win rate and average per-category performance.

**Llama 3 70B vs. competitors (Llama 2 70B, Mixtral 8×22B):**

| **Benchmark** | **Llama 3 70B** | **Mixtral 8×22B** |
|---|---|---|
| SQuAD | $81.8 \pm 0.7$ | $84.1 \pm 0.7$ |
| QuAC | $51.1 \pm 1.1$ | $44.9 \pm 1.1$ |
| RACE | $59.0 \pm 1.4$ | $59.2 \pm 1.4$ |
| HumanEval | $58.5 \pm 7.5$ | $45.1 \pm 7.6$ |
| MBPP | $66.2 \pm 4.1$ | $71.2 \pm 4.0$ |
| GSM8K | $83.7 \pm 2.0$ | $88.4 \pm 1.7$ |
| MATH | $41.4 \pm 1.4$ | $41.8 \pm 1.4$ |
| ARC-C | $92.9 \pm 1.5$ | $91.9 \pm 1.6$ |
| MMLU | $79.3$ | $77.8$ |
| MMLU-Pro | $53.8$ | $51.5$ |
| BB Hard | $81.6 \pm 0.9$ | $79.5 \pm 1.0$ |

**Conclusion:** Llama 3 70B outperforms Llama 2 70B by large margins on most benchmarks (exception: saturated commonsense benchmarks). Outperforms Mixtral 8×22B overall.

#### 2.1.2 Llama 3 405B Performance

| **Benchmark** | **Llama 3 405B** | **GPT-4** | **Nemotron 4 340B** | **Gemini Ultra** |
|---|---|---|---|---|
| HumanEval | $61.0 \pm 7.5$ | $67.0 \pm 7.2$ | $57.3 \pm 7.6$ | $74.4 \pm 6.7$ |
| MBPP | $73.4 \pm 3.9$ | — | — | — |
| GSM8K | $89.0 \pm 1.7$ | $92.0 \pm 1.5$ | — | $88.9 \pm 1.7$ |
| MATH | $53.8 \pm 1.4$ | — | — | $53.2 \pm 1.4$ |
| ARC-C | $96.1 \pm 1.1$ | $96.3 \pm 1.1$ | $94.3 \pm 1.3$ | — |
| MMLU | $85.2$ | $86.4$ | $81.1$ | $83.7$ |
| MMLU-Pro | $61.6$ | — | — | — |
| BB Hard | $85.9 \pm 0.8$ | — | $85.4 \pm 0.9$ | $83.6 \pm 0.9$ |
| WinoGrande | $82.2 \pm 1.8$ | $87.5 \pm 1.5$ | $89.5 \pm 1.4$ | — |

**Conclusion:** Llama 3 405B is competitive with GPT-4, Nemotron 4 340B, and Gemini Ultra. Substantially outperforms all prior open-source models. Category averages not reported for 405B because not all competitor numbers are available across all benchmarks.

#### 2.1.3 Long Context Pre-training Results

| **Benchmark** | **8B** | **70B** | **405B** |
|---|---|---|---|
| QuALITY (5-shot) | $56.0 \pm 2.1$ | $82.8 \pm 1.6$ | $87.6 \pm 1.4$ |
| GSM8K (16-shot) | $60.0 \pm 9.6$ | $83.0 \pm 7.4$ | $90.0 \pm 5.9$ |

---

## 3. Pre-trained Model Robustness Analysis

### 3.1 Definition

Robustness is the invariance of model performance $\hat{S}$ to semantically irrelevant perturbations of the evaluation setup. Formally, for a set of perturbation functions $\{\pi_j\}_{j=1}^{M}$ that modify the prompt format, label set, answer order, or few-shot exemplar structure without changing the underlying task:

$$\text{Robustness}(\mathcal{M}_\theta) = 1 - \frac{\text{Var}_j[\hat{S}(\theta, \pi_j(\mathcal{D}))]}{\hat{S}(\theta, \mathcal{D})^2}$$

A perfectly robust model yields $\text{Var}_j = 0$.

### 3.2 Perturbation Dimensions

Four orthogonal perturbation axes are investigated on MMLU:

#### 3.2.1 Few-Shot Label Bias

Perturbation of the distribution of labels in 4-shot exemplars:
- **ABCD**: all four exemplars have distinct labels
- **AAAA**: all exemplars share the same label
- **AABB / CCDD**: two labels present

**Result:** Llama 3 models (8B, 70B, 405B) exhibit near-identical micro-accuracy across all four conditions. 405B shows the smallest variance. The models are robust to few-shot label distribution manipulation.

#### 3.2.2 Label Variants

Five label token sets:
- Canonical: $\{A., B., C., D.\}$ and $\{A), B), C), D)\}$
- Numerical: $\{1, 2, 3, 4\}$
- Language-independent: $\{\$, \&, \#, @\}$
- Rare tokens: $\{œ, §, з, ü\}$

**Result:** Performance remains stable across all label sets. Pronounced robustness at 405B scale.

#### 3.2.3 Answer Order

Answers are remapped according to fixed permutations with Hamming distance $d \in \{0, 2, 3, 4\}$ from the identity permutation $\sigma_0 = (A, B, C, D)$.

**Result:** Micro-accuracy variation across permutation distances is minimal ($<2\%$ range for 405B).

#### 3.2.4 Prompt Format

Five task prompts with varying levels of information:
- Minimal ("answer the question")
- Assertive ("you are an expert")
- Instructive ("choose the best answer")
- Variants thereof

**Result:** Llama 3 405B maintains accuracy within a narrow band ($\sim 2\%$ range), demonstrating strong prompt-format invariance.

### 3.3 Failure Modes

- Smaller models (8B) show slightly higher sensitivity to label variants and prompt formats
- Commonsense benchmarks (PiQA, WinoGrande) may be saturated, masking robustness differences
- Robustness does not guarantee calibration; a model can be robust but poorly calibrated

---

## 4. Adversarial Evaluation Protocol

### 4.1 Definition

Adversarial evaluation measures the performance gap $\Delta_{\text{adv}}$ between standard and adversarially-constructed test sets for the same task:

$$\Delta_{\text{adv}} = S_{\text{standard}} - S_{\text{adversarial}}$$

A large $\Delta_{\text{adv}}$ indicates vulnerability to distribution shift specifically engineered to exploit model weaknesses.

### 4.2 Benchmark Pairs

| **Task** | **Standard** | **Adversarial** |
|---|---|---|
| Question Answering | SQuAD | Adversarial SQuAD, Dynabench SQuAD |
| Mathematical Reasoning | GSM8K | GSM-Plus |
| Paraphrase Detection | QQP | PAWS |

### 4.3 Results

**Paraphrase Detection:** Neither pre-trained nor post-trained models show significant performance degradation from QQP to PAWS. Points lie near the parity diagonal ($S_{\text{adv}} \approx S_{\text{standard}}$). This confirms findings that modern LLMs are less susceptible to spurious correlation exploitation in paraphrase adversarial datasets.

**Mathematical Reasoning and Question Answering:** Substantial performance drop from standard to adversarial benchmarks ($\Delta_{\text{adv}} > 0$). Pattern is consistent across pre-trained and post-trained models, and across all three model sizes (8B, 70B, 405B). Points fall significantly below the parity diagonal.

### 4.4 Invariants and Failure Modes

- **Invariant:** Performance on adversarial QA and math benchmarks is strictly below non-adversarial performance across all model sizes
- **Failure mode:** Adversarial mathematical reasoning (GSM-Plus) reveals sensitivity to problem reformulations that preserve mathematical content but alter surface form — indicating potential overfitting to benchmark-specific problem structures
- **Failure mode:** Models remain vulnerable to adversarial perturbations in QA despite scale increases

---

## 5. Contamination Analysis

### 5.1 Definition

Contamination analysis estimates the degree to which evaluation benchmark data $\mathcal{D}_b$ overlaps with the pre-training corpus $\mathcal{C}$, and quantifies the resulting performance inflation.

### 5.2 Method

**Detection method:** 8-gram overlap, following Singh et al. (2024).

An example $x_i \in \mathcal{D}_b$ is considered contaminated if the fraction of its tokens that participate in at least one 8-gram also occurring in $\mathcal{C}$ exceeds a dataset-specific threshold $T_{\mathcal{D}}$:

$$\text{Contaminated}(x_i) = \mathbb{1}\left[\frac{|\{t \in x_i : \exists \text{ 8-gram in } \mathcal{C} \text{ containing } t\}|}{|x_i|} \geq T_{\mathcal{D}}\right]$$

**Threshold selection:** $T_{\mathcal{D}}$ is selected per dataset to maximize the significant **estimated performance gain (EPG)**:

$$\text{EPG} = \hat{S}_{\text{contaminated}} - \hat{S}_{\text{clean}}$$

where $\hat{S}_{\text{contaminated}}$ is the average score on contaminated examples and $\hat{S}_{\text{clean}}$ is the average score on the clean subset.

### 5.3 Results

| **Benchmark** | **Contam. %** | **EPG 8B** | **EPG 70B** | **EPG 405B** |
|---|---|---|---|---|
| AGIEval | 98% | 8.5 | 19.9 | 16.3 |
| BIG-Bench Hard | 95% | 26.0 | 36.0 | 41.0 |
| BoolQ | 96% | 4.0 | 4.7 | 3.9 |
| CommonSenseQA | 30% | 0.1 | 0.8 | 0.6 |
| GSM8K | 41% | 0.0 | 0.1 | 1.3 |
| HellaSwag | 85% | 14.8 | 14.8 | 14.3 |
| MATH | 1% | 0.0 | −0.1 | −0.2 |
| NaturalQuestions | 52% | 1.6 | 0.9 | 0.8 |
| OpenBookQA | 21% | 3.0 | 3.3 | 2.6 |
| PiQA | 55% | 8.5 | 7.9 | 8.1 |
| QuAC | 99% | 2.4 | 11.0 | 6.4 |
| SiQA | 63% | 2.0 | 2.3 | 2.6 |
| SQuAD | 0% | 0.0 | 0.0 | 0.0 |
| WinoGrande | 6% | −0.1 | −0.1 | −0.2 |
| WorldSense | 73% | −3.1 | −0.4 | 3.9 |

**Excluded benchmarks:** DROP, HumanEval, MBPP, MMLU, MMLU-Pro, RACE — results not significant or 8-gram overlap saturates before meaningful partition.

### 5.4 Interpretation

- **High contamination + high EPG:** PiQA (55%, ~8 EPG), HellaSwag (85%, ~14.8 EPG), BIG-Bench Hard (95%, ~36 EPG) — scores likely inflated
- **High contamination + near-zero EPG:** NaturalQuestions (52%, ~1 EPG), SQuAD (0%, 0 EPG), MATH (1%, 0 EPG) — contamination present but not performance-boosting
- **Unresolvable:** MBPP, HumanEval, MMLU, MMLU-Pro — 8-gram method produces uniformly high contamination scores, preventing clean/contaminated partition

### 5.5 Failure Modes

- 8-gram overlap method suffers from false positives (common n-grams in code, formulaic text)
- False negatives for paraphrased contamination
- No single contamination method is universally optimal; dataset-specific method selection is required
- Threshold selection via EPG maximization introduces selection bias

### 5.6 Pseudo-Algorithm: Contamination Analysis

```
ALGORITHM: ContaminationAnalysis
INPUT: Benchmark D_b = {(x_i, y_i)}_{i=1}^N, pre-training corpus C, 
       model M_θ, threshold candidates T_set
OUTPUT: Contamination percentage, EPG per model size

1. BUILD rolling_hash_index ← Index all 8-grams in C
2. FOR each threshold T in T_set:
   a. FOR each example x_i in D_b:
      - Compute overlap_ratio_i ← fraction of tokens in x_i 
        belonging to 8-grams present in rolling_hash_index
      - contaminated_i ← (overlap_ratio_i >= T)
   b. D_clean ← {(x_i, y_i) : NOT contaminated_i}
   c. D_contam ← {(x_i, y_i) : contaminated_i}
   d. S_clean ← EvaluateBenchmark(M_θ, D_clean)
   e. S_contam ← EvaluateBenchmark(M_θ, D_contam)
   f. EPG_T ← S_contam - S_clean
   g. Compute statistical significance of EPG_T
3. T_D* ← argmax_{T in T_set} EPG_T subject to significance
4. contam_pct ← |D_contam(T_D*)| / N
5. RETURN contam_pct, EPG(T_D*) for each model size
```

---

## 6. Post-trained Model Evaluation

### 6.1 Benchmark Taxonomy

| **Category** | **Benchmarks** |
|---|---|
| General | MMLU, MMLU-Pro, IFEval |
| Math & Reasoning | GSM8K, MATH, GPQA, ARC-Challenge |
| Code | HumanEval, MBPP, HumanEval+, MBPP EvalPlus (base), MultiPL-E |
| Multilingual | MGSM, Multilingual MMLU (internal) |
| Tool Use | Nexus, API-Bank, API-Bench, BFCL |
| Long Context | ZeroSCROLLS, Needle-in-a-Haystack, InfiniteBench |

**Decontamination:** Exact match applied between post-training data prompts and each benchmark's prompts.

<img src="assets/llama_3_technical_synthesis_p14.png" alt="State-of-the-art benchmark synthesis comparing Llama 3 model scales with frontier baselines" width="100%" />

*Figure. Benchmark synthesis for the Llama 3 family, corresponding to the evaluation-heavy sections that compare 8B, 70B, and 405B models against frontier baselines.*

### 6.2 General Knowledge and Instruction Following

#### 6.2.1 MMLU and MMLU-Pro

- **MMLU:** 5-shot, no CoT, macro-average of subtask accuracy, formatted as generation tasks
- **MMLU-Pro:** 5-shot CoT (due to reasoning focus), 10-option MCQ

**Key result:** All Llama 3 variants (8B, 70B, 405B) outperform comparable-size models on both tasks. 405B outperforms GPT-4 and Nemotron 4 340B; Claude 3.5 Sonnet leads among larger models.

#### 6.2.2 IFEval (Instruction Following)

~500 verifiable instructions ("write in more than 400 words"). Metric: average of prompt-level and instruction-level accuracy, under strict and loose constraints.

**Key result:** All Llama 3 variants outperform comparable models on IFEval.

### 6.3 Proficiency Exams

**Exam sources:** GRE (Practice Tests 1&2), LSAT (Preptests 71, 73, 80, 93), SAT (8 exams), AP (one official practice per subject), GMAT (Online Exam).

**Protocol:**
- MCQ + generation questions; image-accompanied questions excluded
- GRE multi-correct: output scored correct only if all correct options selected
- Few-shot prompting when multiple exam sets available
- GRE scores: scaled to 130–170; all others: accuracy

**Key results (405B):**

| **Exam** | **Llama 3 405B** | **GPT-4o** | **Claude 3.5 Sonnet** |
|---|---|---|---|
| LSAT | $81.1 \pm 3.8$ | $77.4 \pm 4.1$ | $80.0 \pm 3.9$ |
| SAT Math | $94.9 \pm 2.3$ | $95.5 \pm 2.2$ | $95.8 \pm 2.1$ |
| AP Average | $93.5 \pm 1.9$ | $93.0 \pm 2.0$ | $92.2 \pm 2.1$ |
| GRE Quant. | 162 | 166 | 164 |
| GRE Verbal | 166 | 167 | 167 |

405B is competitive with GPT-4o and Claude 3.5 Sonnet. 70B significantly outperforms GPT-3.5 Turbo and beats Nemotron 4 340B on many tests.

### 6.4 Coding Benchmarks

#### 6.4.1 pass@$k$ Metric

For $k$ generated solutions, pass@$k$ estimates the probability that at least one solution passes all unit tests. For pass@1:

$$\text{pass@1} = \mathbb{E}_{x \sim \mathcal{D}}\left[\mathbb{1}\left[\text{passes\_all\_tests}(\mathcal{M}_\theta(x))\right]\right]$$

#### 6.4.2 Python Code Generation

| **Model** | **HumanEval** | **HumanEval+** | **MBPP** | **MBPP EvalPlus** |
|---|---|---|---|---|
| Llama 3 8B | $72.6 \pm 6.8$ | $67.1 \pm 7.2$ | $60.8 \pm 4.3$ | $72.8 \pm 4.5$ |
| Llama 3 70B | $80.5 \pm 6.1$ | $74.4 \pm 6.7$ | $75.4 \pm 3.8$ | $86.0 \pm 3.5$ |
| Llama 3 405B | $89.0 \pm 4.8$ | $82.3 \pm 5.8$ | $78.8 \pm 3.6$ | $88.6 \pm 3.2$ |
| GPT-4o | $90.2 \pm 4.5$ | $86.0 \pm 5.3$ | $81.4 \pm 3.4$ | $87.8 \pm 3.3$ |
| Claude 3.5 Sonnet | $92.0 \pm 4.2$ | $82.3 \pm 5.8$ | $76.6 \pm 3.7$ | $90.5 \pm 3.0$ |

**Conclusion:** 8B and 70B outperform same-size competitors. At 405B scale, competitive with GPT-4o and Claude 3.5 Sonnet.

#### 6.4.3 Multi-Programming Language (MultiPL-E)

| **Model** | **C++** | **Java** | **PHP** | **TS** | **C#** | **Shell** |
|---|---|---|---|---|---|---|
| Llama 3 405B (HE) | $82.0$ | $80.4$ | $76.4$ | $81.1$ | $54.4$ | $57.6$ |
| Llama 3 405B (MBPP) | $67.5$ | $65.8$ | $76.6$ | $72.6$ | $53.1$ | $43.7$ |

**Failure mode:** Significant performance drop from Python to non-Python languages, especially C# and Shell.

### 6.5 Multilingual Benchmarks

**Supported languages:** English, German, French, Italian, Portuguese, Hindi, Spanish, Thai.

- **Multilingual MMLU:** MMLU translated via Google Translate; instructions in English; 5-shot; average across 7 non-English languages
- **MGSM:** 0-shot CoT; native prompts from simple-evals

| **Model** | **MGSM** | **Multilingual MMLU** |
|---|---|---|
| Llama 3 8B | 68.9 | 58.6 |
| Llama 3 70B | 86.9 | 78.2 |
| Llama 3 405B | 91.6 | 83.2 |
| GPT-4o | 90.5 | 85.5 |

**Conclusion:** 405B outperforms most models on MGSM (91.6%). Trails GPT-4o by 2% on Multilingual MMLU. 8B and 70B lead competitors by wide margins.

### 6.6 Long Context Benchmarks

#### 6.6.1 Needle-in-a-Haystack

100% retrieval rate across all document depths and context lengths for all model sizes. Multi-needle (4 inserted, 2 retrieved): near-perfect recall.

| **Model** | **Multi-needle (avg recall)** |
|---|---|
| Llama 3 8B | $98.8 \pm 1.2$ |
| Llama 3 70B | $97.5 \pm 1.7$ |
| Llama 3 405B | $98.1 \pm 1.5$ |
| GPT-4 | $100.0 \pm 0.0$ |
| GPT-4o | $100.0 \pm 0.0$ |

#### 6.6.2 ZeroSCROLLS and InfiniteBench

| **Benchmark** | **Llama 3 405B** | **GPT-4** | **GPT-4o** | **Claude 3.5 Sonnet** |
|---|---|---|---|---|
| QuALITY (EM) | $95.2 \pm 9.1$ | $95.2 \pm 9.1$ | $90.5 \pm 12.5$ | $90.5 \pm 12.6$ |
| Qasper (F1) | $49.8 \pm 18.5$ | $50.5 \pm 18.5$ | $49.2 \pm 18.5$ | $18.5 \pm 14.4$ |
| InfiniteBench En.QA (F1) | $30.5 \pm 4.8$ | $15.7 \pm 3.8$ | $19.1 \pm 4.1$ | $11.3 \pm 3.3$ |
| InfiniteBench En.MC (Acc) | $83.4 \pm 4.8$ | $72.0 \pm 5.8$ | $82.5 \pm 4.9$ | — |

**Conclusion:** 405B matches or surpasses competitors on ZeroSCROLLS. Significantly outperforms on InfiniteBench En.QA.

### 6.7 Tool Use Benchmarks

| **Benchmark** | **Llama 3 8B** | **Llama 3 70B** | **Llama 3 405B** | **GPT-4o** | **Claude 3.5 Sonnet** |
|---|---|---|---|---|---|
| Nexus | $38.5 \pm 4.1$ | $56.7 \pm 4.2$ | $58.7 \pm 4.1$ | $56.1 \pm 4.2$ | $45.7 \pm 4.2$ |
| API-Bank | $82.6 \pm 3.8$ | $90.0 \pm 3.0$ | $92.3 \pm 2.6$ | $91.3 \pm 2.8$ | $92.6 \pm 2.6$ |
| API-Bench | $8.2 \pm 1.3$ | $29.7 \pm 2.1$ | $35.3 \pm 2.2$ | $41.4 \pm 2.3$ | $60.0 \pm 2.3$ |
| BFCL | $76.1 \pm 2.0$ | $84.8 \pm 1.7$ | $88.5 \pm 1.5$ | $80.5 \pm 1.9$ | $90.2 \pm 1.4$ |

**Human evaluation (tool use):** 2000 prompts (code execution, plotting, file uploads). Llama 3 405B significantly beats GPT-4o on text-only code execution and plot generation, but lags on file upload use cases.

---

## 7. Human Evaluation Protocol

### 7.1 Prompt Collection

- **Taxonomy-driven:** categories and subcategories spanning model capabilities
- **~7,000 prompts:** 6 single-turn capabilities (English, reasoning, coding, Hindi, Spanish, Portuguese) + 3 multi-turn capabilities (English, reasoning, coding)
- **Difficulty distribution:** 10% easy, 30% medium, 60% hard
- **Multi-turn:** 2–11 turns per prompt; model evaluated on final turn response
- **Contamination prevention:** modeling teams have no access to evaluation prompts

### 7.2 Evaluation Process

**Pairwise comparison:** Human annotators evaluate two model responses (model identity hidden) on a **7-point scale:**

- Much better / Better / Slightly better / About the same (for each direction)

**Win definition:** annotator rates response as "better" or "much better" → counted as a win.

**Win rate:** per capability category, computed with 95% confidence intervals, excluding ties.

### 7.3 Results: Llama 3 405B vs. Competitors

**vs. GPT-4 (0125 API):**
- Approximately on par overall
- Win rates within margin of error on nearly all capabilities
- Llama 3 outperforms on multi-turn reasoning and coding
- GPT-4 outperforms on multilingual (Hindi, Spanish, Portuguese)

**vs. GPT-4o:**
- On par on English prompts
- Mixed results on other capabilities

**vs. Claude 3.5 Sonnet:**
- On par on multilingual
- Outperforms on single and multi-turn English
- Trails on coding and reasoning

### 7.4 Qualitative Findings

Performance in human evaluations is heavily influenced by:
- Model **tone**
- Response **structure**
- **Verbosity**

These are nuanced factors actively optimized during post-training.

### 7.5 Limitations

- Human evaluations subject to annotator bias, background, preferences
- Difficult to define objective evaluation criteria for open-ended responses
- Despite quality assurance, results may contain inconsistencies

---

## 8. Safety Pipeline

### 8.1 Safety Pre-training

#### 8.1.1 Data Filtering

- PII detection filters applied to web-crawled data
- Domain-specific filters for harmful content removal

#### 8.1.2 Memorization Analysis

**Method:** Following Carlini et al. (2022), sample prompts and ground truths at varying frequencies using a **rolling hash index** of all n-grams in the corpus. Measure **verbatim memorization** as inclusion rate — the proportion of model generations that include the exact ground truth continuation.

$$\text{Inclusion Rate} = \frac{|\{g \in \mathcal{G} : \text{ground\_truth} \subseteq g\}|}{|\mathcal{G}|}$$

where $\mathcal{G}$ is the set of model generations.

**Results (weighted averages by data prevalence):**

| **Model** | **English, 50-gram** | **All, 50-gram** | **All, 1000-gram** |
|---|---|---|---|
| Llama 3 8B | 0.26% | 0.24% | 1.11% |
| Llama 2 7B | 0.20% | — | — |
| Llama 3 70B | 0.60% | 0.55% | 3.56% |
| Llama 2 70B | 0.47% | — | — |
| Llama 3 405B | 1.13% | 1.03% | 3.91% |

**Scaling behavior:** Memorization rate increases with model size (8B → 70B → 405B). Rates are roughly comparable to Llama 2 at equivalent sizes using the same methodology.

**Limitations:** Analysis uses exact match only; recent work advocates approximate matching metrics and alternative prompt search strategies.

### 8.2 Safety Finetuning

#### 8.2.1 Metrics

Two primary metrics optimized jointly:

$$\text{VR} = \frac{|\{x \in \mathcal{D}_{\text{adv}} : \mathcal{M}_\theta(x) \text{ violates policy}\}|}{|\mathcal{D}_{\text{adv}}|}$$

$$\text{FRR} = \frac{|\{x \in \mathcal{D}_{\text{border}} : \mathcal{M}_\theta(x) \text{ refuses helpfully}\}|}{|\mathcal{D}_{\text{border}}|}$$

**Optimization objective:** Minimize VR while keeping FRR low, subject to maintaining helpfulness on standard benchmarks:

$$\min_{\theta} \text{VR}(\theta) \quad \text{s.t.} \quad \text{FRR}(\theta) \leq \epsilon_{\text{FRR}}, \quad S_{\text{helpful}}(\theta) \geq S_{\min}$$

#### 8.2.2 Finetuning Data Design

- **Primary source:** Human-generated adversarial prompts + AI-assisted annotation tools for quality assurance
- **Adversarial prompts:** designed to elicit policy violations
- **Borderline prompts:** semantically similar to adversarial prompts but should receive helpful responses; teach model to distinguish safe/unsafe requests
- **Synthetic data generation techniques:**
  - In-context learning with safety-focused system prompts
  - Guided mutation of seed prompts via new attack vectors
  - **Rainbow Teaming** (based on MAP-Elites): generates prompts constrained across multiple diversity dimensions
- **Tone:** Refusal tone guidelines developed; existing data refined via zero-shot rewriting + human-in-the-loop editing; tone classifier deployed for quality assessment

**Key finding:** Quality > quantity for safety data. Small amounts of high-quality, carefully curated safety data outperform larger quantities of noisy data.

#### 8.2.3 Safety Supervised Finetuning (Safety SFT)

Following Llama 2 recipe:
- All helpfulness data + all safety data combined during alignment stage
- Borderline dataset added to teach nuanced safe/unsafe discrimination
- Strategic balancing of adversarial-to-borderline ratio per risk area

**Model size effect on VR–FRR trade-off:**

$$\text{ratio}_{\text{safety}} \propto \frac{1}{\text{model\_capacity}}$$

Smaller models (8B) require a **higher proportion** of safety data relative to helpfulness data in the SFT mix to achieve comparable VR to larger models (70B). Larger models are more capable of discerning adversarial vs. borderline context, yielding a more favorable VR–FRR Pareto front.

#### 8.2.4 Safety DPO

- Adversarial + borderline examples incorporated into preference datasets
- Response pairs crafted to be **nearly orthogonal in embedding space** for maximal discriminative signal:

$$\cos(\mathbf{e}_{\text{chosen}}, \mathbf{e}_{\text{rejected}}) \approx 0$$

where $\mathbf{e}_{\text{chosen}}, \mathbf{e}_{\text{rejected}}$ are the embedding representations of chosen and rejected responses.

- Optimal ratio of adversarial/borderline/helpfulness examples determined via ablation
- Different safety mixes tailored per model size

### 8.3 Benchmark Construction for Safety

- Internal benchmarks inspired by ML Commons taxonomy of hazards
- Human-written prompts: adversarial (direct harmful elicitation + sophisticated jailbreaks) and borderline (near decision boundary, plausible safe response exists)
- **>4000 prompts per capability or language**
- Mix of single-turn and multi-turn prompts

### 8.4 Safety Results

#### 8.4.1 Overall Performance

**VR comparison (Llama 3 405B vs. competitors):**
- Llama 3 405B achieves competitive VR across English and all 7 supported non-English languages
- With Llama Guard: Pareto-dominant (lower VR, comparable FRR) relative to at least one competitor system across all capabilities

**VR–FRR trade-off positioning:**
- Llama 3 models achieve balanced VR–FRR, avoiding both excessive refusal and excessive violation
- Some competitors are skewed toward one extreme

#### 8.4.2 Multilingual Safety

- Safety knowledge in English **does not readily transfer** to other languages
- High-quality per-language safety data is essential
- Distribution of safety data per language significantly impacts performance
- Some languages benefit from cross-lingual transfer; others require more language-specific data
- Iterative adversarial + borderline data addition with VR/FRR monitoring

**Result:** Llama 3 405B + Llama Guard is at least as safe as two competing systems across all supported languages on internal benchmarks, with competitive FRR. Standalone Llama 3 405B has significantly lower VR than a competing open-source model, trading off slightly higher FRR.

#### 8.4.3 Long-Context Safety

**Vulnerability:** Many-shot jailbreaking (Anil et al., 2024) — unsafe demonstrations in context can induce unsafe behavior.

**Mitigation:** SFT on datasets including examples of safe behavior in presence of unsafe in-context demonstrations. Scalable strategy that:
- Significantly reduces VR
- Neutralizes impact of attacks up to 256-shot
- Shows little/no impact on FRR or helpfulness metrics

**Evaluation methods:**
- **DocQA:** Long documents with potentially adversarial information + prompts testing model's ability to refuse unsafe requests related to document content
- **Many-shot:** Synthetic chat history of unsafe prompt-response pairs + final unrelated prompt testing context-induced unsafe behavior

**Result:** Llama 3 405B (with and without Llama Guard) is Pareto-better than Comp. 2 across VR and FRR on both DocQA and Many-shot. Significantly safer than Comp. 1, with trade-off on FRR.

#### 8.4.4 Tool Usage Safety

Focus on search use case. Llama 3 405B significantly safer than Comp. 1, with slightly higher FRR.

### 8.5 Cybersecurity Evaluation

**Framework:** CyberSecEval (Bhatt et al., 2023, 2024)

#### 8.5.1 Insecure Coding

Larger models generate more insecure code with higher average BLEU scores — a scaling-induced safety concern.

#### 8.5.2 Code Interpreter Abuse

- Llama 3 405B complies with malicious prompts **10.4%** of the time
- Llama 3 70B: **3.8%** compliance rate

#### 8.5.3 Prompt Injection

Prompt injection success rates against Llama 3 405B: **21.7%** on average.

**Comparison across models and attack strategies (success rate range):**
- Llama 3 models: more susceptible than GPT-4 Turbo and Gemini Pro
- Less susceptible than Mixtral models
- Most effective attack vectors: output formatting manipulation, repeated token attack, different user input language

#### 8.5.4 Vulnerability Identification

Llama 3 does not outperform traditional non-LLM tools on capture-the-flag challenges.

#### 8.5.5 Spear Phishing

- Judge LLM (Llama 3 70B) evaluates persuasiveness on a numeric scale
- Llama 3 70B: moderately persuasive, 24% success rate
- Llama 3 405B: moderately persuasive, 14% success rate
- Success rates vary by phishing objective (malware download, security info gathering, data theft, credential theft)

#### 8.5.6 Attack Automation

Autonomous ransomware attack across 4 phases:
1. Network reconnaissance: efficient identification of services/ports
2. Vulnerability identification: moderately effective
3. Exploit execution: entirely unsuccessful (0% success across 20–23 runs)
4. Post-exploitation: entirely unsuccessful

**Conclusion:** No successful end-to-end autonomous cyberattack capability.

#### 8.5.7 Uplift Testing (Cyber)

62 internal volunteers (31 expert, 31 novice), two-stage study:
1. Stage 1: challenge without LLM, with internet access
2. Stage 2: same + Llama 3 405B access

**Result:** Insignificant uplift over internet-only baseline for both novices and experts.

### 8.6 Chemical/Biological Weapons Uplift

**Study design:**
- 6-hour scenarios; teams of 2 participants generate fictitious operational plans
- Scenarios cover: agent acquisition, production, weaponization, delivery
- Participants: low-skill (no formal training) and moderate-skill (some training)
- Control: internet-only; LLM condition: internet + Llama 3 + web search + RAG + code execution
- RAG: keyword-search-generated dataset of hundreds of scientific papers pre-loaded
- SME evaluation: scientific accuracy, detail, detection avoidance, probability of success
- Robust Delphi process for bias mitigation in SME evaluations
- Preliminary study validated design including power analysis for sample size sufficiency

**Result:** No significant uplift in performance from Llama 3 usage. Holds for:
- Aggregate analysis (all LLM vs. control)
- Subgroup breakdowns (70B vs. 405B; chemical vs. biological)

**Risk assessment:** Low risk that Llama 3 release increases ecosystem risk for biological/chemical weapon attacks.

### 8.7 Red Teaming

#### 8.7.1 Attack Taxonomy Discovered

**Short/Long-Context English:**
- Multi-turn refusal suppression
- Hypothetical scenarios ("hypothetically...")
- Personas and role play
- Adding disclaimers/warnings as response priming
- Gradually escalating violation (multi-turn)

**Multilingual:**
- Language mixing within prompts/conversations
- Lower-resource languages → less safety fine-tuning coverage
- Slang and culture-specific references → miscomprehension

**Tool Use:**
- Unsafe tool chaining (benign + violating tool calls)
- Forcing tool use with specific input strings, fragmented/encoded text
- Modifying tool use parameters (word swapping, retrying, obfuscation)

**Child Safety:** Expert-led objective-based assessments across attack vectors, market-specific nuances.

### 8.8 System-Level Safety

#### 8.8.1 Llama Guard 3

**Architecture:** Llama 3 8B fine-tuned for safety classification (prompt + response classification).

**Taxonomy:** 14 hazard categories:
Child Sexual Exploitation, Defamation, Elections, Hate, Indiscriminate Weapons, Intellectual Property, Non-Violent Crimes, Privacy, Sex-Related Crimes, Sexual Content, Specialized Advice, Suicide & Self-Harm, Violent Crimes, Code Interpreter Abuse.

**Training data:**
- Starts from Llama Guard English data, expanded for multilingual + tool use
- Unsafe responses augmented via prompt engineering to bypass model refusals
- Labels from Llama 3 + human annotation + iterative noise reduction
- Human labels preferred for borderline prompts (slightly more accurate than LLM labels)

**Performance (Full Llama Guard, relative to base Llama 3 405B):**

| **Language** | **VR Change** | **FRR Change** |
|---|---|---|
| English | −86% | +102% |
| French | −59% | +29% |
| German | −77% | +37% |
| Hindi | −71% | +62% |
| Italian | −48% | +29% |
| Portuguese | −65% | +39% |
| Spanish | −60% | +27% |
| Thai | −51% | +39% |

**Average VR reduction: ~65%** across capabilities.

**Per-category VR reduction (English, Full Llama Guard):**

| **Category** | **VR Reduction** |
|---|---|
| Defamation | −100% |
| Elections | −100% |
| Intellectual Property | −100% |
| Non-Violent Crimes | −100% |
| Sexual Content | −100% |
| Hate | −91% |
| Sex-Related Crimes | −88% |
| Violent Crimes | −80% |
| Specialized Advice | −70% |
| Suicide & Self-Harm | −62% |
| Child Sexual Exploitation | −59% |
| Privacy | −60% |
| Indiscriminate Weapons | 0% |

**Quantized Llama Guard 3 (int8):**

| **Capability** | **F1 (Non-Quant.)** | **F1 (Quant.)** | **FPR (Non-Quant.)** | **FPR (Quant.)** |
|---|---|---|---|---|
| English | 0.939 | 0.936 | 0.040 | 0.040 |
| Multilingual | 0.862 | 0.851 | 0.033 | 0.031 |
| Tool Use | 0.825 | 0.827 | 0.176 | 0.155 |

**Conclusion:** int8 quantization (>40% size reduction) has negligible impact on classification performance.

#### 8.8.2 Prompt Guard

**Architecture:** mDeBERTa-v3-base (86M parameters), multi-label classifier.

**Classes:**
- Direct jailbreaks
- Indirect prompt injections

**Performance:**

| **Dataset** | **TPR** | **FPR** | **AUC** |
|---|---|---|---|
| Jailbreaks (in-distribution) | 99.9% | 0.4% | 0.997 |
| Injections (in-distribution) | 99.5% | 0.8% | 1.000 |
| OOD Jailbreaks (English) | 97.5% | 3.9% | 0.975 |
| Multilingual Jailbreaks | 91.5% | 5.3% | 0.959 |
| Indirect Injections (CyberSecEval) | 71.4% | 1.0% | 0.996 |

**Failure mode:** Indirect injection detection (71.4% TPR) is substantially weaker than direct jailbreak detection — fundamental challenge in distinguishing legitimate in-context instructions from injected ones.

#### 8.8.3 Code Shield

**Mechanism:** Inference-time filtering using Insecure Code Detector (ICD), a suite of static analysis tools across 7 programming languages. Detects insecure code generation before deployment.

---

## 9. Inference Optimization

### 9.1 Pipeline Parallelism

#### 9.1.1 Problem Formulation

Llama 3 405B in BF16 requires:

$$\text{Memory}_{\text{params}} = 405 \times 10^9 \times 2 \text{ bytes} = 810 \text{ GB}$$

A single 8×H100 node provides $8 \times 80 = 640$ GB HBM → **does not fit**. Two nodes (16 GPUs, 1280 GB) required.

#### 9.1.2 Parallelism Strategy

**Intra-node:** Tensor Parallelism (TP) with degree $T = 8$, leveraging high NVLink bandwidth ($900$ GB/s bidirectional per H100 pair).

**Inter-node:** Pipeline Parallelism (PP) with degree $P = 2$, leveraging lower-bandwidth inter-node connectivity (InfiniBand/RoCE, typically $400$ Gb/s).

**Total parallelism:** $\text{TP8} \times \text{PP2}$ across 16 GPUs.

#### 9.1.3 Micro-batching in Inference

In training, pipeline parallelism incurs **bubble overhead** from backward-pass pipeline flushes:

$$\text{Bubble fraction}_{\text{train}} = \frac{P - 1}{P + M - 1}$$

where $M$ is the number of micro-batches. During inference, **no backward pass** → no pipeline flush → no bubbles. Micro-batching thus enables concurrent execution of micro-batches across pipeline stages with minimal overhead.

**Configuration evaluated:** 2 micro-batches, 4096 input tokens, 256 output tokens.

#### 9.1.4 Throughput-Latency Analysis

**Pre-fill stage:**

| **Batch Size** | **Without Microbatching (tok/s)** | **With Microbatching (tok/s)** |
|---|---|---|
| 1 | ~2000 | ~2000 |
| 2 | ~3500 | ~4000 |
| 4 | ~5000 | ~6500 |
| 8 | ~6500 | ~8500 |

**Decode stage:**

| **Batch Size** | **Without Microbatching (tok/s)** | **With Microbatching (tok/s)** |
|---|---|---|
| 1 | ~50 | ~50 |
| 8 | ~250 | ~400 |
| 32 | ~600 | ~900 |
| 128 | ~900 | ~1300 |

**Trade-off:** Micro-batching increases throughput at same local batch size. Additional synchronization points introduce latency overhead, but overall Pareto frontier is strictly improved.

### 9.2 FP8 Quantization

#### 9.2.1 Quantization Scope

FP8 quantization applied to **most matrix multiplications in the feedforward network (FFN) layers**, accounting for ~50% of inference compute time. Self-attention layers are **not quantized**.

**FP8 format (E4M3 for weights, E5M2 for activations on H100):**

$$x_{\text{FP8}} = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x|)}{E4M3\_\text{max}}$$

where $s$ is the dynamic scaling factor.

#### 9.2.2 Dynamic Scaling

**Tensor-wise scaling:** Single scale factor $s$ per entire tensor:

$$s_{\text{tensor}} = \frac{\max(|\mathbf{X}|)}{448}$$

where 448 is the max representable value in E4M3.

**Row-wise scaling (adopted):** Per-row scale factor for both parameter and activation matrices:

$$s_i = \frac{\max(|\mathbf{X}_{i,:}|)}{448} \quad \forall i \in [1, \ldots, d_{\text{rows}}]$$

Row-wise quantization provides more granular scaling, reducing quantization error for matrices with heterogeneous magnitude distributions across rows.

#### 9.2.3 Quantization Error Mitigation

Three specific mitigations address Llama 3 405B's sensitivity to quantization:

**1. First/last layer exclusion:**

$$\text{Layers quantized} = \{l : 2 \leq l \leq L-1\}$$

where $L$ is the total number of Transformer layers. First and last layers remain in BF16.

**2. Scale factor upper bounding:**

$$s_i^{\text{clamped}} = \min(s_i, 1200)$$

High-perplexity tokens (e.g., dates) produce large activation values → high scaling factors → excessive underflows in FP8 representation. Clamping at 1200 prevents cascading decoding errors.

**3. Row-wise over tensor-wise quantization:** As described above.

#### 9.2.4 Quality Assessment

**Standard benchmarks:** FP8 performs on par with BF16 — but this is **insufficient** for detecting quantization effects. Standard benchmarks fail to capture distributional shifts in model output.

**Reward model distribution analysis:** 100,000 responses generated with both FP8 and BF16; reward model scores computed for each. The reward score distributions are compared:

$$D_{\text{KL}}\left(p_{\text{FP8}}(r) \| p_{\text{BF16}}(r)\right) \approx 0$$

The empirical reward score distributions for BF16 and FP8 row-wise quantization are nearly identical, confirming negligible quality degradation.

**Failure mode without mitigation:** When scale factors are not upper-bounded, model occasionally produces corrupted responses despite strong benchmark performance — benchmark scores mask distributional tail effects.

#### 9.2.5 Efficiency Results

**Pre-fill stage throughput improvement:** Up to **50%** throughput increase with FP8 vs. BF16 (TP8/PP2).

**Decode stage:** Substantially improved throughput-latency trade-off.

**Memory reduction:** FP8 reduces FFN parameter memory by ~50%:

$$\text{Memory}_{\text{FP8 FFN}} = \frac{\text{Memory}_{\text{BF16 FFN}}}{2}$$

This enables single-node (8×H100) deployment for the 405B model when FFN layers are quantized, eliminating inter-node communication overhead.

<img src="assets/llama_3_technical_synthesis_p12.png" alt="Inference optimization via FP8 row-wise quantization with topology, throughput, and error-mitigation details" width="100%" />

*Figure. FP8 row-wise quantization diagram, corresponding to the mitigation strategy, throughput gains, and benchmark-insensitivity warning described in Section 9.2.*

### 9.3 Pseudo-Algorithm: FP8 Inference

```
ALGORITHM: FP8InferenceLlama405B
INPUT: Model M with L layers, input tokens X, 
       max_scale = 1200
OUTPUT: Generated tokens Y

1. LOAD model parameters:
   - Layers 1 and L: BF16
   - Layers 2 to L-1, FFN weights: quantize to FP8 row-wise
     FOR each weight matrix W in FFN:
       FOR each row i:
         s_i ← min(max(|W[i,:]|) / 448, max_scale)
         W_fp8[i,:] ← round(W[i,:] / s_i)
         STORE s_i

2. PREFILL phase (process input tokens X):
   h ← embed(X)          // BF16
   FOR l = 1 TO L:
     // Self-attention (always BF16)
     h_attn ← SelfAttention_BF16(h, layer=l)
     
     // FFN
     IF l == 1 OR l == L:
       h_ffn ← FFN_BF16(h_attn, layer=l)
     ELSE:
       // Dynamic activation quantization
       a_fp8, s_act ← RowWiseQuantize(h_attn, max_scale)
       // FP8 GEMM with row-wise scaling
       h_ffn ← FP8_GEMM(a_fp8, W_fp8[l], s_act, s_weight[l])
       h_ffn ← cast_to_BF16(h_ffn)
     
     h ← LayerNorm(h_attn + h_ffn)
   
   STORE KV cache

3. DECODE phase (autoregressive generation):
   REPEAT:
     h ← embed(last_token)
     FOR l = 1 TO L:
       h_attn ← SelfAttention_BF16(h, KV_cache[l])
       IF l == 1 OR l == L:
         h_ffn ← FFN_BF16(h_attn, layer=l)
       ELSE:
         a_fp8, s_act ← RowWiseQuantize(h_attn, max_scale)
         h_ffn ← FP8_GEMM(a_fp8, W_fp8[l], s_act, s_weight[l])
         h_ffn ← cast_to_BF16(h_ffn)
       h ← LayerNorm(h_attn + h_ffn)
     logits ← LMHead_BF16(h)
     next_token ← sample(logits)
     UPDATE KV cache
   UNTIL next_token == EOS or max_length reached

4. RETURN generated tokens Y
```

### 9.4 Complexity Analysis

**Memory complexity (BF16 baseline):**

$$M_{\text{BF16}} = 2P + 2 \cdot n_{\text{layers}} \cdot 2 \cdot d_{\text{model}} \cdot n_{\text{seq}} \cdot n_{\text{heads}} \quad [\text{bytes}]$$

where $P$ is the parameter count.

**Memory complexity (FP8 FFN):**

$$M_{\text{FP8}} = 2P_{\text{attn}} + P_{\text{FFN}} + \text{scale\_storage} + \text{KV cache}$$

Scale storage overhead: $O(d_{\text{model}} \cdot n_{\text{layers}})$ — negligible relative to parameter memory.

**Compute complexity:**

FP8 GEMMs achieve approximately $2\times$ theoretical peak FLOPS on H100 TensorCores relative to BF16 GEMMs (989 TFLOPS FP8 vs. 989 TFLOPS BF16 on paper, but effective throughput improvement of ~1.5× due to memory bandwidth bounds and quantization overhead).

**Communication complexity (PP2 vs. PP1):**

$$T_{\text{comm}} = \frac{2 \cdot d_{\text{model}} \cdot B}{\text{BW}_{\text{inter-node}}} \cdot (P - 1)$$

FP8 quantization enabling PP1 eliminates this entirely.

---

## 10. Deployment Constraints and System-Level Considerations

### 10.1 Serving Topology

| **Configuration** | **GPUs** | **Precision** | **Throughput Regime** |
|---|---|---|---|
| TP8 / PP2 | 16 (2 nodes) | BF16 | Baseline |
| TP8 / PP2 + Microbatch | 16 (2 nodes) | BF16 | +30–50% throughput |
| TP8 / PP1 | 8 (1 node) | FP8 | Best throughput-latency |

<img src="assets/llama_3_405b_technical_blueprint_p13.png" alt="Deployment optimization and inference scaling for Llama 3 405B including pipeline parallelism, FP8 quantization, and KV-cache management" width="100%" />

*Figure. Deployment scaling view for 405B serving, corresponding to the serving-topology, memory-budget, and KV-cache constraints summarized in Section 10.*

### 10.2 Failure Modes in Deployment

- **FP8 without scale clamping:** Occasional corrupted responses undetectable by standard benchmarks
- **Inter-node pipeline parallelism:** Tail latency sensitivity to network jitter
- **KV cache growth:** At 128k context, KV cache dominates memory; FP8 does not address this (attention layers remain BF16)
- **Micro-batch synchronization:** Additional synchronization points increase per-request latency even as throughput improves

### 10.3 Observability

- **Reward model score distribution monitoring:** Preferred over benchmark-based quality assessment for detecting quantization-induced distributional drift
- **Violation rate monitoring:** System-level Llama Guard deployed as real-time classifier on input/output streams
- **Static analysis (Code Shield):** Inference-time code safety filter across 7 languages

### 10.4 Safety System Integration

**Multi-layer defense stack:**

```
User Input → Prompt Guard (86M, jailbreak/injection detection)
           → Llama Guard 3 Input Filter (8B, hazard classification)
           → Llama 3 405B (FP8, main model)
           → Llama Guard 3 Output Filter (8B, hazard classification)
           → Code Shield (static analysis, if code output)
           → Response to User
```

Each layer operates independently with category-level on/off granularity, enabling developers to customize the VR–FRR trade-off per use case.

---

## 11. Summary of Key Quantitative Results

### 11.1 Pre-trained Model Scaling

| **Metric** | **8B** | **70B** | **405B** |
|---|---|---|---|
| MMLU | 66.7 | 79.3 | 85.2 |
| MMLU-Pro | 37.1 | 53.8 | 61.6 |
| GSM8K | 57.2 | 83.7 | 89.0 |
| MATH | 20.3 | 41.4 | 53.8 |
| HumanEval (pre-train) | 37.2 | 58.5 | 61.0 |
| ARC-C | 79.7 | 92.9 | 96.1 |

### 11.2 Post-trained Model (pass@1 Python)

| **Metric** | **8B** | **70B** | **405B** |
|---|---|---|---|
| HumanEval | 72.6 | 80.5 | 89.0 |
| HumanEval+ | 67.1 | 74.4 | 82.3 |
| MBPP | 60.8 | 75.4 | 78.8 |
| MBPP EvalPlus | 72.8 | 86.0 | 88.6 |

### 11.3 Safety

| **Metric** | **8B** | **70B** | **405B** |
|---|---|---|---|
| Memorization (50-gram) | 0.26% | 0.60% | 1.13% |
| Memorization (1000-gram) | 1.11% | 3.56% | 3.91% |
| Code Interpreter Abuse Compliance | — | 3.8% | 10.4% |
| Prompt Injection Success Rate | — | — | 21.7% |

### 11.4 Inference Efficiency

| **Config** | **Prefill Throughput (8 batch)** | **Decode Throughput (128 batch)** |
|---|---|---|
| TP8/PP2 BF16 | ~6500 tok/s | ~900 tok/s |
| TP8/PP2 BF16 + Microbatch | ~8500 tok/s | ~1300 tok/s |
| FP8 Row-wise (TP8/PP1) | ~12000 tok/s | ~1500 tok/s |
