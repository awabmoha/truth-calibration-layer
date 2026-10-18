# Truth Calibration Layer (TCL): A Framework for Jointly Trained, Multi-Dimensional Trust Estimation in Language Models

**Awab Mohamed**\
Independent Research Preprint, Qatar\
Version 2 (draft for review), October 2026

## Abstract

Large language models (LLMs) are trained to predict the next token. That objective rewards fluency, and it gives no direct pressure toward knowing when an output is reliable. Prior work shows that frozen hidden states already carry signal about answer correctness, and that post-hoc probes can read it out. I propose the Truth Calibration Layer (TCL), a framework in which a lightweight trust head is trained **jointly** with the language model, under the objective `L_LM + lambda * L_trust`, to emit a structured trust vector alongside generated text. The vector has four operationally defined dimensions: correctness, self-consistency, grounding, and an evidence-density proxy. Each dimension is defined by a reproducible labeling procedure, not by an unobservable property. My central hypothesis is that joint training makes reliability information more accessible and more robust than a frozen probe trained on identical data, without degrading generation. This version fixes an inconsistency in version 1, which described the base model as frozen while also defining a joint objective. It also replaces two unobservable trust dimensions with measurable ones and states six falsifiable hypotheses with predeclared tests. I claim no implementation or validation of full TCL here. A companion diagnostic, TCL-v0, tests only the supporting assumption that frozen hidden states carry correctness signal.

**Keywords:** confidence calibration, uncertainty estimation, hidden-state probing, joint training objective, trust estimation, selective prediction, language models

## 1. Introduction

Modern LLMs produce fluent, well-formed text across many domains. That fluency hides a basic limitation: they are not trained to know what they do not know. The standard objective maximizes the conditional probability of each token given its context. It is indifferent to whether the generated sequence is true. A model trained only this way can assign high probability to a well-formed but false sentence, and it can sound equally confident when recalling a well-documented fact and when confabulating.

Two related failure modes follow. **Hallucination** is the generation of plausible but incorrect or unverifiable content (Maynez et al., 2020; Bang et al., 2023). **Miscalibration** is a mismatch between expressed confidence and actual accuracy (Guo et al., 2017). Instruction tuning and preference optimization (Ouyang et al., 2022; Bai et al., 2022) improve helpfulness, but they can make token-level probabilities a worse guide to correctness (Tian et al., 2023).

A substantial line of work shows that the information needed to detect unreliable outputs is often already inside the model. Linear probes on frozen hidden states predict whether a statement is true or whether an answer is correct (Azaria & Mitchell, 2023; Marks & Tegmark, 2024; Orgad et al., 2025). Unsupervised methods find truth-like directions (Burns et al., 2023). Probes can even approximate expensive sampling-based uncertainty from a single forward pass (Kossen et al., 2024). Models can also be fine-tuned to predict whether they know an answer (Kadavath et al., 2022; Kapoor et al., 2024).

My own diagnostic, TCL-v0, replicates the probing finding on three small open models. It also shows the finding's limits: hidden-state probes improve the **ranking** of correct versus incorrect answers over calibrated raw confidence, but their apparent calibration advantage largely disappears once raw confidence receives the same post-hoc calibration (Section 7).

TCL asks the next question. **If reliability information is present but only partly accessible to a post-hoc probe, does training the model to expose it make it more accessible, more robust, and richer than a single confidence score?**

### 1.1 Contributions

1. **A consistent formulation of TCL** as a jointly trained trust head, with the frozen-probe variant as its explicit baseline. This resolves the frozen-versus-joint contradiction in version 1 (Section 4).
2. **An operational four-dimensional trust vector** in which every dimension is defined by a reproducible labeling procedure: correctness, self-consistency, grounding, and an evidence-density proxy (Section 5).
3. **A theoretical analysis** of what joint training can and cannot add: it can change how *accessible* reliability information is to a low-capacity head; it cannot create information the input does not contain (Section 6).
4. **Six falsifiable hypotheses** with predeclared metrics, baselines, controls, and pass/fail thresholds (Section 8).

### 1.2 Changes From Version 1

| Version 1 | Version 2 | Reason |
|---|---|---|
| Base transformer frozen (Sec. 4.1) and joint objective `L_LM + lambda * L_TCL` (Sec. 4.4) | Backbone adapted via LoRA under the joint objective; frozen head is the baseline | With a frozen backbone, `L_LM` has no gradient and the objective reduces to a probe |
| Trust vector: confidence, evidence density, contradiction risk, source diversity | Correctness, self-consistency, grounding, evidence-density proxy | Evidence density and source diversity had no label procedure; source diversity is not observable without provenance data |
| ECE as the single primary metric | AUROC and AURC primary; ECE and Brier after uniform post-hoc calibration | ECE can be improved by monotone recalibration alone; it does not test whether hidden states add information |
| Related work limited to calibration, Bayesian methods, RAG | Adds hidden-state probing, sampling-based uncertainty, verbalized confidence, fine-tuned uncertainty | TCL must be positioned against its closest prior work |
| Claim-level extraction in the core architecture | Unit of analysis is a short answer; claim-level trust is a later stage (H6) | Claim extraction is an unsolved sub-problem that should not block testing the core idea |

## 2. Related Work

**Calibration of neural networks.** Guo et al. (2017) showed that modern networks tend to be overconfident and that temperature scaling is a strong post-hoc fix. Platt scaling (Platt, 1999) and isotonic regression (Zadrozny & Elkan, 2002) are the other standard recalibrators. Zhao et al. (2021) showed that few-shot prompting introduces systematic biases that can be corrected by contextual calibration. All of these are monotone transformations of an existing score. They can improve calibration, but they cannot improve the ranking of correct versus incorrect outputs. That distinction is central to how I evaluate TCL.

**Uncertainty quantification.** Bayesian approaches (Gal & Ghahramani, 2016) and deep ensembles (Lakshminarayanan et al., 2017) estimate uncertainty but are costly at LLM scale. Conformal prediction gives distribution-free coverage guarantees (Shafer & Vovk, 2008). Selective classification studies when a model should abstain (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017), and its risk-coverage framework is the natural way to evaluate a trust signal that is used to decide whether to answer.

**Sampling-based uncertainty for LLMs.** Semantic entropy measures uncertainty over the meanings of sampled answers instead of their surface forms (Kuhn et al., 2023; Farquhar et al., 2024). SelfCheckGPT detects hallucinations by checking consistency across samples (Manakul et al., 2023). These methods are strong but need many generations per query. My self-consistency dimension (Section 5.2) uses this family as its label source, and TCL tests whether a trained head can predict it from a single pass.

**Self-evaluation and verbalized confidence.** Kadavath et al. (2022) showed that models can evaluate their own answers (P(True)) and can be trained to predict whether they know an answer (P(IK)). Lin et al. (2022) taught models to express uncertainty in words. Mielke et al. (2022) and Band et al. (2024) trained models to make their linguistic confidence match their accuracy. Tian et al. (2023) found that verbalized confidence can be better calibrated than token probabilities in RLHF-tuned models.

**Hidden-state probing.** Azaria and Mitchell (2023) trained classifiers on hidden states to detect false statements. Burns et al. (2023) found truth-like directions without supervision. Marks and Tegmark (2024) found linear structure separating true and false statements. Li et al. (2023) intervened along such directions at inference time. Orgad et al. (2025) showed that correctness signal is concentrated in specific answer tokens and that error detectors built on hidden states fail to generalize across datasets. Kossen et al. (2024) trained probes to predict semantic entropy. This literature establishes the premise that TCL-v0 replicates.

**Fine-tuning for uncertainty.** Kapoor et al. (2024) argue that prompting alone is insufficient for good calibration and that training through a model's features is necessary. They report that fine-tuning on about a thousand graded examples, made tractable with LoRA, gives uncertainty estimates that generalize well. This is the closest prior work to TCL's central hypothesis.

**How TCL differs.** TCL builds on the fine-tuning result rather than claiming it as new. It differs in three ways:

1. **Controlled isolation of the joint objective.** TCL compares a frozen head, trust-only fine-tuning, LM-only fine-tuning, and joint fine-tuning on identical data (Section 8, H3). This separates "training the backbone helps" from "training it *jointly with the LM objective* helps" and from "any fine-tuning changes accuracy."
2. **An explicit generation-preservation constraint.** The `L_LM` term is there to anchor generation. Generation quality is a pass/fail criterion, not an afterthought.
3. **A multi-dimensional, operationally defined trust vector** whose dimensions are tested for learnability and non-redundancy (Section 5, H4).

**Retrieval augmentation** (Lewis et al., 2020) grounds outputs in external evidence at inference time. It is complementary to TCL: the grounding dimension (Section 5.3) estimates whether an answer is supported by provided context. It does not replace retrieval.

## 3. Problem Formulation

### 3.1 The Standard Objective

An autoregressive language model with parameters `theta` is trained to maximize sequence log-likelihood:

```text
L_LM(theta) = - sum_t log P_theta(x_t | x_<t)
```

This objective is internal to the model's own distribution over tokens. Calibration, by contrast, relates the model's confidence to the external world. Nothing in `L_LM` forces the probability of a sequence to match the probability that its content is correct. A false statement that occurs frequently in training data is rewarded just like a true one.

### 3.2 The Trust Estimation Problem

Let `x` be an input (question, optionally with context) and `a` a generated answer. Let `y = (y_1, ..., y_D)` be a vector of reliability labels for the pair `(x, a)`, with each `y_d` defined by a labeling procedure in Section 5. I want a function `tau(x, a)` in `[0,1]^D` such that each component `tau_d` is:

1. **Discriminative:** it ranks pairs with `y_d = 1` above pairs with `y_d = 0` (measured by AUROC).
2. **Calibrated:** after at most a monotone post-hoc map fitted on validation data, `P(y_d = 1 | tau_d = p) ≈ p` (measured by ECE and Brier score).
3. **Useful for selective prediction:** abstaining on low-`tau` answers reduces the error rate of the answers that remain (measured by AURC).

For the correctness dimension, property 1 is what a post-hoc recalibration of raw confidence cannot provide, so it is the primary test of whether hidden states add information.

### 3.3 Formal Problem Statement

```text
Given a pretrained language model P_theta(token | context), learn parameters (theta', phi)
such that the augmented model emits

    P_theta'(token | context)  and  tau_phi(h_theta'(x, a))

where tau is the trust vector, h_theta' is the (possibly adapted) hidden representation,
and:
  (i)  tau improves discrimination and selective prediction over baselines that use
       only theta's output probabilities, sampling, or a frozen-representation probe;
  (ii) generation quality under theta' is not meaningfully worse than under theta.
```

## 4. The Truth Calibration Layer

### 4.1 Architecture

TCL adds a trust head to a transformer language model. The token-generation pathway is unchanged in structure. The backbone is adapted with low-rank adapters (LoRA; Hu et al., 2022), so only a small number of parameters change and generation stays close to the base model.

```text
H        = Transformer_theta'(x, a)            # hidden states, adapted backbone
logits   = OutputHead(H)                       # standard generation
h_bar    = Pool_l(H, span(a))                  # pooled answer-span state at layer l
tau      = sigmoid(W_2 * GELU(W_1 * h_bar))    # trust head, D outputs
output   = (tokens, tau)
```

**Pooling.** `Pool_l` averages the hidden states of the answer tokens at layer `l`. Orgad et al. (2025) found correctness signal concentrated in answer tokens, and TCL-v0 used final-layer answer-mean pooling. The choice of `l` is a hyperparameter selected on validation data.

**Trust head.** A two-layer MLP with hidden width 256 and one sigmoid output per trust dimension. It is deliberately small, so that improvements must come from what the representation exposes, not from head capacity.

### 4.2 Two Variants

| Variant | Backbone | Objective | Role |
|---|---|---|---|
| **TCL-F** (frozen) | Frozen (`theta' = theta`) | `L_trust` only, head parameters `phi` | Baseline and cheapest deployable form. Equivalent to a supervised probe. TCL-v0 is a single-dimension instance. |
| **TCL-J** (joint) | LoRA-adapted | `L_LM + lambda * L_trust`, parameters `phi` and LoRA weights | The central proposal. |

Version 1 described TCL-F's architecture with TCL-J's objective. Version 2 separates them. **The central hypothesis of TCL concerns TCL-J.** TCL-F is valuable on its own, but it is not new (Section 2).

### 4.3 Joint Training Objective

```text
L_total(theta', phi) = L_LM(theta') + lambda * L_trust(theta', phi)

L_trust = sum_d  w_d * (1 / |S_d|) * sum_{i in S_d}  ell_d(tau_d(x_i, a_i), y_{i,d})
```

- `S_d` is the set of training examples that have a label for dimension `d`. Not every example has every label; for example, grounding is defined only when context is present. Missing labels are masked, not imputed.
- `ell_d` is binary cross-entropy for every dimension. For dimensions with soft labels in `[0,1]` (self-consistency, evidence density), BCE with soft targets is still a proper scoring rule.
- `w_d` are dimension weights, fixed to 1 unless a validation-set ablation justifies otherwise.
- `lambda` controls the trade-off. I test `lambda` in {0.1, 0.5, 1.0}.

**What `L_LM` is computed on.** `L_LM` is the negative log-likelihood of the **reference** answer given the question (teacher forcing). I do not compute it on the model's own sampled answers, because that would reinforce wrong answers. Drift in general language modeling is measured, not trained against: the generation-quality bar includes perplexity on held-out text. Training on reference answers can by itself change accuracy, which is why the experimental design includes an LM-only control (arm D).

**What the trust labels are computed on.** Trust labels refer to answers generated by the model. In v1 experiments, I generate answers once with the base model and label them. Once the backbone is adapted, the model's own answers can drift away from these labels. I handle this by evaluating every trained model on answers it regenerates itself, labeled under the same rules. Iterative relabeling during training is left to later work.

### 4.4 Inference

At inference, the model generates an answer normally, then runs one extra forward pass of the trust head on the pooled answer span. For TCL-J, generation uses the adapted backbone. For TCL-F, generation is identical to the base model. In both cases the extra cost is one small MLP evaluation per answer, compared with roughly ten full generations for semantic entropy.

## 5. The Trust Vector

Each dimension below is defined by a reproducible labeling procedure. These are **operational definitions**: they define what the trust head is trained and tested to predict. None of them is claimed to measure truth in a philosophical sense. A dimension without a label procedure is out of scope. That is why source diversity, present in version 1, is removed until a retrieval or provenance setup can supply labels.

| Dimension | Symbol | Question it answers | Label type |
|---|---|---|---|
| Correctness | `tau_c` | Is this answer correct? | Binary |
| Self-consistency | `tau_k` | Would the model give a semantically equivalent answer if asked again? | Soft, [0,1] |
| Grounding | `tau_g` | Is this answer supported by the provided context? | Binary (context present only) |
| Evidence density (proxy) | `tau_e` | How well represented is the question's subject in web-scale data? | Soft, [0,1] |

### 5.1 Correctness

**Label.** `y_c = 1` if the answer matches an accepted reference answer under the normalized matching rules used in TCL-v0, with targeted manual review of high-confidence disagreements.

**Datasets.** TriviaQA (Joshi et al., 2017), Natural Questions open (Kwiatkowski et al., 2019), SQuAD (Rajpurkar et al., 2016).

**Interpretation.** Directly comparable with prior probing and calibration work.

### 5.2 Self-Consistency (Contradiction Risk)

**Label.** For each question, sample `N = 10` additional answers at temperature 1.0. Cluster the answers by bidirectional entailment using an NLI model, following semantic entropy (Kuhn et al., 2023). The label is the fraction of samples in the same meaning cluster as the evaluated answer:

```text
y_k = |{ j : a_j equivalent to a }| / N
```

Contradiction risk is `1 - y_k`.

**Interpretation.** This dimension measures the model's own stability, not external truth. A model can be consistently wrong. It is valuable because it is cheap to label at scale, because it is known to correlate with hallucination (Farquhar et al., 2024), and because predicting it from a single pass removes the cost of sampling at inference time.

### 5.3 Grounding

**Label.** For questions that come with a context passage, `y_g = 1` if the passage entails the answer, judged by an NLI model and audited on a manually reviewed sample. Unanswerable questions from SQuAD 2.0 (Rajpurkar et al., 2018) are a source of negatives: any non-abstaining answer to them is ungrounded.

**Interpretation.** Grounding is distinct from correctness. An answer can be correct but unsupported by the given passage (recalled from memory), or supported by a passage that is itself wrong.

### 5.4 Evidence Density (Proxy)

**Label.** For PopQA (Mallen et al., 2023), each question's subject entity has a popularity value measured by monthly Wikipedia page views. The label is the rank-normalized log page-view count, in `[0,1]`.

**Interpretation.** This replaces version 1's "estimated strength of training-data support," which cannot be observed without access to training data. Popularity is a documented proxy: accuracy on factual questions rises with how often the relevant entity appears in pretraining-scale data (Mallen et al., 2023; Kandpal et al., 2023). The label is a property of the question's subject, not of the answer. Learning it tests whether the model internally represents how familiar a subject is. That signal is useful for deciding when to retrieve instead of answering from memory.

### 5.5 Expected Relationships

The dimensions are expected to be correlated. Correctness and self-consistency, in particular, will be positively related. A dimension is useful to TCL only if it carries information the others do not. H4 therefore requires both that a dimension is learnable and that it is not redundant with correctness (Section 8).

## 6. Theoretical Analysis

### 6.1 Proper Scoring and Calibration

Binary cross-entropy is a strictly proper scoring rule (Gneiting & Raftery, 2007). For a fixed representation `h`, the head that minimizes expected BCE outputs `P(y_d = 1 | h)`. With enough data and capacity, the trust head is therefore calibrated **with respect to the information in `h`, on the training distribution**. Two consequences follow:

- Calibration on the training distribution is not calibration under shift. Transfer has to be tested, not assumed (H2, H5).
- A perfectly calibrated head can still be useless if `h` carries little information about `y_d`. Calibration alone says nothing about discrimination, which is why AUROC and AURC are primary.

### 6.2 What Joint Training Can and Cannot Add

The representation `h_theta(x, a)` is a deterministic function of the input. By the data processing inequality, no choice of `theta` can make `h` contain more information about `y` than `(x, a)` itself contains. **Joint training cannot create reliability information.**

What it can change is **accessibility**. A frozen backbone's representation is shaped only by next-token prediction. Reliability information may be present but entangled, so that a low-capacity head trained on a limited number of labels cannot extract it, or extracts features that do not transfer. Joint training lets the gradient of `L_trust` reshape the representation so that this information becomes easier for a small head to read, and potentially more robust across distributions.

This also defines what TCL-J must beat. Because `theta' = theta` is a feasible point of the joint optimization, the joint objective can always match TCL-F on the **training** trust loss, up to the cost of the `L_LM` term. Whether it does better on **held-out** data is an empirical question about generalization, not something the theory guarantees. H3 tests exactly that.

### 6.3 The Role of `L_LM`

Without `L_LM`, adapting the backbone only for trust prediction can degrade generation: the representation drifts toward what helps the head and away from what helps next-token prediction. `L_LM` acts as an anchor. The trade-off is controlled by `lambda`. If no `lambda` lets TCL-J improve trust estimation while keeping generation quality within tolerance, TCL-J fails on its own terms.

### 6.4 Why Fused-Minimum Scores Are Not Used

TCL-v0 reported a conservative score `min(raw, probe)`. Taking a minimum of two scores lowers confidence everywhere. That mechanically reduces counts of high-confidence errors without necessarily improving discrimination, and the result is not a proper probability estimate. TCL-v0's baseline checkpoint confirmed that raw-only recalibration achieves similar reductions. Version 2 keeps `min(raw, probe)` only as a baseline and uses calibrated single-score outputs for all headline results.

### 6.5 Assumptions

- **A1.** Hidden representations of the answer span carry information about each trust label. (Tested for correctness by TCL-v0; for other dimensions by H4.)
- **A2.** Labels are accurate enough that label noise does not dominate. (Mitigated by manual review and NLI audits.)
- **A3.** LoRA capacity is enough to reshape representations without large changes to generation. (Tested by the generation-quality criterion in H3.)
- **A4.** Base-model labels remain approximately valid for the adapted model's answers. (Tested by regenerating and relabeling answers at evaluation time.)

## 7. Evidence From TCL-v0

TCL-v0 is a frozen, single-dimension instance of TCL-F: final-layer answer-mean hidden states, a logistic-regression probe, and correctness labels. It was evaluated on Qwen2.5-0.5B-Instruct, Phi-3.5-mini-instruct, and Gemma 2B-it, on SQuAD (with context) and TriviaQA (`rc.nocontext`), in six reviewed 1,000-example runs with 200 held-out test examples each.

| Finding | Status |
|---|---|
| Hidden-state probe ranks correct vs incorrect answers better than the best raw-only calibrated baseline (AUROC) | 5 of 6 runs; not yet tested with confidence intervals |
| Hidden-state score has better Brier score than raw-only calibration | 5 of 6 runs with class-balanced fits; 2 of 6 with unweighted fits |
| Hidden-state score has better ECE than raw-only calibration | 4 of 6 runs with class-balanced fits; 0 of 6 with unweighted fits |
| `min(raw, probe)` reduces high-confidence errors | Yes, but raw-only recalibration achieves similar or larger reductions |
| Clear negative case | Phi TriviaQA: raw-only calibration is stronger on all metrics |

Class-balanced fitting shifts a model's effective base rate and distorts calibration metrics. For example, on Phi SQuAD the best raw-only calibrated ECE is 0.307 with balanced weights and 0.027 without. The unweighted comparison is the fair one for calibration.

**What this supports.** Frozen hidden states add **ranking** information about correctness beyond raw confidence in most tested settings. This is consistent with prior probing work and supports assumption A1 for the correctness dimension.

**What it does not support.** It does not show that frozen probes give better-calibrated probabilities than recalibrated raw confidence. It tests nothing about joint training, other trust dimensions, or transfer.

**Implication for TCL.** If frozen probes improve ranking but their extracted signal is limited, that is the situation in which TCL-J's accessibility argument (Section 6.2) predicts a benefit from joint training. TCL-v0 motivates H3; it does not test it.

## 8. Hypotheses and Validation Protocol

### 8.1 Hypotheses

| ID | Hypothesis | Pass criterion |
|---|---|---|
| H1 | Frozen hidden states improve correctness ranking beyond calibrated raw confidence. | Probe AUROC exceeds the best calibrated raw baseline with a 95% bootstrap CI on the difference excluding zero, in at least 4 of 6 runs. |
| H2 | Frozen-probe signal transfers across QA datasets with bounded degradation. | A probe trained on one dataset retains at least 70% of its in-distribution AUROC gain over raw confidence on another. |
| H3-T (primary) | A jointly trained trust head transfers across datasets better than a frozen head trained on identical data. | Trained on TriviaQA and tested on SQuAD, and the reverse: TCL-J's AUROC exceeds arm A's with CI excluding zero in both directions, and TCL-J meets the generation-quality bar (answer-accuracy drop of at most 1 point, held-out perplexity increase of at most 3%). |
| H3-J (secondary) | The joint objective beats trust-only fine-tuning, at equal or better generation quality. | Same transfer test: TCL-J's AUROC exceeds arm B's with CI excluding zero in both directions, and TCL-J meets the generation-quality bar. |
| H4 | At least one additional trust dimension is learnable and non-redundant. | The dimension reaches AUROC of at least 0.70, has correlation below 0.7 with `tau_c`, and adding it improves AURC for selective prediction with CI excluding zero. |
| H5 | Joint-trained trust estimates generalize to a non-QA task and a larger model. | H3 and H4 conclusions hold on GSM8K (Cobbe et al., 2021) and on a 7B-8B model, within CIs. |
| H6 | Trust can be estimated at the level of individual claims in long-form text. | Per-claim trust predicts FActScore-style claim labels (Min et al., 2023) with AUROC of at least 0.70. |

H3 is the core test of TCL. Transfer (H3-T) is primary because frozen probes fail to transfer (Orgad et al., 2025; Phase 1 of this program), so that is where a trained head can show an advantage. H1 and H2 concern the frozen foundation; H4-H6 extend the framework.

### 8.2 Experimental Arms for H3-T and H3-J

All arms use the same training questions, the same labels, and the same trust-head architecture.

| Arm | Backbone | Objective | What it controls for |
|---|---|---|---|
| A | Frozen | `L_trust` (head only) | Strong frozen baseline (TCL-F) |
| B | LoRA | `L_trust` only | Whether *any* backbone adaptation for trust helps, without the LM anchor |
| C | LoRA | `L_LM + lambda * L_trust` | TCL-J, the proposal |
| D | LoRA, then frozen | `L_LM` only, then a head on the adapted backbone | Whether LM fine-tuning alone changes trust signal or accuracy |

If C beats A but not B, the benefit comes from adapting the backbone, not from the joint objective. If C beats A and B but D matches C, the benefit comes from LM fine-tuning. Only if C beats all three does the evidence support the joint objective specifically.

**Reading a tie between C and B (fixed in advance).** If C and B are indistinguishable on trust but B fails the generation-quality bar while C meets it, the joint objective's role is to preserve generation while training for trust, and I claim no more than that. If both meet the bar and tie, the joint objective adds nothing measurable beyond trust-only fine-tuning at this scale.

### 8.3 Metrics

- **Primary:** AUROC; area under the risk-coverage curve (AURC); accuracy at 50% and 80% coverage.
- **Secondary:** ECE (15 equal-mass bins) and Brier score, computed after fitting the same post-hoc calibrator on validation data for every method.
- **Generation quality:** exact match / F1 on held-out questions; perplexity on held-out text.

### 8.4 Baselines

| Baseline | Description |
|---|---|
| Raw likelihood | Length-normalized sequence probability of the generated answer |
| Calibrated raw likelihood | Best of temperature, Platt, and isotonic scaling, fitted on validation |
| P(True) | The model's probability that its own answer is true (Kadavath et al., 2022) |
| Semantic entropy | Entropy over meaning clusters of 10 samples (Kuhn et al., 2023) |
| Frozen probe | TCL-F (arm A) |

### 8.5 Statistical Protocol

- At least three random seeds for every trained method.
- 1,000-sample bootstrap 95% confidence intervals on every reported metric and on differences between methods.
- A method "wins" only when the CI of the difference excludes zero.
- Fixed train, validation, and test splits, committed before each experiment. Test sets are evaluated once per stage.
- Negative results are reported in full.

### 8.6 Staging

1. Settle H1-H2 on existing TCL-v0 records, plus sampling baselines.
2. Test H3-T and H3-J on Qwen2.5-0.5B-Instruct with LoRA, correctness dimension only, training on TriviaQA and on SQuAD and testing transfer between them.
3. Test H4, adding self-consistency first, then grounding, then evidence density.
4. Test H5 on GSM8K and a 7B-8B model.
5. Test H6 on long-form generation.

Each stage begins only after the previous stage's gate has been evaluated. If H3-T fails, TCL continues as a frozen multi-dimensional trust head (TCL-F), and its value must come from H4 and H5.

## 9. Limitations and Threats to Validity

**Benchmark overfitting.** The trust head may learn correlations between dataset artifacts and labels rather than general reliability. Cross-dataset transfer (H2) and non-QA evaluation (H5) are designed to detect this. Orgad et al. (2025) already report that hidden-state error detectors fail to generalize across datasets, so this threat is real.

**Label validity.** Each dimension is only as good as its labeling procedure. Answer matching misses paraphrases, NLI models make errors, and popularity is an indirect proxy for training-data exposure. I mitigate this with manual review and NLI audits, and by describing dimensions as operational definitions, not measurements of truth.

**Self-consistency is not truth.** A model can be confidently and consistently wrong. The self-consistency dimension estimates stability, and it has to be read alongside correctness.

**Label staleness under adaptation.** Under TCL-J the model's answers change, so base-model labels may not describe them. Evaluation on regenerated answers detects this but does not remove it from training.

**Scale.** Initial experiments use models of 0.5B-4B parameters. Results may not hold at larger scales: Kadavath et al. (2022) find that larger models are well calibrated on some question formats, so calibration behavior depends on both scale and format.

**Overinterpretation.** Better AUROC or ECE on evaluated benchmarks would show better reliability estimation on those distributions. It would not show that a model is truthful, that hallucination is solved, or that trust scores are safe to use without human oversight.

## 10. Conclusion

TCL proposes that language models should emit structured trust estimates alongside their outputs, and that this ability is best learned jointly with generation instead of being added afterwards. Version 2 makes this proposal testable. It separates the jointly trained proposal (TCL-J) from its frozen baseline (TCL-F), defines every trust dimension by a labeling procedure, positions TCL against the probing and fine-tuned uncertainty literature, and states six hypotheses with predeclared tests. The existing TCL-v0 evidence supports the frozen foundation for one dimension: hidden states improve correctness ranking beyond calibrated raw confidence in most tested settings. The core claim, that joint training makes reliability information more accessible without harming generation, is untested. It is the next experiment.

## References

[1] Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, 967-976.

[2] Bai, Y., et al. (2022). Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv:2204.05862.

[3] Band, N., Li, X., Ma, T., & Hashimoto, T. (2024). Linguistic calibration of long-form generations. In *ICML*.

[4] Bang, Y., et al. (2023). A multitask, multilingual, multimodal evaluation of ChatGPT on reasoning, hallucination, and interactivity. In *Proceedings of IJCNLP-AACL 2023*, 675-718.

[5] Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. In *ICLR*.

[6] Cobbe, K., et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.

[7] El-Yaniv, R., & Wiener, Y. (2010). On the foundations of noise-free selective classification. *Journal of Machine Learning Research*, 11, 1605-1641.

[8] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625-630.

[9] Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In *ICML*.

[10] Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *NeurIPS*.

[11] Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *Journal of the American Statistical Association*, 102(477), 359-378.

[12] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *ICML*.

[13] Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. In *ICLR*.

[14] Joshi, M., Choi, E., Weld, D., & Zettlemoyer, L. (2017). TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *ACL*.

[15] Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.

[16] Kandpal, N., Deng, H., Roberts, A., Wallace, E., & Raffel, C. (2023). Large language models struggle to learn long-tail knowledge. In *ICML*.

[17] Kapoor, S., et al. (2024). Large language models must be taught to know what they don't know. In *NeurIPS*.

[18] Kossen, J., et al. (2024). Semantic entropy probes: Robust and cheap hallucination detection in LLMs. arXiv:2406.15927.

[19] Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In *ICLR*.

[20] Kwiatkowski, T., et al. (2019). Natural Questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7, 453-466.

[21] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In *NeurIPS*.

[22] Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. In *NeurIPS*.

[23] Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. In *NeurIPS*.

[24] Lin, S., Hilton, J., & Evans, O. (2022). Teaching models to express their uncertainty in words. *Transactions on Machine Learning Research*.

[25] Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *ACL*.

[26] Manakul, P., Liusie, A., & Gales, M. J. F. (2023). SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In *EMNLP*.

[27] Marks, S., & Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In *COLM*.

[28] Maynez, J., Narayan, S., Bohnet, B., & McDonald, R. (2020). On faithfulness and factuality in abstractive summarization. In *ACL*.

[29] Mielke, S. J., Szlam, A., Dinan, E., & Boureau, Y.-L. (2022). Reducing conversational agents' overconfidence through linguistic calibration. *Transactions of the Association for Computational Linguistics*, 10, 857-872.

[30] Min, S., et al. (2023). FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In *EMNLP*.

[31] Orgad, H., et al. (2025). LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In *ICLR*.

[32] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. In *NeurIPS*.

[33] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*.

[34] Rajpurkar, P., Jia, R., & Liang, P. (2018). Know what you don't know: Unanswerable questions for SQuAD. In *ACL*.

[35] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. In *EMNLP*.

[36] Shafer, G., & Vovk, V. (2008). A tutorial on conformal prediction. *Journal of Machine Learning Research*, 9, 371-421.

[37] Tian, K., et al. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In *EMNLP*.

[38] Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. In *KDD*.

[39] Zhao, Z., Wallace, E., Feng, S., Klein, D., & Singh, S. (2021). Calibrate before use: Improving few-shot performance of language models. In *ICML*, PMLR 139, 12697-12706.
