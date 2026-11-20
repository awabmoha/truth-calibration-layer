# Joint Training Does Not Make Trust Heads Transfer: A Preregistered Negative Result

**Awab Mohamed**

Draft, November 20, 2026. Code, data, run records, and the locked protocols: https://github.com/awabmoha/truth-calibration-layer

## Abstract

A trust head is a small model that reads a language model's hidden states and predicts whether its answer is correct. A natural hope is that training this head *jointly* with the language model, under `L_LM + lambda * L_trust`, will shape the backbone's representations so that trust estimates generalize better than a head on a frozen model. I test this with a preregistered comparison of four arms on identical data: a frozen-backbone head (A), LoRA fine-tuning for trust only (B), joint LoRA fine-tuning (C), and LoRA fine-tuning for language modeling only followed by a head (D). The model is Qwen2.5-0.5B-Instruct, trained on TriviaQA or SQuAD and tested on the other dataset, with 3 seeds per arm and paired bootstrap confidence intervals. The pass rule, fixed before training, required C to beat A in both transfer directions while keeping generation quality. **It fails.** Joint training beats the frozen head from TriviaQA to SQuAD (+0.022 AUROC) but is significantly worse from SQuAD to TriviaQA (-0.038), and its gain in the first direction is matched by language-model fine-tuning alone (+0.031). After the analysis I found an implementation flaw: the language-modeling target lacked the end-of-turn token, so fine-tuned models answered in 9-20 words instead of 2. I fixed it and ran the one rerun my protocol allowed. Answers returned to 1-2 words, but the transfer result did not change (+0.034 and -0.038; LM-only fine-tuning +0.041 and +0.002). In-distribution, joint training improves on trust-only fine-tuning by about 0.01 AUROC but never beats LM-only fine-tuning. Across every arm and both runs, a head trained on one dataset's correctness labels transfers far below the model's raw token confidence (0.56-0.65 vs 0.71-0.76) and semantic entropy (0.77), which are trained on neither. At this scale, joint training does not solve the transfer problem of learned trust estimators.

## 1. Introduction

Language models produce wrong answers as fluently as right ones. A common response is to estimate, for each answer, how likely it is to be correct, and to use that estimate to decide whether to show, flag, or withhold the answer. Many such estimators read the model's hidden states with a trained probe (Azaria & Mitchell, 2023; Marks & Tegmark, 2024; Orgad et al., 2025). They often work well on the data they were trained on, and poorly on data from a different distribution (Orgad et al., 2025).

One proposed remedy is to stop treating the model as fixed: train the trust head and the backbone together, so the model learns representations that support trust estimation while a language-modeling loss keeps its generation intact. Fine-tuning through a model's features has been shown to give better uncertainty estimates than prompting (Kapoor et al., 2024), and joint training of a detection head with LoRA adapters has been reported to make hallucinated and faithful tokens more separable (Ridder et al., 2026). But neither work isolates the joint objective: the comparisons that would separate "joint training helps" from "any fine-tuning helps" or "language-model fine-tuning helps" are missing, and none is made across datasets with seeds and confidence intervals.

This paper makes that comparison. It is part of my Truth Calibration Layer (TCL) research program, in which the joint version (TCL-J) is the central claim, and every test has its hypotheses, arms, and decision rules written and committed before any result exists. The question is narrow: **does a trust head trained jointly with the backbone rank correct above incorrect answers better than a frozen head, on a dataset it was not trained on?**

**Contributions.**

1. A controlled four-arm comparison (frozen, trust-only, joint, LM-only) on identical data, with transfer in both directions between a closed-book and a reading-comprehension QA dataset, 3 seeds per arm, paired bootstrap CIs, and an explicit generation-quality bar.
2. A negative result, replicated after fixing an implementation flaw: joint training does not transfer better than a frozen head, and its gains in one direction are explained by language-model fine-tuning alone.
3. Evidence that the transfer gap is general: every trained head, frozen or fine-tuned, falls far below untrained signals (raw token confidence, semantic entropy) on a new dataset.
4. A full record of the process, including a flaw found after the first analysis and the single preregistered rerun it led to, with label audits.

## 2. Related Work

**Hidden-state probes for correctness.** Classifiers on hidden states detect false statements (Azaria & Mitchell, 2023), and truth-like directions can be found with or without supervision (Burns et al., 2023; Marks & Tegmark, 2024). Orgad et al. (2025) showed that correctness signal concentrates in specific answer tokens and that error detectors built on hidden states fail to generalize across datasets. Kossen et al. (2024) trained probes to predict semantic entropy. Khanmohammadi et al. (2025) trained a classifier on how a frozen model's representations respond to perturbation. In all of these, the language model stays frozen.

**Training the model for uncertainty.** Kapoor et al. (2024) fine-tuned models with LoRA on graded examples and found uncertainty estimates that generalize well. LitCab (Liu et al., 2024) trains a single linear layer while keeping the model frozen. RAGognizer (Ridder et al., 2026) is the closest prior work: it trains a token-level hallucination-detection head jointly with LoRA adapters under `L_CE + lambda * L_BCE` for retrieval-augmented QA. It reports no frozen-backbone head on the same data, no comparison of its cross-dataset results against non-joint training, and single runs without confidence intervals; its own ablation found that training for detection alone gave slightly higher separability than the joint objective.

**Untrained uncertainty signals.** Token probabilities, calibrated post hoc (Platt, 1999; Guo et al., 2017), and semantic entropy over sampled answers (Kuhn et al., 2023; Farquhar et al., 2024) need no correctness labels. Models can also be asked to judge their own answers (P(True); Kadavath et al., 2022). I use these as reference baselines.

**Evaluation.** I rank answers by AUROC and report risk-coverage results (Geifman & El-Yaniv, 2017), because a trust score is used to decide which answers to keep.

## 3. Setup

### 3.1 Model, data, and labels

- **Model:** Qwen2.5-0.5B-Instruct (Qwen Team, 2024), with its chat prompt and greedy decoding.
- **Data:** TriviaQA, `rc.nocontext` (Joshi et al., 2017), and SQuAD v1.1 (Rajpurkar et al., 2016) with splits that keep each Wikipedia article in one split. Each has 2,500 training, 500 validation, and 1,500 test questions.
- **Answers:** the base model answers every question once. These fixed answers are used for training and for the primary evaluation, so every arm is judged on identical items.
- **Labels:** an answer is correct if, after normalization, it matches one of the dataset's accepted answers under a fixed rule (rule v3, validated before Phase 2 on 600 manually audited answers). Base-model accuracy is 14.3% on TriviaQA and 69.2% on SQuAD; both test sets have at least 150 correct and 150 incorrect answers.

### 3.2 Trust head and arms

The trust head reads the mean hidden state over the answer tokens at layer 12 of 24, the best depth in an earlier layer sweep. It is a two-layer MLP: LayerNorm, Linear(896, 256), GELU, dropout 0.1, Linear(256, 1), sigmoid. It is trained with unweighted binary cross-entropy against the label.

| Arm | Backbone | Objective |
|---|---|---|
| A | Frozen | `L_trust` (head only) |
| B | LoRA | `L_trust` |
| C (TCL-J) | LoRA | `L_LM + lambda * L_trust` |
| D | LoRA, then frozen | `L_LM`, then a head on the adapted backbone |

`L_LM` is the negative log-likelihood of the dataset's reference answer given the prompt (never the model's own answer, to avoid reinforcing errors).

- **LoRA** (Hu et al., 2022): rank 16, alpha 32, on all attention and MLP projections.
- **Optimizer:** AdamW, learning rate 2e-4 for LoRA and 1e-3 for the head, effective batch size 16, 3 epochs, fp16. The checkpoint kept is the epoch with the lowest validation trust loss.
- **Lambda for C:** chosen from {0.1, 0.5, 1.0} by validation AUROC with seed 0, then run with seeds 1 and 2.
- **Seeds:** every arm has 3 seeds per training dataset.

### 3.3 Evaluation and decision rules

- **Primary metric:** AUROC of each arm's trust scores on the **other** dataset's 1,500 fixed base-model answers (transfer: TriviaQA -> SQuAD and SQuAD -> TriviaQA).
- **Statistics:** the difference between two arms, `d(X, Y)`, is averaged over seeds. Its 95% CI comes from 1,000 paired bootstrap resamples of the test items.
- **Generation quality:** each fine-tuned model regenerates its test answers, which are labeled with the same rule, and its perplexity is measured on 200 WikiText-2 test paragraphs (Merity et al., 2017). **The bar:** accuracy may drop at most 1 point on each test set, and perplexity may rise at most 3%.
- **Decision rules,** fixed before training:
  - **H3-T passes** if `d(C, A)` has a CI lower bound above zero in **both** directions and C meets the generation bar.
  - **H3-J passes** under the same condition for `d(C, B)`.
  - If `d(C, D)` includes zero in both directions, any gain is attributed to LM fine-tuning.
- **Reference baselines** on the same answers and labels:
  - raw token confidence (the geometric mean of the answer tokens' probabilities);
  - P(True);
  - semantic entropy over 10 samples;
  - a logistic probe on frozen features.

### 3.4 The rerun (Phase 2b)

After the first analysis I found that the LM target lacked the chat template's end-of-turn token (`<|im_end|>`), so the LM loss taught the model to produce the answer but never to stop. Before any further training, I preregistered one rerun:

- **The only change:** the LM target gains the end-of-turn token.
- **Rerun:** arms C and D, with lambda selected again (16 trainings). A and B have no LM loss, so they were reused.
- **Rules:** the same decision rules, plus a commitment, written in advance, to drop the claim if H3-T failed again, with no further reruns.

## 4. Results

### 4.1 The rerun fixed answer length

Median words per regenerated test answer, pooled over seeds (base model: 2).

| Trained on | Arm | First run | Rerun |
|---|---|---:|---:|
| TriviaQA | C | 10.5 / 5 | 1 / 1 |
| TriviaQA | D | 13 / 6 | 2 / 1 |
| SQuAD | C | 13 / 17 | 2 / 2 |
| SQuAD | D | 9 / 19 | 2 / 2 |

Each cell shows TriviaQA test / SQuAD test.

### 4.2 Transfer (primary)

Test AUROC on the other dataset's fixed answers, mean over 3 seeds. A and B are shared by both runs.

| | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---:|---:|
| A: frozen head | 0.557 | 0.645 |
| B: trust-only LoRA | 0.572 | 0.629 |
| C: joint LoRA, first run | 0.578 | 0.606 |
| C: joint LoRA, rerun | 0.591 | 0.606 |
| D: LM-only LoRA, first run | 0.588 | 0.643 |
| D: LM-only LoRA, rerun | 0.598 | 0.647 |
| *Raw token confidence* | *0.762* | *0.712* |
| *Semantic entropy (10 samples)* | *0.775* | *0.773* |
| *P(True)* | *0.622* | *0.643* |

Paired differences with 95% CIs:

| Difference | Run | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---|---|---|
| C - A | first | **+0.022** [+0.009, +0.035] | **-0.038** [-0.060, -0.017] |
| C - A | rerun | **+0.034** [+0.022, +0.047] | **-0.038** [-0.061, -0.015] |
| C - B | first | +0.006 [-0.008, +0.020] | **-0.022** [-0.035, -0.011] |
| C - B | rerun | **+0.018** [+0.007, +0.031] | **-0.022** [-0.036, -0.009] |
| C - D | first | -0.009 [-0.018, +0.000] | **-0.036** [-0.053, -0.019] |
| C - D | rerun | -0.007 [-0.015, +0.002] | **-0.040** [-0.057, -0.023] |
| D - A | first | **+0.031** [+0.018, +0.045] | -0.002 [-0.022, +0.019] |
| D - A | rerun | **+0.041** [+0.028, +0.055] | +0.002 [-0.016, +0.022] |

Bold: CI excludes zero.

**H3-T and H3-J fail in both runs.** From TriviaQA to SQuAD, joint training beats the frozen head, but LM-only fine-tuning does at least as well. From SQuAD to TriviaQA, joint training is significantly worse than the frozen head, the trust-only head, and the LM-only head, and the fix did not move it. Every trained head is at least 0.06 AUROC below raw token confidence and 0.12 below semantic entropy.

### 4.3 In-distribution

Test AUROC on the training dataset's own test split.

| | TriviaQA | SQuAD |
|---|---:|---:|
| A | 0.830 | 0.762 |
| B | 0.828 | 0.793 |
| C, first run / rerun | 0.849 / 0.837 | 0.803 / 0.802 |
| D, first run / rerun | 0.840 / 0.840 | 0.797 / 0.796 |
| *Raw token confidence* | *0.712* | *0.762* |

In-distribution, all trained heads beat raw confidence on TriviaQA, and the fine-tuned arms beat it on SQuAD. Joint training beats trust-only fine-tuning in both runs (rerun: +0.010 on each dataset, CIs above zero). It is never significantly better than LM-only fine-tuning (rerun: -0.003 and +0.006). After the fix, its edge over the frozen head on TriviaQA is no longer significant (+0.008 [-0.003, +0.019]); on SQuAD it remains (+0.040 [+0.026, +0.054]).

### 4.4 Generation quality

| Trained on | Arm | TriviaQA acc. | SQuAD acc. | Perplexity rise |
|---|---|---:|---:|---:|
| (base model) | | 14.3% | 69.2% | |
| TriviaQA | C, first run / rerun | 18.4% / 14.8% | 57.5% / 52.6% | +5.6% / +1.9% |
| TriviaQA | D, first run / rerun | 18.4% / 15.0% | 55.2% / 53.2% | +1.7% / +2.6% |
| SQuAD | C, first run / rerun | 18.2% / 14.7% | 87.8% / 80.4% | +10.6% / +9.1% |
| SQuAD | D, first run / rerun | 17.3% / 14.3% | 87.3% / 80.0% | +8.5% / +7.3% |

No fine-tuned arm meets the bar in either run. In the first run, long answers inflated accuracy under a containment-based label rule. After the fix, models trained on TriviaQA answer SQuAD questions in single words and lose 16 points there. They have adopted TriviaQA's short-answer style, and some of that loss may be label error (Section 4.5). Models trained on SQuAD raise perplexity by 7-9%.

### 4.5 Label audits

- **Before the first analysis:** I relabeled 300 fresh labels by hand. The rule was wrong on 13 (4.3%). On base-model SQuAD answers it erred on 9 of 100, all correct answers marked wrong (morphological variants). On long regenerated answers it erred the other way.
- **For the rerun:** I relabeled 100 regenerated answers, drawn before any rerun metric was read. The rule was wrong on 5, all correct short SQuAD answers marked wrong.
- **Effect:** the transfer comparisons score every arm on the same fixed answers and labels, so label noise lowers absolute SQuAD AUROCs but does not favor any arm.
- **Caveat:** all audit labels are mine, with no second reviewer.

## 5. Discussion

**Joint training is not what limits transfer.** The design separates the joint objective from its confounds, and the confounds explain everything that improved. In the one direction where C beats A, D beats A by as much or more, so the gain comes from adapting the backbone to the training data's language, not from training it for trust. In the other direction, C is worse than every alternative.

**The transfer gap is shared by every trained head.** The frozen head, the frozen logistic probe from my earlier work, and all three fine-tuned heads transfer at 0.56-0.68 AUROC, against 0.71-0.76 for raw token confidence and 0.77 for semantic entropy. A head trained on one dataset's correctness labels seems to learn that dataset's notion of a likely-correct answer. For TriviaQA (closed-book, 14% correct) that means recognizing questions the model knows. For SQuAD (reading comprehension, 69% correct) it means recognizing well-extracted spans. Neither carries over. Fine-tuning the backbone does not prevent this, and joint training, which also adapts the backbone to one dataset's answer style, may make it worse.

**The asymmetry.** The SQuAD-trained joint head is the worst transfer arm in both runs, with different trained weights each time. It is trained on 70% positive labels and long contexts and then applied to closed-book questions with 14% positives. Its training objective rewards whatever distinguishes good extractions, which TriviaQA does not have.

**What would change the conclusion.** This is one 0.5B model, two datasets, and 2,500 training questions per dataset. A larger model, training on a mixture of datasets, or a trust head that starts from raw confidence and learns only a correction could behave differently. Each of those is a new hypothesis, not a rescue of this one.

## 6. Limitations

- **Scale:** one small model (0.5B parameters) and LoRA fine-tuning only.
- **Datasets:** two short-answer QA datasets. Their base accuracies differ greatly (14% vs 69%), which makes transfer especially hard; closer pairs might transfer better.
- **Labels:** an automatic rule audited at 4-5% error, with all audits done by one person. The SQuAD audit errors run in one direction (correct answers marked wrong).
- **Generation bar:** the bar mixes accuracy, measured with the same rule, and perplexity on encyclopedic text. Neither is a full measure of generation quality.
- **Process:** the flaw in the LM target was found after the first analysis, so the rerun is a second attempt. It was preregistered with the outcome rules stated in advance, and both runs are reported in full.

## 7. Conclusion

Under a preregistered four-arm design, training a trust head jointly with a 0.5B language model does not produce trust estimates that transfer across datasets better than a frozen head, before or after fixing a flaw in the language-modeling target. Where it helps, plain language-model fine-tuning helps as much. The larger obstacle is common to every learned head tested: trained on one dataset's correctness labels, they fall well below untrained confidence signals on another. Progress on learned trust estimation needs evaluation on held-out datasets as the default, and methods that start from, and must beat, those untrained signals.

## Reproducibility

Every protocol was committed before the results it governs. The repository holds:

- the protocols and their amendments: `docs/phase2/TCL-Phase2-Protocol.md` and `TCL-Phase2b-Protocol.md`;
- the checkpoints: `TCL-Phase2-Checkpoint.md` and `TCL-Phase2b-Checkpoint.md`;
- the training and analysis code;
- every run's scores and regenerated answers;
- the label audits;
- the Kaggle notebook pinned to the code commit.

Compute: about 24 Kaggle T4 session hours for the runs that count.

## References

[1] Azaria, A., & Mitchell, T. (2023). The internal state of an LLM knows when it's lying. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, 967-976.

[2] Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2023). Discovering latent knowledge in language models without supervision. In *ICLR*.

[3] Farquhar, S., Kossen, J., Kuhn, L., & Gal, Y. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630, 625-630.

[4] Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. In *NeurIPS*.

[5] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *ICML*.

[6] Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. In *ICLR*.

[7] Joshi, M., Choi, E., Weld, D., & Zettlemoyer, L. (2017). TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *ACL*.

[8] Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.

[9] Kapoor, S., et al. (2024). Large language models must be taught to know what they don't know. In *NeurIPS*.

[10] Khanmohammadi, R., et al. (2025). Calibrating LLM confidence by probing perturbed representation stability. In *EMNLP*.

[11] Kossen, J., et al. (2024). Semantic entropy probes: Robust and cheap hallucination detection in LLMs. arXiv:2406.15927.

[12] Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In *ICLR*.

[13] Liu, X., Khalifa, M., & Wang, L. (2024). LitCab: Lightweight language model calibration over short- and long-form responses. In *ICLR*.

[14] Marks, S., & Tegmark, M. (2024). The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In *COLM*.

[15] Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2017). Pointer sentinel mixture models. In *ICLR*.

[16] Orgad, H., et al. (2025). LLMs know more than they show: On the intrinsic representation of LLM hallucinations. In *ICLR*.

[17] Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In *Advances in Large Margin Classifiers*.

[18] Qwen Team. (2024). Qwen2.5 technical report. arXiv:2412.15115.

[19] Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ questions for machine comprehension of text. In *EMNLP*.

[20] Ridder, F., Lessel, L., & Schilling, M. (2026). RAGognizer: Hallucination-aware fine-tuning via detection head integration. arXiv:2604.15945.
