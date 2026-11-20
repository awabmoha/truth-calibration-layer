# TCL Phase 2 Checkpoint: Joint Training Does Not Transfer

Written November 20, 2026, after the single preregistered analysis run. Protocol: [TCL-Phase2-Protocol.md](TCL-Phase2-Protocol.md) (locked October 18, 2026, with amendments in its Section 11).

## 1. Decision

| Hypothesis | Rule | Result |
|---|---|---|
| **H3-T** (primary) | `d(C, A)` CI above zero in **both** transfer directions, and C meets the generation-quality bar | **Fails.** C beats A from TriviaQA to SQuAD (+0.022), but is significantly **worse** than A from SQuAD to TriviaQA (-0.038), and C fails the generation-quality bar. |
| **H3-J** (secondary) | `d(C, B)` CI above zero in both directions, and C meets the bar | **Fails.** C ties B from TriviaQA to SQuAD and is worse than B from SQuAD to TriviaQA. |
| Tie reading | Applies when C ties B | Not applicable as a claim: C fails the generation-quality bar, so neither "generation-preserving trust training" nor "no added value" can be stated. |
| Attribution to LM fine-tuning | `d(C, D)` CI includes zero in both directions | Not met (C is worse than D from SQuAD to TriviaQA). But the one positive direction is matched by D: LM-only fine-tuning gives +0.031 over A, at least as much as C's +0.022. |

**In plain words:** training the trust head jointly with the model makes it better *on the dataset it was trained on*, but not on a new dataset, which is what H3 was about. On a new dataset, every trained head, joint or not, ranks answers far worse than the model's raw token confidence.

## 2. What Ran

- **Model:** Qwen2.5-0.5B-Instruct. **Data:** Phase 1 TriviaQA and SQuAD splits (2,500 train, 500 validation, 1,500 test each), base-model answers labeled with rule v3.
- **Arms:** A (frozen backbone, head only), B (LoRA, trust loss only), C (LoRA, `L_LM + lambda * L_trust`), D (LoRA, LM loss only, then a head on the frozen adapted backbone). Three seeds each, both training datasets: 28 trainings.
- **Lambda for C** (seed 0, validation AUROC): 0.5 for TriviaQA (0.836 / **0.839** / 0.839 for 0.1 / 0.5 / 1.0) and 1.0 for SQuAD (0.777 / 0.793 / **0.799**).
- **Training check:** every LoRA run took 153-157 optimizer steps per epoch with 0-4 skipped by fp16 loss scaling, so every model was actually trained.
- **Compute:** about 18 Kaggle T4 session hours for the runs that counted (six sessions of 2.0-4.0 hours, two trainings at a time), after two failed attempts (an out-of-memory crash on SQuAD and two sessions stopped by Kaggle). The amendments in protocol Section 11 record each change; none touched the objectives, hyperparameters, or decision rules.

## 3. Primary Result: Transfer

Test AUROC on the **other** dataset's 1,500 fixed base-model answers, mean over 3 seeds. Reference rows use the same answers and labels.

| | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---:|---:|
| A: frozen head | 0.557 | **0.645** |
| B: trust-only LoRA | 0.572 | 0.629 |
| C: joint LoRA (TCL-J) | 0.578 | 0.606 |
| D: LM-only LoRA, then head | **0.588** | 0.643 |
| *Raw token confidence (no training)* | *0.762* | *0.712* |
| *Semantic entropy (10 samples)* | *0.775* | *0.773* |
| *P(True)* | *0.622* | *0.643* |
| *Frozen logistic probe (Phase 1 method)* | *0.575* | *0.682* |

Paired differences (seed-averaged, 95% bootstrap CI, 1,000 resamples):

| Difference | TriviaQA -> SQuAD | SQuAD -> TriviaQA |
|---|---|---|
| C - A | **+0.022** [+0.009, +0.035] | **-0.038** [-0.060, -0.017] |
| C - B | +0.006 [-0.008, +0.020] | **-0.022** [-0.035, -0.011] |
| C - D | -0.009 [-0.018, +0.000] | **-0.036** [-0.053, -0.019] |
| B - A | **+0.016** [+0.002, +0.030] | -0.016 [-0.037, +0.004] |
| D - A | **+0.031** [+0.018, +0.045] | -0.002 [-0.022, +0.019] |

Bold = CI excludes zero. No base-model answer is a refusal, so the refusal-excluded results are identical.

Calibration does not transfer either: with the Platt map fitted on the source validation split, transfer ECE is 0.53-0.57 (TriviaQA -> SQuAD) and 0.31-0.40 (SQuAD -> TriviaQA), because the two datasets' accuracy differs so much (14% vs 69%).

## 4. In-Distribution (Not the Hypothesis, but Informative)

Test AUROC on the training dataset's own test split.

| | TriviaQA | SQuAD |
|---|---:|---:|
| A | 0.830 | 0.762 |
| B | 0.828 | 0.793 |
| C | **0.849** | **0.803** |
| D | 0.840 | 0.797 |
| *Raw token confidence* | *0.712* | *0.762* |

| Difference | TriviaQA | SQuAD |
|---|---|---|
| C - A | **+0.019** [+0.007, +0.032] | **+0.041** [+0.026, +0.055] |
| C - B | **+0.021** [+0.008, +0.035] | **+0.010** [+0.003, +0.018] |
| C - D | +0.009 [-0.002, +0.018] | +0.006 [-0.005, +0.017] |

In-distribution, the joint arm is the best on both datasets and beats both the frozen head and trust-only fine-tuning with CIs above zero. Its edge over LM-only fine-tuning (D) is not significant.

## 5. Generation Quality

Accuracy of each model's own regenerated answers (rule v3) and WikiText-2 perplexity, mean over 3 seeds. Base model: TriviaQA 14.3%, SQuAD 69.2%, perplexity 21.93. The bar: accuracy drops at most 1 point on each test set, perplexity rises at most 3%.

| Trained on | Arm | TriviaQA acc. | SQuAD acc. | Perplexity rise | Bar |
|---|---|---:|---:|---:|---|
| TriviaQA | B | 16.9% | 64.2% | +1.7% | fails (SQuAD -5.0) |
| TriviaQA | C | 18.4% | 57.5% | +5.6% | fails |
| TriviaQA | D | 18.4% | 55.2% | +1.7% | fails (SQuAD -14.0) |
| SQuAD | B | 15.3% | 66.1% | +7.4% | fails |
| SQuAD | C | 18.2% | 87.8% | +10.6% | fails (perplexity) |
| SQuAD | D | 17.3% | 87.3% | +8.5% | fails (perplexity) |

**These accuracy numbers need care.** Fine-tuning changed the *form* of the answers. The base model answers in a median of 2 words; after LoRA training, the chosen arms answer in a median of about 9-20 words (Section 7.1), except B trained on SQuAD, which stays at 2. Rule v3 counts an answer as correct when it contains an accepted answer, so long answers that list several candidates can gain accuracy without being better, and the large SQuAD gains (69% -> 88%) partly reflect the model copying more of the passage. The bar failures on perplexity are not affected by this.

## 6. Label Audit (Protocol Section 9)

300 labels reviewed: 100 fresh base-model test answers per dataset (seed 2028, excluding Phase 1's audited rows) and 50 regenerated answers per dataset from arm C, seed 0. Full sample with my label and reason for each change: [`label_audit_reviewed.csv`](../../tcl_experiments/runs/phase2_analysis/label_audit_reviewed.csv).

| Sample | Rule errors | Type |
|---|---:|---|
| Base answers, TriviaQA | 1 / 100 | correct marked wrong |
| Base answers, SQuAD | 9 / 100 | all correct marked wrong (e.g. "annexation of" vs "annexing", "its own weight" vs "their own weight") |
| Arm C answers, TriviaQA | 2 / 50 | wrong marked correct (a long answer that mentions the right token in the wrong role) |
| Arm C answers, SQuAD | 1 / 50 | wrong marked correct (hedges between two years) |
| **Total** | **13 / 300 (4.3%)** | 2 of the 13 are judgment calls, flagged in the file |

As with the Phase 1 audit, every label here is my own decision; none was reviewed by a second person. The audit was done after the analysis ran; the protocol placed it before. Its sample was drawn by a fixed seed from labels only, and no decision rule depends on it, but I record the order.

**What the audit changes:** rule v3 misses morphological variants on SQuAD (about 9% of base answers), which lowers every arm's absolute AUROC on SQuAD by adding label noise. Because every arm is scored on the same fixed answers and labels, the paired comparisons in Sections 3 and 4 remain fair. On long regenerated answers, the rule errs the other way, which supports the caution in Section 5.

## 7. What I Found Wrong With the Setup

### 7.1 The LM loss never taught the model where to stop

The LM loss (protocol Section 4) is the likelihood of the reference answer given the prompt. In my implementation, the target is the reference tokens **without the end-of-turn token** that follows an answer in the chat format. The model is trained to produce the answer but never to stop after it. This plausibly explains why arms C and D answer in 9-20 words instead of 2, and part of their perplexity rise and generation-bar failure.

It does not explain everything: arm B, which has no LM loss, also lengthened its answers when trained on TriviaQA (median 2 -> 9-10 words). Fine-tuning the backbone for trust alone also changes how it generates.

This flaw affects the backbone that C and D's trust heads read from, so it may have affected the transfer comparison too. I cannot tell from these runs by how much.

### 7.2 Training the head on one dataset's correctness does not generalize

The deeper result is independent of 7.1: **arm A, the frozen head, already fails to transfer** (0.557 and 0.645, against raw confidence at 0.762 and 0.712), exactly as Phase 1's frozen probe did. No form of training I tested closes that gap. What a trained head learns about correctness on TriviaQA (a closed-book quiz where 14% of answers are right) does not carry over to SQuAD (reading comprehension where 69% are right), and vice versa. Raw token confidence and semantic entropy, which are not trained on either dataset, transfer far better.

### 7.3 The decision table did not anticipate a split

My program plan's decision table (Section 8) has rows for "H3-T fails, A ≈ C" and "H3-T fails, C < A". The result is neither: C beats A in one direction and loses in the other. The closer row is "C < A", since C is significantly worse in one direction and fails the generation bar, which says: *investigate lambda and label staleness once; if it still fails, drop the joint claim and publish the negative result.*

## 8. What This Means for TCL

- **The core claim of TCL-J is not supported at this scale.** Joint training did not produce trust estimates that transfer better than a frozen head, and it did not preserve generation quality as implemented.
- **There is a real in-distribution effect.** Joint training gives the best in-distribution trust ranking on both datasets (+0.02 to +0.04 AUROC over the frozen head), but D (LM-only fine-tuning) gets most of it, so the effect is not clearly due to training *for trust*.
- **Transfer is the problem for every trained method, frozen or not.** This matches Phase 1, and it is now the central obstacle for any version of TCL that learns trust from labeled correctness.

## 9. Next Step (My Decision, Not Yet Made)

> **Update (November 20, 2026):** I chose the investigation. Phase 2b ([protocol](TCL-Phase2b-Protocol.md), [checkpoint](TCL-Phase2b-Checkpoint.md)) fixed the end-of-turn token and failed H3-T again, so the joint-transfer claim is dropped. The text below is kept as written before that decision.

Following the plan's "C < A" row, I can investigate once before dropping the joint claim. The one change I would test is the flaw in 7.1: add the end-of-turn token to the LM target, so the model learns to stop. That would be a new, separately preregistered experiment (Phase 2b), with the same arms, data, and decision rules, not a re-analysis of these runs. These Phase 2 results stand as reported.

The alternative is to accept the negative result now, report it as a main finding, and move TCL toward what does transfer (raw confidence, semantic entropy, and multi-dimensional signals that do not depend on one dataset's labels), as the plan's Phase 3 allows.

## 10. Files

| Path | Contents |
|---|---|
| `tcl_experiments/runs/phase2/<train dataset>/<arm>/seed<k>/` | Per-run test scores on fixed answers, regenerated answers, metrics (including optimizer step counts), training log |
| `tcl_experiments/runs/phase2_analysis/decisions.json` | H3-T, H3-J, tie reading, attribution, generation bar |
| `tcl_experiments/runs/phase2_analysis/details.json` | Every difference with CI and per-seed values, secondary metrics, generation quality, reference baselines |
| `tcl_experiments/runs/phase2_analysis/arm_metrics.csv` | Per-arm, per-seed metrics |
| `tcl_experiments/runs/phase2_analysis/label_audit_reviewed.csv` | The 300-row label audit with my labels and reasons |
| `tcl_experiments/runs/phase2_analysis/analysis_run.log` | Output of the single analysis run |
