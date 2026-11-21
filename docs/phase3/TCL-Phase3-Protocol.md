# TCL Phase 3 Protocol: Can a Learned Head Beat Raw Confidence on a New Dataset?

Status: **locked** on November 21, 2026, before any Phase 3 code was written and before any Phase 3 metric was computed. The only Phase 3 numbers seen before locking are the correct/incorrect counts per split (Section 3), computed from labels alone to apply the size rule. Changes after this point are logged as amendments (Section 10) with the reason; the decision rule in Section 7 cannot change once any Phase 3 test metric has been computed.

This protocol tests **H4-T** from my program plan (`TCL-Research-Program.md`, Sections 4 and 7). The H4 trust dimensions follow in a later protocol.

## 1. Question

**H4-T:** does a correctness head trained on a **mixture** of datasets, with calibrated raw confidence among its inputs, beat calibrated raw confidence on a dataset **held out** from training?

Every head I have trained on one dataset's labels transferred below raw token confidence (Phase 1: frozen probes; Phases 2 and 2b: frozen and fine-tuned heads). Two plausible causes can each be addressed directly:

1. **Too little variety.** A head trained on one dataset may learn that dataset's quirks (its accuracy level, its answer style). Training on several datasets at once targets this.
2. **Starting from nothing.** A head that does not see raw confidence must rediscover it from hidden states and can fall below it. Giving the head raw confidence as an input targets this: it only has to learn a correction.

## 2. Two Stages

| Stage | Data | Role | Compute |
|---|---|---|---|
| **3a: design selection** | Phase 1 records (hidden states already saved): **training and validation splits only** | Chooses the head design (Section 5.3). **No test split is read in 3a.** 3a cannot pass or fail H4-T. | Laptop CPU |
| **3b: confirmatory test** | Phase 1 **test** splits plus the **new PopQA test set** | Decides H4-T with the design fixed by 3a | Kaggle GPU for PopQA inference only; the rest on CPU |

3a may settle only the choices listed in Section 5.3. 3b may change nothing else.

## 3. Data

**Models:** Qwen2.5-0.5B-Instruct and Phi-3.5-mini-instruct, with the Phase 1 prompts, decoding, and answer spans (`run_inference_v2.py`). Gemma 2B is not used: its token confidence is near chance (0.53-0.62 AUROC) and it refuses 17-30% of questions, so its comparisons against raw confidence cannot be read cleanly.

**Training pool:** TriviaQA, SQuAD, and NQ-Open, with the Phase 1 question sets and splits (`tcl_experiments/data/phase1_kaggle/`) and the Phase 1 records and hidden states. All labels are recomputed with **rule v3** (`answer_match_v3`), the Phase 2 rule.

**New held-out dataset: PopQA** (Mallen et al., 2023), closed-book questions about entities of varying popularity, never used in any earlier phase. It is used **only as a test set**; no head is trained or tuned on it.
- **Sample:** 1,500 questions drawn at random with seed 2030 from the PopQA test set, before any model answers them. Accepted answers are PopQA's `possible_answers`.
- **Size extension:** if, for a model, the smaller class (correct or incorrect) has fewer than 150 answers, I add further random batches of 1,500 (same seed sequence) until it reaches 150, up to 4,500 questions in all. This is decided from labels alone, before any metric. If 4,500 is not enough, that model's PopQA pair is dropped and recorded as dropped.
- **Inference:** `run_inference_v2.py` on Kaggle, with the same settings as Phase 1 (answer-span confidence, hidden states at 25/50/100% depth, P(True), semantic entropy with 10 samples).

**Size rule and eligible pairs.** A (model, held-out dataset) pair counts only if its test split has at least 150 correct and 150 incorrect answers. From the rule v3 labels (counts only, no metric):

| Model | Held-out dataset | Test correct / incorrect | Eligible |
|---|---|---:|---|
| Qwen | TriviaQA | 216 / 1,284 | Yes |
| Qwen | SQuAD | 1,040 / 460 | Yes |
| Qwen | NQ-Open | 108 / 1,392 | No (training data only) |
| Phi | TriviaQA | 779 / 721 | Yes |
| Phi | SQuAD | 1,405 / 95 | No (training data only) |
| Phi | NQ-Open | 322 / 1,178 | Yes |
| Qwen | PopQA | to be counted after inference | If it meets the rule |
| Phi | PopQA | to be counted after inference | If it meets the rule |

So 3b has **up to 6 eligible pairs**: 4 leave-one-out pairs on the training pool and up to 2 PopQA pairs.

## 4. Training Sets

- **Leave-one-dataset-out pairs:** for held-out dataset X, the head is trained on the **training splits of the other two datasets**, for the same model.
- **PopQA pairs:** the head is trained on the training splits of **all three** pool datasets.
- **Mixture weighting** is one of the 3a choices (Section 5.3).

## 5. Heads

### 5.1 Inputs

- **Hidden-state features:** mean hidden state over the answer span at **50% depth** (the Phase 1 layer sweep's best setting, used in Phase 2), standardized with statistics from the training mixture.
- **Raw confidence:** the answer-span geometric-mean token probability, passed as its logit.

### 5.2 Arms

| Arm | Inputs | Trained on |
|---|---|---|
| **R** | Raw confidence only, Platt-mapped on the training mixture's validation splits | Mixture (validation only) |
| **S** | Hidden states, one training dataset at a time (the Phase 1/2 setup); reported as the mean over the available single datasets | One dataset |
| **M** | Hidden states | Mixture |
| **M+R** | Hidden states and raw confidence | Mixture |
| *SE* | *Semantic entropy, 10 samples (reference, ungated)* | *None* |

### 5.3 Choices 3a settles (and nothing else)

| Choice | Options |
|---|---|
| Head type | Logistic regression, or the Phase 2 MLP (LayerNorm, 256 hidden units, GELU, dropout 0.1) |
| Regularization | L2 strength (logistic) or weight decay (MLP) from {1e-4, 1e-3, 1e-2, 1e-1, 1} |
| Mixture weighting | Equal weight per dataset, or proportional to dataset size |

**Selection rule (fixed now):** for each setting, run every leave-one-dataset-out rotation for Qwen and Phi (6 rotations), train on the other datasets' **training** splits, and score AUROC on the held-out dataset's **validation** split. Pick the setting with the best **mean validation AUROC of M+R across the 6 rotations**; ties go to the simpler setting (logistic before MLP, stronger regularization, equal weighting). The same setting is used for M and S. One setting is chosen for all pairs; nothing is tuned per pair or on any test split.

## 6. Evaluation (3b)

- **Primary metric:** AUROC on the held-out dataset's **test** answers.
- **Also reported:** AURC; accuracy at 50% and 80% coverage; ECE (15 equal-mass bins) and Brier score, with each arm's scores Platt-mapped on the training mixture's validation splits.
- **Refusals:** counted and reported; every result is also reported with refusals excluded.
- **Seeds:** each trained head is fitted with 5 seeds (0-4; data order and initialization); its scores are averaged over seeds per pair.
- **Statistics:** for a difference between two arms on a pair, 1,000 paired bootstrap resamples of the test items (seed 2026); each resample computes the seed-averaged AUROC difference. The 95% CI is the 2.5th-97.5th percentile. Per-seed values are also reported.
- **Label audit:** before any 3b metric is read, I review a random sample of 100 PopQA answers per model (seed 2031) and report the rule v3 error rate.

## 7. Decision Rule

Let `d(X, Y)` be the seed-averaged AUROC difference X minus Y on a pair's test set, with its 95% CI.

- **H4-T passes** if `d(M+R, R)` has a CI lower bound above zero on **a majority of the eligible pairs** (with 6 eligible pairs: at least 4; with 5: at least 3), **including at least one PopQA pair**.
- **Also reported, not gated:**
  - `d(M, S)`: does mixture training alone help transfer?
  - `d(M+R, M)`: does giving the head raw confidence help?
  - `d(M+R, SE)`: how the best head compares with semantic entropy.

**What follows (program plan, Section 8):**
- **H4-T passes:** I continue to H4 with the M+R head.
- **H4-T fails:** no learned correctness head beats calibrated raw confidence out of distribution at this scale. TCL's correctness dimension becomes calibrated raw confidence (and semantic entropy where sampling is affordable), and the H4 dimensions are judged on whether they add to it. I report this as a main result.

## 8. Order of Work

1. Write `run_phase3_heads.py` and the analysis script; test them on synthetic data. Commit.
2. Run 3a on training and validation splits only. Commit the selected setting and the 3a log.
3. Pin the Kaggle notebook to that commit and run PopQA inference for both models. Count labels, apply the size rule, and draw the audit sample.
4. Do the label audit, then commit everything.
5. Run the 3b analysis **once** and write the checkpoint.

## 9. Code and Outputs

- `tcl_experiments/scripts/prepare_phase3_popqa.py`: draws the PopQA sample (seed 2030).
- `tcl_experiments/scripts/run_phase3_heads.py`: trains and scores the heads from saved features (3a and 3b modes).
- `tcl_experiments/scripts/run_phase3_analysis.py`: applies Section 7 once.
- Outputs: `tcl_experiments/runs/phase3a/`, `tcl_experiments/runs/phase3_popqa/<model>/`, `tcl_experiments/runs/phase3b_analysis/`.
- Compute estimate: 3a and 3b heads on CPU; PopQA inference about 2-4 Kaggle T4 hours per model.

## 10. Amendments

None yet.

## References

- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *ACL*.
