# TCL Phase 3 Protocol (DRAFT, not locked): Can a Learned Head Beat Raw Confidence on a New Dataset?

Status: **draft for my review. Not locked.** Written November 20, 2026, after the plan revision that introduced H4-T. Nothing in Phase 3 has been run or computed. Once I settle the open decisions in Section 9, I lock this protocol by removing "DRAFT" and committing it before any Phase 3 code touches a test split.

## 1. Question

**H4-T:** does a correctness head trained on a **mixture** of datasets, with calibrated raw confidence among its inputs, beat calibrated raw confidence on a dataset **held out** from training?

The case for H4-T: in Phases 1, 2, and 2b, every head trained on **one** dataset's labels transferred below raw token confidence. Two plausible causes can each be addressed directly:

1. **Too little variety.** A head trained on one dataset may learn that dataset's quirks (its accuracy level, its answer style). Training on several datasets at once targets this.
2. **Starting from nothing.** A head that does not see raw confidence must rediscover it from hidden states and can fall below it. Giving the head raw confidence as an input targets this: it only has to learn a correction.

H4 (the extra trust dimensions) follows in a later protocol and is judged against the H4-T result.

## 2. Two Stages

| Stage | Data | Role | Compute |
|---|---|---|---|
| **3a: pilot** | Phase 1 records: 3 models x 3 datasets (TriviaQA, SQuAD, NQ-Open), with hidden states already saved | Exploratory. These test splits were already evaluated in Phase 1, so 3a cannot pass or fail H4-T. It decides the head design for 3b. | Laptop CPU, no GPU |
| **3b: confirmatory** | The same training datasets, plus **at least one new dataset never used in any phase**, as the held-out test | Decides H4-T | Kaggle GPU, to generate answers and hidden states for the new dataset |

The design choices that 3a may settle are listed in advance (Section 4.3). 3b may not change anything else.

## 3. Data

- **Training pool:** TriviaQA, SQuAD, and NQ-Open with the Phase 1 splits (2,500 train and 500 validation per dataset; NQ-Open has fewer, see its metadata). Labels use rule v3.
- **Leave-one-dataset-out (3a):** train on two datasets and test on the third's 1,500 test answers, rotating through all three.
- **New held-out dataset(s) (3b):** proposed **PopQA** (Mallen et al., 2023). It is closed-book QA over entities of varying popularity and is already planned for the evidence-density dimension. Never used so far. A second candidate is listed in Section 9.
- **Size rule:** a held-out test set counts only with at least 150 correct and 150 incorrect answers. NQ-Open fails this for Qwen (7% correct), so for Qwen it serves only as training data.

## 4. Heads

### 4.1 Inputs

- **Hidden-state features:** the mean hidden state over the answer span at 50% depth, from the saved Phase 1 features.
- **Raw confidence:** calibrated raw confidence (span geometric mean, Platt-mapped on the training mixture's validation split), passed as a logit.

### 4.2 Arms

| Arm | Inputs | Training data |
|---|---|---|
| R | Calibrated raw confidence only (no learning beyond the Platt map) | Mixture validation |
| S | Hidden states, single training dataset (the Phase 1/2 setup) | One dataset |
| M | Hidden states, mixture | Mixture |
| M+R | Hidden states and raw confidence, mixture | Mixture |
| *SE* | *Semantic entropy, 10 samples (reference, ungated)* | *None* |

### 4.3 Choices the pilot may settle (and nothing else)

- **Head type:** logistic regression or the Phase 2 MLP.
- **Regularization strength:** chosen on the mixture's validation split.
- **Mixture weighting:** equal per dataset, or proportional to size.

The rule for choosing is fixed now: pick the setting with the best **mean held-out AUROC across the 3a rotations**, using only Qwen and Phi records.

## 5. Evaluation

- **Primary metric:** AUROC on the held-out dataset's test answers.
- **Also reported:** AURC, accuracy at 50% and 80% coverage, and ECE and Brier score after the Platt map fitted on the training mixture's validation split.
- **Refusals:** results are also reported with refusals excluded (Gemma refuses 17-30% of questions).
- **Statistics:** paired bootstrap, 1,000 resamples, seed fixed in the analysis script, 95% CIs. With LoRA no longer involved, heads are cheap, so each head is trained with 5 seeds and the differences are averaged over seeds.

## 6. Decision Rule (3b)

- **H4-T passes** if `d(M+R, R)` (seed-averaged held-out AUROC difference) has a 95% CI lower bound above zero for **a majority** of the (model, held-out dataset) pairs that meet the size rule, **including at least one pair on a new dataset**.
- **Also read:**
  - `d(M, S)`: does mixture training by itself help transfer?
  - `d(M+R, M)`: does giving the head raw confidence help?
  - Each is reported with its CI. Neither gates.

**What follows:**
- If H4-T passes, I continue to H4 with the M+R head.
- If it fails, I follow the program plan's Section 8 row "H4-T fails".

## 7. Label Audit

Before any 3b metric is read, I review a random sample of 100 labels from each new dataset, drawn with a seed fixed in the analysis script, and report the error rate.

## 8. Code and Outputs

- `tcl_experiments/scripts/run_phase3_heads.py` (to be written): trains and scores the heads from saved features. It is written and tested on synthetic data before touching any test split.
- Outputs: `tcl_experiments/runs/phase3a/` and `tcl_experiments/runs/phase3b/`.

## 9. Open Decisions (to settle before locking)

1. **New held-out dataset(s) for 3b.** I propose PopQA. A second option is a science QA set (for example SciQ), which differs more from the training pool. More new datasets give a stronger test but cost more GPU time.
2. **Models for 3b.** Qwen2.5-0.5B (continuity with Phase 2) and Phi-3.5-mini (higher accuracy, so more datasets pass the size rule). Gemma's weak token confidence and frequent refusals make its comparisons hard to read; I propose dropping it from 3b.
3. **Majority rule.** Whether "a majority of pairs" is the right bar, or whether H4-T should require every eligible pair.
4. **Whether 3a is worth running at all**, since its test splits have been seen. My view is yes: it is free, it fixes the head design before the confirmatory test, and its results are labeled exploratory.

## References

- Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., & Hajishirzi, H. (2023). When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In *ACL*.
