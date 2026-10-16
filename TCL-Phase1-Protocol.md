# TCL Phase 1 Protocol: Closing TCL-v0 (Local Analysis)

Status: locked before any Phase 1 test-set evaluation. Written October 16, 2026.

This document fixes how I test H1 and H2 from `TCL-Research-Program.md` on the existing TCL-v0 records. I am committing it, together with the analysis script, before running the script. Nothing below may change after the test results are seen, except through the change-control rules in the program plan.

## 1. Scope

**In scope (local CPU):**
- H1: do frozen hidden states improve correctness ranking beyond calibrated raw confidence?
- H2: does the frozen-probe signal transfer across QA datasets?

**Deferred to the Kaggle part of Phase 1 (needs a GPU):**
- P(True) and semantic-entropy baselines
- The one-model layer sweep
- NQ-Open as a third dataset. The existing NQ-Open records use an older non-chat prompt and labeling rule v1, so they are not comparable with the six reviewed runs. NQ-Open will be re-run with the current chat prompt and labeling rule v2.

## 2. Data

The six reviewed 1,000-example runs:

| Model | Dataset | Run folder |
|---|---|---|
| Qwen2.5-0.5B-Instruct | SQuAD (with context) | `freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean` |
| Qwen2.5-0.5B-Instruct | TriviaQA (`rc.nocontext`) | `freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean` |
| Phi-3.5-mini-instruct | SQuAD | `freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean` |
| Phi-3.5-mini-instruct | TriviaQA | `freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean` |
| Gemma 2B-it | SQuAD | `freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean` |
| Gemma 2B-it | TriviaQA | `freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean` |

**Splits.** The `split` field already stored in each record: 650 train, 150 validation, 200 test. These were fixed when the runs were generated and I do not re-draw them.

**Features.** Final-layer answer-mean hidden states (`hidden_state_vector`), as recorded.

**Labels.**
- Train and validation: automatic labels (`correctness_label`, rule `strict_answer_segment_match_v2`). Targeted manual review only covered test rows.
- Test: reviewed labels from `reviewed_test_predictions.csv`.

**Known issue in earlier work.** The June 9 "reviewed answer-mean ablation checkpoint" evaluated its test predictions against the automatic labels, not the reviewed ones. Its test labels differ from the reviewed labels in 2-9 rows per run. The June 7 raw-only baseline checkpoint did use reviewed labels. Phase 1 uses reviewed test labels throughout.

**Known bias.** The targeted review chose rows using score-dependent criteria (high-confidence wrong answers, raw-vs-conservative disagreements). Label corrections are therefore concentrated where those scores disagree. I report this as a limitation; I cannot remove it without reviewing all test rows.

## 3. Methods

All fits are unweighted (no class balancing).

**Raw.** `raw_generation_confidence`, the geometric-mean token probability of the generated answer.

**Calibrated raw (the H1 comparator).** Three candidate maps: temperature scaling on the logit of raw confidence, Platt scaling, and isotonic regression. I choose one with 5-fold cross-validation on the validation split (lowest mean Brier score), then refit it on the full validation split. Selection never touches test data.

**Frozen probe (TCL-F, correctness only).**
1. Standardize features on the train split.
2. Fit L2 logistic regression on the train split. Choose the regularization strength C from {1e-4, 1e-3, 1e-2, 1e-1, 1} by stratified 5-fold cross-validation log-loss on the train split.
3. Fit a Platt map on the probe's logit using the validation split. That map is monotone, so it does not change AUROC.

**Reference rows (reported, not used in decisions).** `min(raw, probe)` and `min(raw, calibrated raw)`.

**Seeds.** Cross-validation fold assignment uses seed 0 for the primary result. I repeat the full pipeline with seeds 1 and 2 and report whether each decision holds under them.

## 4. Metrics

All metrics are computed on the 200 test rows of each run with reviewed labels.

- **AUROC** (primary).
- **AURC**: sort by confidence, highest first; at each coverage k/n, risk is the error rate among the top k; AURC is the mean risk over k = 1..n. Lower is better.
- **Accuracy at 50% and 80% coverage**: accuracy of the top ceil(c * n) rows by confidence.
- **ECE**: 15 equal-mass bins over sorted confidence; sum of bin weight times |accuracy - mean confidence|.
- **Brier score.**

Ties in confidence are broken by record order.

**Bootstrap.** 1,000 paired resamples of the test rows (seed 2026). Each method is scored on the same resample, and the 95% CI of every difference is the 2.5th-97.5th percentile. Resamples with only one class are skipped for AUROC, and the number skipped is reported.

## 5. Decision Rules

### H1

For each run, compute the difference `AUROC(probe) - AUROC(calibrated raw)` with its 95% bootstrap CI.

**H1 passes** if the CI lower bound is above zero in **at least 4 of 6 runs** (primary seed 0).

### H2

For each model, train the probe and its Platt map on one dataset's train/validation splits and apply them unchanged to the other dataset's test split. That gives six transfer directions (3 models x 2 directions).

For each direction:

```text
in-distribution gain = AUROC(target's own probe, target test) - AUROC(raw, target test)
transfer gain        = AUROC(source probe, target test)      - AUROC(raw, target test)
retention            = transfer gain / in-distribution gain
```

**H2 passes** if retention is **at least 0.70 in at least 4 of 6 directions**. A direction whose in-distribution gain is zero or negative counts as not passing.

This count rule was not in the program plan; the plan stated only the 70% retention threshold. I am fixing it here, before evaluation, so the threshold is applied to a defined set of directions.

### If H1 or H2 fails

Follow the program plan: proceed to Phase 2 regardless, and record that the case for TCL then rests on joint training (H1 fail) or that transfer must be designed into later phases (H2 fail).

## 6. Outputs

- Script: `tcl_experiments/scripts/run_phase1_analysis.py`
- Results: `tcl_experiments/runs/phase1_local/` (per-run metrics, bootstrap CIs, transfer table, seed-robustness table, test predictions)
- Checkpoint write-up: `TCL-Phase1-Local-Checkpoint.md`, written after the run, reporting every result including failures.
