# TCL Phase 1 Kaggle Checkpoint: H1-R, H2-R, and Baselines

Status: Phase 1 closed. October 18, 2026.

Protocol: `docs/phase1/TCL-Phase1-Kaggle-Protocol.md` (locked in commit 0778598, amendments in Section 7).
Inference: `tcl_experiments/scripts/run_inference_v2.py` at commit 0d1c332, on Kaggle T4 x2.
Analysis: `tcl_experiments/scripts/run_phase1_kaggle_analysis.py`, run once on all retained pairings.
Outputs: `tcl_experiments/runs/phase1_kaggle_analysis/`.

## Decisions

| Hypothesis | Rule | Result | Decision |
|---|---|---|---|
| H1-R: frozen probe improves correctness ranking beyond calibrated raw confidence | AUROC difference CI excludes zero in >= 4 of 6 retained pairings | 4 of 6 (Qwen TriviaQA, Phi NQ-Open, Gemma TriviaQA, Gemma SQuAD) | **Passes, at the threshold** |
| H2-R: frozen-probe signal transfers across datasets | Retention >= 0.70 in >= 4 of 6 directions | 2 of 6 (both Gemma directions) | **Fails** |

Both decisions are identical under all three seeds.

These sit next to the original local results, which stand as recorded: H1 failed (2 of 6) and H2 failed (2 of 6) on the TCL-v0 records.

## Data

Nine runs (3 models x 3 datasets), 1,500 test questions each. Three pairings were dropped by the size rule before any metric was computed:

| Pairing | Test correct / incorrect | Status |
|---|---:|---|
| Qwen TriviaQA | 223 / 1,277 | Retained |
| Qwen SQuAD | 1,008 / 492 | Retained |
| Qwen NQ-Open | 101 / 1,399 | Dropped |
| Phi TriviaQA | 776 / 724 | Retained |
| Phi SQuAD | 1,386 / 114 | Dropped |
| Phi NQ-Open | 312 / 1,188 | Retained |
| Gemma TriviaQA | 347 / 1,153 | Retained |
| Gemma SQuAD | 1,039 / 461 | Retained |
| Gemma NQ-Open | 107 / 1,393 | Dropped |

## H1-R Results (seed 0)

AUROC on 1,500 test questions per pairing. Differences are probe minus the named method, with 95% bootstrap CIs.

| Pairing | Raw | P(True) | Semantic entropy | Probe | Probe - calibrated raw | Probe - semantic entropy |
|---|---:|---:|---:|---:|---|---|
| Qwen TriviaQA | 0.712 | 0.636 | 0.765 | 0.776 | **+0.064 [0.021, 0.112]** | +0.011 [-0.023, 0.048] |
| Qwen SQuAD | 0.772 | 0.614 | 0.763 | 0.718 | **-0.054 [-0.084, -0.024]** (worse) | **-0.045 [-0.074, -0.013]** (worse) |
| Phi TriviaQA | 0.833 | 0.735 | 0.837 | 0.826 | -0.003 [-0.025, 0.020] | -0.011 [-0.033, 0.013] |
| Phi NQ-Open | 0.770 | 0.660 | 0.761 | 0.795 | **+0.034 [0.008, 0.062]** | **+0.034 [0.006, 0.062]** |
| Gemma TriviaQA | 0.527 | 0.734 | 0.560 | 0.841 | **+0.306 [0.277, 0.337]** | **+0.281 [0.246, 0.321]** |
| Gemma SQuAD | 0.620 | 0.832 | 0.503 | 0.870 | **+0.250 [0.219, 0.283]** | **+0.367 [0.329, 0.406]** |

(Calibrated raw has the same AUROC as raw, except where isotonic regression creates ties; the differences above use the calibrated version, as the protocol requires.)

Secondary metrics, probe minus calibrated raw (negative is better):

| Pairing | AURC | ECE | Brier |
|---|---|---|---|
| Qwen TriviaQA | -0.000 [-0.025, 0.024] | +0.012 [-0.009, 0.026] | +0.001 [-0.006, 0.009] |
| Qwen SQuAD | +0.034 [0.018, 0.052] | +0.000 [-0.022, 0.024] | +0.015 [0.007, 0.023] |
| Phi TriviaQA | +0.010 [-0.007, 0.028] | -0.022 [-0.046, 0.005] | -0.002 [-0.014, 0.010] |
| Phi NQ-Open | **-0.030 [-0.048, -0.006]** | -0.027 [-0.023, 0.012] | **-0.009 [-0.016, -0.002]** |
| Gemma TriviaQA | **-0.166 [-0.199, -0.140]** | +0.003 [-0.019, 0.020] | **-0.045 [-0.055, -0.035]** |
| Gemma SQuAD | **-0.094 [-0.114, -0.076]** | **-0.029 [-0.058, -0.006]** | **-0.099 [-0.111, -0.086]** |

ECE is a biased statistic under resampling, so its point estimate can fall outside the percentile CI (Phi NQ-Open). I report it as computed.

### Refusals excluded (secondary, not gated)

| Pairing | Probe - calibrated raw AUROC |
|---|---|
| Qwen TriviaQA | +0.064 [0.021, 0.112] |
| Qwen SQuAD | -0.054 [-0.084, -0.024] |
| Phi TriviaQA | -0.003 [-0.026, 0.020] |
| Phi NQ-Open | +0.034 [0.008, 0.062] |
| Gemma TriviaQA | +0.122 [0.083, 0.165] |
| Gemma SQuAD | +0.047 [0.000, 0.100] |

## H2-R Results (seed 0)

| Direction | Raw AUROC (target) | Target's own probe | Transferred probe | Retention | Transfer gain over raw [95% CI] |
|---|---:|---:|---:|---:|---|
| Qwen TriviaQA -> SQuAD | 0.772 | 0.718 | 0.602 | undefined (no in-distribution gain) | -0.170 [-0.213, -0.128] |
| Qwen SQuAD -> TriviaQA | 0.712 | 0.776 | 0.626 | < 0 | -0.086 [-0.133, -0.033] |
| Phi TriviaQA -> NQ-Open | 0.770 | 0.795 | 0.770 | 0.02 | +0.000 [-0.028, 0.032] |
| Phi NQ-Open -> TriviaQA | 0.833 | 0.826 | 0.773 | undefined (no in-distribution gain) | -0.060 [-0.084, -0.036] |
| Gemma TriviaQA -> SQuAD | 0.620 | 0.870 | 0.827 | 0.83 | +0.207 [0.168, 0.245] |
| Gemma SQuAD -> TriviaQA | 0.527 | 0.841 | 0.774 | 0.79 | +0.247 [0.208, 0.285] |

## Effect of the Raw-Confidence Fix

Answer-span raw confidence minus full-output raw confidence, AUROC:

| Pairing | Difference [95% CI] |
|---|---|
| Qwen TriviaQA | -0.030 [-0.043, -0.018] |
| Qwen SQuAD | -0.008 [-0.016, -0.001] |
| Phi TriviaQA | +0.011 [0.002, 0.019] |
| Phi NQ-Open | +0.004 [-0.013, 0.022] |
| Gemma TriviaQA | +0.008 [-0.016, 0.034] |
| Gemma SQuAD | +0.024 [0.015, 0.035] |

## Layer Sweep (Qwen, descriptive)

Probe AUROC by feature. Selection uses validation only.

| Feature | TriviaQA validation | TriviaQA test | SQuAD validation | SQuAD test |
|---|---:|---:|---:|---:|
| 25% depth, answer mean | 0.723 | 0.751 | 0.630 | 0.631 |
| 25% depth, answer last | 0.685 | 0.721 | 0.622 | 0.634 |
| **50% depth, answer mean** | **0.819** | 0.826 | **0.738** | 0.777 |
| 50% depth, answer last | 0.788 | 0.810 | 0.721 | 0.783 |
| Final layer, answer mean | 0.789 | 0.776 | 0.706 | 0.718 |
| Final layer, answer last | 0.768 | 0.773 | 0.712 | 0.760 |

Best by mean validation AUROC: **50% depth, answer mean**. Per the protocol, this becomes the trust head's input layer in Phase 2.

## What I Conclude

1. **H1-R passes, but only at the threshold, and the evidence is uneven.** With 1,500 test questions, the probe shows significant ranking gains for Qwen TriviaQA (+0.064) and Phi NQ-Open (+0.034). These small but real effects were undetectable with 200 questions. But the probe is *significantly worse* than raw confidence on Qwen SQuAD (-0.054), and no better on Phi TriviaQA.

2. **Gemma's large gains are partly refusal detection.** Gemma refuses 30% of TriviaQA and 17% of SQuAD questions, and all refusals are labeled incorrect. Excluding refusals, the Gemma TriviaQA gain falls from +0.306 to +0.122, and Gemma SQuAD from +0.250 to +0.047 (CI touching zero). With refusals excluded, only 3 of 6 pairings would show a significant gain. The primary rule counts refusals as wrong answers, which is defensible from a user's point of view, but it means the H1-R pass leans on Gemma's refusal behavior.

3. **My raw-confidence hypothesis was wrong.** I expected that confidence computed on the answer span would fix Gemma's near-chance raw AUROC. It does not: the fix changes AUROC by at most 0.03 for any model, and Gemma's raw confidence stays weak (0.527, 0.620). Gemma's token probabilities are simply poor correctness signals, while its P(True) is strong (0.734, 0.832).

4. **Frozen probes do not transfer (H2-R fails again).** Qwen's probes transfer *worse than raw confidence* in both directions, and Phi's add nothing. Only Gemma transfers, plausibly because refusals look alike across datasets. This replicates the local H2 failure on clean data and matches Orgad et al. (2025).

5. **A single-pass probe roughly matches 10-sample semantic entropy.** Across Qwen and Phi, probe minus semantic entropy ranges from -0.045 to +0.034. The probe costs one forward pass instead of 10 generations plus NLI comparisons. This is a practical point in favor of learned trust heads, independent of H1-R.

6. **Middle layers carry more signal than the final layer.** For Qwen, 50% depth beats the final layer on both datasets, consistent with prior probing work.

## Consequences Under the Program Plan

- Per the protocol (Section 6), H1-R passing while H1 failed means the original failure was at least partly caused by the TCL-v0 problems. Here, mainly too-small test sets and the SQuAD passage overlap; the raw-confidence fix turned out not to matter.
- The frozen-probe advantage is **real but small and model-dependent** for models whose raw confidence is informative.
- **Transfer is the main weakness of frozen probes.** Phase 2 must evaluate H3 on cross-dataset transfer as well as in-distribution, because a jointly trained head that transfers would address exactly what frozen probes fail at.
- **Refusals must be handled explicitly in Phase 2**, either as a separate abstention outcome or by reporting results with and without them, so a trust head cannot score well by recognizing refusals alone.
- Phase 2 uses **50% depth, answer-mean** features for the trust head.

## Label Audit

Complete (`docs/phase1/TCL-Phase1-Label-Audit.md`). The automatic label is wrong on 40 of 600 audited test rows (6.7%): 0-4% for TriviaQA, 7-14% for SQuAD and NQ-Open, almost all correct answers marked wrong. All audit labels were decided by the assistant and are not human-reviewed. The H1-R and H2-R decisions are not re-run; the audit is a stated limitation and defines the labeling fixes for Phase 2.

## Phase 1 Status

**Closed.** Original H1 and H2 failed on the TCL-v0 records. On corrected, larger data, H1-R passes at the threshold and H2-R fails.
