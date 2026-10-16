# TCL Phase 1 Local Checkpoint: H1 and H2

Status: local part of Phase 1 complete. October 16, 2026.

Protocol: `TCL-Phase1-Protocol.md`, committed (ad47163) before this analysis ran.
Script: `tcl_experiments/scripts/run_phase1_analysis.py`.
Outputs: `tcl_experiments/runs/phase1_local/`.

## Decisions

| Hypothesis | Rule | Result | Decision |
|---|---|---|---|
| H1: frozen probe improves correctness ranking beyond calibrated raw confidence | AUROC difference CI excludes zero in >= 4 of 6 runs | 2 of 6 (Gemma SQuAD, Gemma TriviaQA) | **Fails** |
| H2: frozen-probe signal transfers across datasets | Retention >= 0.70 in >= 4 of 6 directions | 2 of 6 (both Gemma directions) | **Fails** |

Both decisions are identical under all three cross-validation seeds (0, 1, 2).

## H1 Results (seed 0, 200 test rows per run, reviewed labels)

| Run | Test negatives / positives | Raw AUROC | Calibrated raw AUROC | Probe AUROC | Difference [95% CI] |
|---|---|---:|---:|---:|---|
| Qwen SQuAD | 52 / 148 | 0.674 | 0.674 | 0.742 | +0.068 [-0.046, 0.178] |
| Qwen TriviaQA | 161 / 39 | 0.747 | 0.747 | 0.756 | +0.009 [-0.105, 0.115] |
| Phi SQuAD | 7 / 193 | 0.706 | 0.606 | 0.544 | -0.062 [-0.317, 0.264] |
| Phi TriviaQA | 88 / 112 | 0.838 | 0.838 | 0.849 | +0.012 [-0.049, 0.070] |
| Gemma SQuAD | 68 / 132 | 0.534 | 0.534 | 0.930 | **+0.395 [0.306, 0.487]** |
| Gemma TriviaQA | 148 / 52 | 0.712 | 0.712 | 0.830 | **+0.118 [0.029, 0.211]** |

Secondary metrics, probe minus calibrated raw (negative is better for AURC, ECE, Brier):

| Run | AURC | ECE | Brier |
|---|---|---|---|
| Qwen SQuAD | -0.011 [-0.069, 0.050] | -0.009 [-0.069, 0.048] | -0.017 [-0.041, 0.008] |
| Qwen TriviaQA | +0.010 [-0.054, 0.069] | -0.024 [-0.054, 0.038] | +0.008 [-0.007, 0.025] |
| Phi SQuAD | +0.009 [-0.026, 0.022] | +0.021 [-0.008, 0.043] | +0.003 [-0.002, 0.008] |
| Phi TriviaQA | -0.000 [-0.037, 0.039] | -0.017 [-0.073, 0.044] | -0.011 [-0.036, 0.017] |
| Gemma SQuAD | **-0.187 [-0.255, -0.127]** | -0.017 [-0.094, 0.025] | **-0.126 [-0.155, -0.098]** |
| Gemma TriviaQA | -0.060 [-0.139, 0.011] | -0.028 [-0.090, 0.006] | **-0.028 [-0.047, -0.009]** |

## H2 Results (seed 0)

| Direction | Raw AUROC (target) | Target's own probe | Transferred probe | Retention | Transfer gain over raw [95% CI] |
|---|---:|---:|---:|---:|---|
| Qwen SQuAD -> TriviaQA | 0.747 | 0.756 | 0.598 | < 0 | -0.149 [-0.257, -0.034] |
| Qwen TriviaQA -> SQuAD | 0.674 | 0.742 | 0.512 | < 0 | -0.162 [-0.308, 0.001] |
| Phi SQuAD -> TriviaQA | 0.838 | 0.849 | 0.776 | < 0 | -0.062 [-0.128, 0.002] |
| Phi TriviaQA -> SQuAD | 0.706 | 0.544 | 0.647 | undefined (no in-distribution gain) | -0.059 [-0.446, 0.314] |
| Gemma SQuAD -> TriviaQA | 0.712 | 0.830 | 0.838 | 1.07 | +0.126 [0.035, 0.223] |
| Gemma TriviaQA -> SQuAD | 0.534 | 0.930 | 0.854 | 0.81 | +0.320 [0.221, 0.421] |

Transferred probes are badly calibrated on the target dataset (ECE 0.26-0.58). Base rates differ sharply between datasets (SQuAD 66-97% correct, TriviaQA 20-56%), so a Platt map fitted on the source does not carry over.

## What I Conclude

1. **The frozen hidden-state advantage is model-specific, not general.** It is large and significant for Gemma 2B-it, on both datasets and in both transfer directions. For Qwen and Phi it is not distinguishable from zero, and their probes transfer *worse than raw confidence*.

2. **Gemma's advantage coincides with weak raw confidence.** Gemma's raw AUROC on SQuAD is 0.534, close to chance. Where raw confidence is already informative (Phi TriviaQA, 0.838), the probe adds almost nothing. The hidden states may be compensating for a poor raw signal, not adding information beyond a good one. I also need to rule out a defect in how raw confidence was computed for Gemma, since its chat template required a patched notebook.

3. **The tests are underpowered.** With 200 test rows, AUROC differences have CIs about ±0.1 wide, and Phi SQuAD has only 7 negative test rows. Point estimates favor the probe in 5 of 6 runs, but only the two largest effects reach significance. A null result here means "not shown," not "absent." Future phases need much larger test sets.

4. **This matches prior work.** Orgad et al. (2025) report that hidden-state error detectors fail to generalize across datasets. My Qwen and Phi transfer results reproduce that.

## Consequences Under the Program Plan

The plan (Section 7, Phase 1) says: if H1 fails, proceed to Phase 2 anyway, and record that frozen probes add little, so that the case for TCL rests on joint training. I record that now.

- **Frozen probes (TCL-F) are not a reliable improvement over calibrated raw confidence** at this scale and sample size, except for Gemma.
- **The case for TCL now rests on H3** (joint training) and on H4 (additional trust dimensions).
- The H2 failure means **transfer must be designed into Phase 2 evaluation**, not assumed.

## Exploratory Observations (Not Preregistered)

These were not part of the decision rules. They are hypotheses for later phases, not findings.

- `min(raw, probe)` often had higher AUROC than either signal alone (Qwen SQuAD 0.766, Phi SQuAD 0.760). Raw confidence and hidden states may carry complementary information. A learned combination should be a preregistered baseline in Phase 2.
- The raw calibrator selected for Phi SQuAD under seed 0 was isotonic regression, whose ties lowered AUROC from 0.706 to 0.606. With 7 negative test rows and only 4 negative validation rows (fewer than the 5 cross-validation folds), every Phi SQuAD number is fragile.

## Data Issue Found

The June 9 "reviewed answer-mean ablation checkpoint" (`TCL-v0-Reviewed-AnswerMean-Ablation-Checkpoint.md`) evaluated test predictions against **automatic** labels, not reviewed labels. Its test labels differ from the reviewed ones in 2-9 rows per run. The June 7 baseline checkpoint did use reviewed labels. Phase 1 uses reviewed labels throughout. The June 9 checkpoint's numbers should not be cited without this caveat.

## Limitations

- The targeted manual review selected rows with score-dependent criteria, and covered only test rows. Train and validation labels are automatic.
- One hidden-state configuration (final layer, answer-mean). The layer sweep is in the Kaggle part of Phase 1.
- The sampling baselines (P(True), semantic entropy) are not yet run. They are in the Kaggle part of Phase 1.

## Remaining Phase 1 Work (Kaggle)

1. P(True) and semantic-entropy baselines.
2. One-model layer sweep.
3. NQ-Open re-run with the current prompt and labeling rule.
4. A check of Gemma's raw-confidence computation.
