# TCL-v0 Conservative Confidence Report

Run ID: `benchmark-triviaqa200-qwen-answermean-20260603T1700Z`

Status: conservative confidence diagnostic

## Purpose

This step tested a conservative confidence rule after validation calibration failed to remove high-confidence false positives cleanly.

Rule:

```text
tcl_v0_conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

The goal is to prevent the hidden-state probe from raising confidence above the base model's raw generation confidence when the generated answer is fluent but wrong.

## Files

- Conservative analysis directory: `analysis_conservative`
- Method summary: `analysis_conservative/method_summary.csv`
- Conservative reliability bins: `analysis_conservative/answer_mean/tcl_v0_conservative_reliability_bins.csv`
- Test predictions with all confidence variants: `analysis_conservative/answer_mean/test_predictions.csv`

## Metric Comparison

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.3292 | 0.1923 | 0.4544 | 0.6750 | 0.9048 | 0 | 0 |
| TCL-v0 probe confidence | 0.2086 | 0.1850 | 0.9567 | 0.7750 | 0.7576 | 4 | 3 |
| Validation-calibrated TCL-v0 | 0.2742 | 0.2270 | 0.9742 | 0.6750 | 0.7576 | 4 | 1 |
| Conservative TCL-v0 | 0.1347 | 0.1257 | 0.6784 | 0.8250 | 0.7922 | 0 | 0 |

## Interpretation

The conservative rule performed best on the main calibration metrics for this test split:

- Best ECE: 0.1347
- Best Brier score: 0.1257
- Best threshold accuracy: 0.8250
- Zero incorrect test examples with confidence >= 0.8
- Zero incorrect test examples with confidence >= 0.9

It did not beat raw generation confidence on AUC:

- Raw AUC: 0.9048
- Conservative AUC: 0.7922

This suggests the conservative rule is better for calibrated confidence safety on this split, while raw confidence still ranks correct vs incorrect answers better.

## Research Meaning

This is the strongest TCL-v0 benchmark diagnostic so far.

The result supports a refined TCL-v0 direction:

```text
Use hidden-state probe confidence as a calibration signal,
but constrain it with raw generation confidence to avoid overconfident false positives.
```

This is not the full TCL trust vector. It is a practical TCL-v0 safety modification.

## Claim Boundary

Allowed:

On the 200-example TriviaQA diagnostic split, the conservative TCL-v0 confidence rule improved ECE, Brier score, and threshold accuracy over raw confidence and uncalibrated probe confidence, while eliminating test errors above 0.8 confidence.

Not allowed:

- This validates TCL broadly.
- This proves the conservative rule will generalize.
- This solves hallucination.
- This removes the need for manual label review or larger benchmark testing.

## Next Step

The next step is to update the experiment plan and README to include conservative TCL-v0 as the current best benchmark diagnostic variant, then run the same conservative analysis on a larger TriviaQA subset only after the documentation is updated.

