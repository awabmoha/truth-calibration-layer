# TCL-v0 Validation Calibration Report

Run ID: `benchmark-triviaqa200-qwen-answermean-20260603T1700Z`

Status: validation-split calibration diagnostic

## Purpose

This step tested whether post-hoc calibration on the reserved validation split could reduce the high-confidence TCL-v0 failures observed in the first TriviaQA benchmark diagnostic.

Calibration method:

- Train the base TCL-v0 logistic-regression probe on the recorded `train` split.
- Compute probe decision scores on the recorded `validation` split.
- Fit a 1D logistic calibrator on validation scores.
- Evaluate raw confidence, uncalibrated TCL-v0 confidence, and calibrated TCL-v0 confidence on the recorded `test` split.

## Files

- Calibrated analysis directory: `analysis_calibrated`
- Calibrated method summary: `analysis_calibrated/method_summary.csv`
- Calibrated reliability bins: `analysis_calibrated/answer_mean/tcl_v0_calibrated_reliability_bins.csv`
- Calibrated test predictions: `analysis_calibrated/answer_mean/test_predictions.csv`
- Calibrated high-confidence errors: `calibrated_high_confidence_errors.csv`

## Split Counts

- Train: 130
- Validation: 30
- Test: 40

## Metric Comparison

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.3292 | 0.1923 | 0.4544 | 0.6750 | 0.9048 | 0 | 0 |
| TCL-v0 probe confidence | 0.2086 | 0.1850 | 0.9567 | 0.7750 | 0.7576 | 4 | 3 |
| Calibrated TCL-v0 confidence | 0.2742 | 0.2270 | 0.9742 | 0.6750 | 0.7576 | 4 | 1 |

## Interpretation

Calibration did not solve the problem.

It helped one narrow issue:

- Incorrect examples with TCL-v0 confidence >= 0.9 dropped from 3 to 1.

But it made several important metrics worse:

- ECE worsened from 0.2086 to 0.2742.
- Brier score worsened from 0.1850 to 0.2270.
- Accuracy at 0.5 dropped from 0.7750 to 0.6750.
- Wrong examples with confidence >= 0.8 stayed at 4.
- MCE remained very high.

This means a simple validation-score logistic calibrator is not enough for TCL-v0 on this benchmark split.

## Research Meaning

This is a useful negative result.

The current probe can improve average calibration before calibration, but its high-confidence wrong answers are not fixed by this simple post-hoc calibration step. The likely causes are:

- The validation split is small: only 30 examples.
- The model is weak on TriviaQA, with only 36 correct labels out of 200.
- The probe may be learning fluent-answer hidden-state patterns more than correctness.
- The current hidden-state representation may need richer features than `answer_mean` alone.

## Claim Boundary

Allowed:

Validation-score calibration reduced the count of extreme wrong predictions above 0.9, but worsened ECE, Brier score, and threshold accuracy on the 200-example TriviaQA diagnostic.

Not allowed:

- Calibration validates TCL-v0.
- Post-hoc calibration solved the high-confidence false-positive problem.
- The current probe is safe as a standalone reliability signal.

## Next Step

Before running a larger benchmark, add explicit failure metrics to the standard analysis output and test a more conservative confidence strategy:

```text
TCL-v0 conservative confidence = min(raw_generation_confidence, probe_confidence)
```

Reason:

The raw confidence had no wrong examples above 0.8 on this test split, while the probe did. A conservative combination may preserve some hidden-state calibration benefit while preventing the probe from increasing confidence above raw generation confidence on fluent wrong answers.

