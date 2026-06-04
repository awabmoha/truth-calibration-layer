# Extended Manual Review

Run ID: `benchmark-triviaqa500-qwen-answermean-20260604T0932Z`

Scope:

- 25 top raw-vs-probe confidence disagreement cases.
- All 26 automatic positive labels in the 100-example held-out test split.

## Review Result

- Held-out test examples: 100
- Automatic positive labels: 26
- Reviewed positive labels: 23
- Label changes: 3
- Top confidence-disagreement label changes: 0

The three label changes were automatic false positives caused by accepted-answer substring matching.

## Changed Labels

| ID | Auto Label | Reviewed Label | Reason |
|---|---:|---:|---|
| `triviaqa-rc.nocontext-validation-49` | 1 | 0 | The model restated "European Recovery Program" but did not give the common name "Marshall Plan." |
| `triviaqa-rc.nocontext-validation-98` | 1 | 0 | The model answered "George VI"; the intended answer is The Madness of King George / George III. |
| `triviaqa-rc.nocontext-validation-110` | 1 | 0 | The question asks for a river; the model answered Boulder, Colorado as a location, not the Colorado River. |

## Reviewed-Label Test Metrics

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0 | 0 |
| TCL-v0 probe confidence | 0.1568 | 0.1622 | 0.4511 | 0.8300 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2339 | 0.1910 | 0.4098 | 0.7100 | 2 | 0 |
| Conservative TCL-v0 | 0.1021 | 0.1413 | 0.2427 | 0.8300 | 0 | 0 |

## Interpretation

The reviewed-label metrics preserve the same conclusion as the automatic-label metrics:

- The plain hidden-state probe contains useful signal but still makes dangerous high-confidence errors.
- Conservative TCL-v0 remains the strongest current diagnostic score.
- Conservative TCL-v0 improves ECE, Brier score, MCE, and threshold accuracy over raw generation confidence.
- Conservative TCL-v0 still has zero wrong held-out test examples at confidence >= 0.8 or >= 0.9.

## Artifacts

- `positive_label_manual_review.csv`
- `top_confidence_disagreements_reviewed.csv`
- `reviewed_test_predictions.csv`
- `reviewed_test_metrics.json`

## Claim Boundary

These are reviewed-label TCL-v0 diagnostics on one model and one benchmark subset. They do not validate full TCL, solve hallucination, or prove that hidden states generally predict truth.
