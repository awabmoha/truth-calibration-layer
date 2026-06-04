# TCL-v0 Cross-Model Comparison

Status: early diagnostics, not full TCL validation

This comparison uses the same 500-example TriviaQA `rc.nocontext` validation subset and the same split file for two small CPU-runnable models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Both runs use:

- hidden-state method: `answer_mean`
- probe: logistic regression on frozen hidden states
- conservative confidence: `min(raw_generation_confidence, tcl_v0_probe_confidence)`
- metrics on the 100-example held-out test split

## Test Metric Summary

### Qwen 500, Reviewed Labels

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0 | 0 |
| TCL-v0 probe confidence | 0.1568 | 0.1622 | 0.4511 | 0.8300 | 5 | 5 |
| Validation-calibrated TCL-v0 | 0.2339 | 0.1910 | 0.4098 | 0.7100 | 2 | 0 |
| Conservative TCL-v0 | 0.1021 | 0.1413 | 0.2427 | 0.8300 | 0 | 0 |

### SmolLM2 500, Automatic Labels After Targeted Review

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4419 | 0.2924 | 0.9212 | 0.4900 | 0.5951 | 6 | 3 |
| TCL-v0 probe confidence | 0.0714 | 0.0676 | 0.8545 | 0.9200 | 0.9416 | 4 | 2 |
| Validation-calibrated TCL-v0 | 0.1763 | 0.1068 | 0.7577 | 0.8600 | 0.9416 | 6 | 4 |
| Conservative TCL-v0 | 0.0493 | 0.0478 | 0.3404 | 0.9400 | 0.9457 | 0 | 0 |

## Interpretation

The same qualitative pattern appears in both models:

- Frozen hidden states contain useful correctness/calibration signal.
- The plain probe can still produce high-confidence wrong answers.
- Conservative TCL-v0 is safer than the plain probe.
- Conservative TCL-v0 improves calibration metrics over raw generation confidence.
- Conservative TCL-v0 avoids high-confidence wrong answers in both held-out test splits reviewed so far.

## Claim Boundary

Allowed:

- TCL-v0 has now been tested on two small CPU-runnable LLMs using the same TriviaQA subset.
- Conservative TCL-v0 is the strongest current diagnostic variant across these runs.
- The results support continuing TCL-v0 experiments.

Not allowed:

- TCL is validated.
- TCL solves hallucination.
- TCL-v0 generalizes broadly.
- The full TCL trust vector has been implemented.

## Next Step

The next research step should either:

- test a second benchmark source, or
- improve the correctness-labeling rule before scaling to more data.

The labeling-rule improvement is attractive because the Qwen review found 3 automatic false positives from accepted-answer substring matching.
