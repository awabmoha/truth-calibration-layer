# High-Risk Manual Review

Run ID: `benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z`

Scope:

- High-confidence wrong raw-confidence cases.
- High-confidence wrong plain-probe cases.
- High-confidence wrong calibrated-probe cases.
- All automatic positive labels in the held-out test split.

## Result

- High-risk wrong cases reviewed: 12
- Automatic positive labels reviewed: 8
- Label changes: 0
- Conservative TCL-v0 wrong >= 0.8: 0
- Conservative TCL-v0 wrong >= 0.9: 0

## Notes

Several raw-confidence high-risk cases were blank outputs with high token confidence. Several probe/calibrated high-risk cases were fluent but wrong answers. Conservative TCL-v0 avoided both types of high-confidence wrong answer on this held-out test split by limiting probe confidence to raw generation confidence.

## Interpretation

This supports the current conservative TCL-v0 hypothesis but remains a diagnostic result only. It should not be described as full TCL validation.
