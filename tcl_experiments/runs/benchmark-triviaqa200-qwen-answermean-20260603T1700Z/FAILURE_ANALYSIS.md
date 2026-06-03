# TCL-v0 Failure Analysis: High-Confidence Probe Errors

Run ID: `benchmark-triviaqa200-qwen-answermean-20260603T1700Z`

Status: manual failure analysis of high-confidence TCL-v0 errors

## Scope

This analysis inspects the incorrect test examples where the TCL-v0 probe assigned confidence at or above 0.8.

Source file:

- `high_confidence_probe_errors.csv`

Threshold:

- `tcl_v0_probe_confidence >= 0.8`

Number of cases:

- 4

## Case Review

### 1. `triviaqa-rc.nocontext-validation-55`

Question:

Which country does the airline TAAG come from?

Accepted answer:

Angola

Model answer:

Taiwan

Raw generation confidence:

0.5629

TCL-v0 probe confidence:

0.9992

Review:

This is a genuine model error. TAAG is the Angolan airline, not Taiwanese. The label appears correct.

Failure type:

High-confidence entity-country confusion.

### 2. `triviaqa-rc.nocontext-validation-188`

Question:

What kind of disaster claimed some 100,000 lives in Armenia in 1988?

Accepted answer:

Earthquake

Model answer:

Armenian spring famine.

Raw generation confidence:

0.3450

TCL-v0 probe confidence:

0.9448

Review:

This is a genuine model error. The 1988 Armenia disaster was an earthquake. The label appears correct.

Failure type:

High-confidence historical-event confusion.

### 3. `triviaqa-rc.nocontext-validation-185`

Question:

Who became Israel's head of state in 1993?

Accepted answer:

Ezer Weizman

Model answer:

Yitzhak Rabin was Israel's head of state

Raw generation confidence:

0.6784

TCL-v0 probe confidence:

0.9260

Review:

This is a genuine model error. Yitzhak Rabin was prime minister; Ezer Weizman became president/head of state in 1993. The label appears correct.

Failure type:

High-confidence role/title confusion.

### 4. `triviaqa-rc.nocontext-validation-22`

Question:

Which was the only eastern bloc country to participate in the 1984 LA Olympics?

Accepted answer:

Romania / Rumania

Model answer:

Poland

Raw generation confidence:

0.4963

TCL-v0 probe confidence:

0.8693

Review:

This is a genuine model error. Romania was the Eastern Bloc country that participated in the 1984 Los Angeles Olympics. The label appears correct.

Failure type:

High-confidence geopolitical/sports-history confusion.

## Summary

All 4 high-confidence TCL-v0 errors appear to be genuine model errors rather than obvious label mistakes.

This matters because the probe is not merely failing because of noisy labels. In these cases, the hidden-state probe assigned high correctness confidence to wrong answers.

## Pattern Observations

The failures share a few traits:

- The generated answers are fluent and plausible.
- The answers are short factual claims, often involving named entities.
- The raw generation confidence was moderate, not always extremely high.
- The probe sometimes converted moderate raw confidence into very high TCL-v0 confidence.

This suggests that the current logistic-regression probe may be learning hidden-state patterns associated with fluent confident answers, but not enough signal to reliably separate factual correctness from plausible entity confusion.

## Research Implication

The benchmark result should be described as mixed:

- TCL-v0 improved average ECE and Brier score on this small TriviaQA subset.
- TCL-v0 produced severe high-confidence errors.
- Therefore, TCL-v0 is not yet safe as a standalone confidence signal.

The important research lesson is not simply "TCL-v0 improved ECE." The stronger statement is:

TCL-v0 shows preliminary calibration promise, but high-confidence false positives reveal the need for validation-set calibration, manual label review, and failure-aware evaluation before stronger claims are made.

## Recommended Fixes Before Larger Runs

1. Add post-hoc calibration on the reserved validation split.
2. Report high-confidence false-positive rate alongside ECE and Brier score.
3. Add a metric for overconfident wrong answers, such as count of incorrect examples with confidence >= 0.8 and >= 0.9.
4. Compare logistic regression with a calibrated logistic regression or isotonic calibration layer.
5. Manually review a sample of both false positives and false negatives before writing any paper-level empirical claim.

## Claim Boundary

Allowed:

The high-confidence error analysis found 4 genuine wrong answers where TCL-v0 confidence was at or above 0.8, showing that the current probe can become dangerously overconfident despite improving average calibration metrics.

Not allowed:

- TCL-v0 is validated.
- TCL-v0 can reliably detect truth.
- The observed ECE improvement is sufficient for deployment or strong research claims.

