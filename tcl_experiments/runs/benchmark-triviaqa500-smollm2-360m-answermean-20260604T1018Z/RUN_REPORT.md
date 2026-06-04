# TCL-v0 TriviaQA 500-Example Second-Model Diagnostic Report

Run ID: `benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z`

Status: second-model TCL-v0 diagnostic, not full TCL validation

## Purpose

This run tests whether the conservative TCL-v0 pattern observed with Qwen also appears on a second small language model. The dataset, splits, hidden-state method, and scoring setup are kept the same as the Qwen 500-example run.

This run evaluates only the confidence-only TCL-v0 probe:

```text
frozen generated-answer hidden states -> logistic-regression probe -> correctness confidence
```

## Model and Data

- Model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Dataset source: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`
- Prepared subset size: 500 examples
- Split: 325 train, 75 validation, 100 test
- Hidden-state method: `answer_mean`
- Hidden-state layer: final layer (`-1`)
- Max new tokens: 12
- Raw confidence method: geometric mean generated-token probability
- Correctness method: normalized exact match or word-boundary accepted-answer match

## Label Summary

- Total records: 500
- Correct labels: 46
- Incorrect labels: 454
- Overall model correctness: 0.092
- Train split: 325 examples, 36 correct, 289 incorrect
- Validation split: 75 examples, 2 correct, 73 incorrect
- Test split: 100 examples, 8 correct, 92 incorrect

## Test Metrics

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4419 | 0.2924 | 0.9212 | 0.4900 | 0.5951 | 6 | 3 |
| TCL-v0 probe confidence | 0.0714 | 0.0676 | 0.8545 | 0.9200 | 0.9416 | 4 | 2 |
| Validation-calibrated TCL-v0 | 0.1763 | 0.1068 | 0.7577 | 0.8600 | 0.9416 | 6 | 4 |
| Conservative TCL-v0 | 0.0493 | 0.0478 | 0.3404 | 0.9400 | 0.9457 | 0 | 0 |

## Manual Review

Reviewed:

- 12 high-risk wrong cases from raw, probe, calibrated, or conservative confidence.
- All 8 automatic positive labels in the held-out test split.

Review result:

- Label changes: 0
- Conservative TCL-v0 had 0 wrong test examples with confidence >= 0.8.
- Conservative TCL-v0 had 0 wrong test examples with confidence >= 0.9.

Review artifacts:

- `high_risk_manual_review.csv`
- `positive_label_manual_review.csv`
- `top_confidence_disagreements.csv`

## Interpretation

This second-model diagnostic strengthens the TCL-v0 research direction because the conservative confidence pattern appears again under a different small LLM.

SmolLM2 is weaker than Qwen on this benchmark split, but that makes the calibration test useful: raw confidence is badly miscalibrated, while the hidden-state probe separates many wrong answers more effectively. The plain probe still produces high-confidence errors, so the conservative rule remains important.

The strongest current diagnostic statement is:

```text
Across two small CPU-runnable LLMs on the same TriviaQA subset, conservative TCL-v0 improved calibration metrics over raw generation confidence while avoiding high-confidence wrong answers on the held-out test splits.
```

## Claim Boundary

Allowed diagnostic claim:

On a 500-example TriviaQA `rc.nocontext` validation subset using SmolLM2-360M-Instruct, conservative TCL-v0 improved ECE, Brier score, MCE, threshold accuracy, and AUC over raw generation confidence while avoiding high-confidence wrong answers on the held-out test split.

Not allowed:

- This does not validate TCL as a full framework.
- This does not prove hidden states generally predict truth.
- This does not show TCL solves hallucination.
- This does not justify deployment use.
- This does not validate the four-dimensional trust vector.

## Recommended Next Steps

1. Update the cross-model comparison summary.
2. Commit the second-model milestone to GitHub.
3. Next, test a second benchmark source or improve the automatic correctness-labeling rule to reduce false positives.
