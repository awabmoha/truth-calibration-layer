# TCL-v0 Method Comparison: Qwen 100-Question Diagnostic

Status: diagnostic method comparison, not benchmark validation

This document compares two hidden-state extraction methods for TCL-v0 on the same local 100-question factual QA diagnostic set.

## Shared Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset: `diagnostic_questions_100`
- Questions: 100
- Correct labels: 91
- Incorrect labels: 9
- Train/test split: 65 / 35
- Hidden-state layer: final layer (`-1`)
- Probe: logistic regression
- Raw confidence: geometric mean generated-token probability
- Correctness labeling: normalized exact match or accepted-answer substring match

## Compared Runs

Answer mean:

- Run ID: `diagnostic-qwen100-answermean-20260603T1632Z`
- Records: `runs/diagnostic-qwen100-answermean-20260603T1632Z/records_answer_mean.jsonl`
- Analysis: `runs/diagnostic-qwen100-answermean-20260603T1632Z/analysis`

Answer last:

- Run ID: `diagnostic-qwen100-answerlast-20260603T1645Z`
- Records: `runs/diagnostic-qwen100-answerlast-20260603T1645Z/records_answer_last.jsonl`
- Analysis: `runs/diagnostic-qwen100-answerlast-20260603T1645Z/analysis`

## Metric Summary

| Method | Raw ECE | Probe ECE | Raw Brier | Probe Brier | Raw MCE | Probe MCE |
|---|---:|---:|---:|---:|---:|---:|
| `answer_mean` | 0.3015 | 0.0816 | 0.1833 | 0.0861 | 0.6030 | 0.1979 |
| `answer_last` | 0.3015 | 0.1363 | 0.1833 | 0.1058 | 0.6030 | 0.5655 |

Additional AUC notes:

- `answer_mean` probe AUC: 0.5104
- `answer_last` probe AUC: 0.7188

## Interpretation

On this diagnostic split, `answer_mean` produced better calibration metrics:

- Lower ECE.
- Lower Brier score.
- Lower MCE.

However, `answer_last` produced a higher AUC, suggesting it may rank correct versus incorrect answers better in this small test even though its probability calibration is worse.

For TCL-v0's calibration goal, `answer_mean` is the stronger current default. For later selective prediction or ranking experiments, `answer_last` should not be discarded.

## Claim Boundary

Allowed diagnostic claim:

On a local 100-question diagnostic set with Qwen/Qwen2.5-0.5B-Instruct, mean-pooled generated-answer hidden states produced better TCL-v0 calibration metrics than final generated-token hidden states.

Not allowed:

- This does not validate TCL as a full framework.
- This does not prove hidden states generally predict truth.
- This does not show the method will transfer to TriviaQA, Natural Questions, other models, or larger datasets.
- This does not solve hallucination.

## Recommended Next Step

Move from the local diagnostic CSV to a standard benchmark subset while keeping `answer_mean` as the default hidden-state method. The next benchmark run should include:

- A fixed random seed.
- A dataset split file.
- Manual review of a sample of correctness labels.
- At least several hundred examples if the machine can tolerate the runtime.

