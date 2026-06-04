# TCL-v0 SQuAD-500 Manual Review Report

Status: targeted held-out review, not full benchmark relabeling

This review audited SQuAD-500 held-out test cases that were labeled incorrect but received high confidence from at least one score:

- raw generation confidence
- TCL-v0 probe confidence
- conservative TCL-v0 confidence
- validation-calibrated TCL-v0 confidence

It also reviewed the `strict_answer_segment_match_v2` label changes.

## Artifacts

Qwen:

- candidates: `runs/benchmark-squad500-qwen-answermean-20260604T1652Z/manual_review_high_conf_wrong_v2_candidates.csv`
- completed review: `runs/benchmark-squad500-qwen-answermean-20260604T1652Z/manual_review_high_conf_wrong_v2_completed.csv`
- v2 label changes: `runs/benchmark-squad500-qwen-answermean-20260604T1652Z/label_changes_v2.csv`

SmolLM2:

- candidates: `runs/benchmark-squad500-smollm2-360m-answermean-20260604T1820Z/manual_review_high_conf_wrong_v2_candidates.csv`
- completed review: `runs/benchmark-squad500-smollm2-360m-answermean-20260604T1820Z/manual_review_high_conf_wrong_v2_completed.csv`
- v2 label changes: `runs/benchmark-squad500-smollm2-360m-answermean-20260604T1820Z/label_changes_v2.csv`

## High-Confidence Wrong Review

| Model | Candidate Cases | Manual Correct | Genuine Wrong or Incomplete |
|---|---:|---:|---:|
| Qwen | 14 | 1 | 13 |
| SmolLM2 | 27 | 1 | 26 |

The two likely false-negative labels were:

- Qwen `squad-validation-125`: the model listed the three venue names, but omitted location qualifiers and changed order.
- SmolLM2 `squad-validation-172`: the model answered `Super Bowl XXXVII`, which is the 2003 California Super Bowl referenced by the gold answer.

Most reviewed high-confidence cases were genuine failures:

- wrong entity: `Ronnie Hillman` instead of `C. J. Anderson`
- wrong injury: `ACL tear` instead of `broken arm`
- wrong team: `Denver Broncos` instead of `Carolina Panthers`
- incomplete multi-answer response: one player or venue when the question asks for two or three
- answer type mismatch: event name or sentence fragment where the question asks for a date, team, score, or Roman numeral

## v2 Label Changes

Qwen had no SQuAD-500 v2 label changes.

SmolLM2 had 12 SQuAD-500 v2 label changes. The changes are reasonable SQuAD-style corrections where the answer sentence contains the gold span, for example:

- `Von Miller had five solo tackles in Super Bowl 50.`
- `Levi's Stadium cost $1.2 billion.`
- `The theme color for the Super Bowl was gold.`

These changes improve measurement quality without rerunning inference.

## Manual-Corrected Metric Check

This check only changes held-out labels for the two reviewed false negatives. It does not retrain the probe and should be read as a sensitivity check, not as the official metric table.

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.1079 | 0.1515 | 0.7900 | 0.7535 | 8 | 6 |
| Qwen | Conservative TCL-v0 | 0.1190 | 0.1455 | 0.8100 | 0.8193 | 4 | 3 |
| SmolLM2 | Raw generation confidence | 0.2572 | 0.2810 | 0.5600 | 0.7415 | 20 | 2 |
| SmolLM2 | Conservative TCL-v0 | 0.1314 | 0.1650 | 0.7700 | 0.8378 | 6 | 0 |

## Interpretation

The manual review does not overturn the SQuAD-500 result.

For Qwen, conservative TCL-v0 remains useful for Brier score, threshold accuracy, AUC, and reducing high-confidence wrong answers, but raw confidence remains stronger on ECE.

For SmolLM2, conservative TCL-v0 remains clearly stronger than raw confidence across ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts.

## Claim Boundary

This is a targeted review, not a full manual relabel of SQuAD-500. It supports using the SQuAD-500 diagnostics as preliminary TCL-v0 evidence, but not as final validation.
