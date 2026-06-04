# TCL-v0 Labeling Rule Update

Status: stricter label audit, not model rerun

## Why This Was Needed

The original automatic correctness method used normalized exact match or word-boundary accepted-answer matching. That was useful for early diagnostics, but manual review found false positives where the model repeated words from the question or gave a related but wrong entity.

Examples from the Qwen held-out test split:

- The model restated "European Recovery Program" without answering "Marshall Plan."
- The model answered "George VI" for a question intended to identify King George / George III.
- The model answered "Boulder, Colorado" when the question asked for the river containing Boulder Dam.

## New Method

New correctness method:

```text
strict_answer_segment_match_v1
```

The stricter rule:

- extracts likely answer segments before matching
- ignores multi-word accepted answers that merely echo the question
- handles blank or prompt-echo outputs as incorrect
- treats special cases such as river questions more carefully
- avoids fuzzy edit-distance matching because it created unsafe false positives

## Relabeling Scope

Relabeled existing records only. No model inference was rerun.

| Run | Old Positives | Strict Positives | Label Changes |
|---|---:|---:|---:|
| Qwen 500 | 110 | 100 | 14 |
| SmolLM2 500 | 46 | 35 | 13 |

## Strict-Label Test Metrics

### Qwen 500

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.2793 | 0.2107 | 0.3537 | 0.6300 | 0.8402 | 0 | 0 |
| TCL-v0 probe confidence | 0.1252 | 0.1380 | 0.3601 | 0.8400 | 0.8041 | 4 | 3 |
| Conservative TCL-v0 | 0.1088 | 0.1268 | 0.3601 | 0.8400 | 0.8108 | 0 | 0 |

### SmolLM2 500

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4519 | 0.2953 | 0.9212 | 0.4800 | 0.5714 | 6 | 3 |
| TCL-v0 probe confidence | 0.0961 | 0.0824 | 0.8522 | 0.9000 | 0.9339 | 5 | 2 |
| Conservative TCL-v0 | 0.0661 | 0.0597 | 0.6442 | 0.9300 | 0.9309 | 0 | 0 |

SmolLM2 validation calibration was not available under strict labels because the validation split did not have enough positive examples for a fitted calibrator.

## Interpretation

The stricter label audit preserves the main TCL-v0 result:

- Conservative TCL-v0 remains the best current diagnostic variant across both models.
- Conservative TCL-v0 improves ECE, Brier score, and threshold accuracy over raw generation confidence.
- Conservative TCL-v0 avoids high-confidence wrong answers in both strict-label held-out test splits.
- The plain probe still produces high-confidence wrong answers, so the conservative rule remains necessary.

## Claim Boundary

This update improves label quality for TCL-v0 diagnostics. It does not validate full TCL, prove hidden states generally predict truth, or solve hallucination.
