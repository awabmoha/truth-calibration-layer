# TCL-v0 SQuAD Context 500 Diagnostic

Status: context-grounded benchmark diagnostic, not broad validation

This run scaled the SQuAD context benchmark from 100 to 500 examples after the smaller diagnostic showed that SQuAD gives the local CPU-runnable models a healthier mix of correct and incorrect answers than NQ-Open.

Protocol:

- dataset: `rajpurkar/squad`
- config: `plain_text`
- source split: `validation`
- prepared subset: 500 examples
- split: 325 train, 75 validation, 100 test
- prompt: `chat_short_factual_answer_v1` with context included
- records preserve `context`, `raw_model_output`, and cleaned `model_answer`
- hidden-state method: `answer_mean`
- relabeled correctness method: `strict_answer_segment_match_v2`

## Runs

- Qwen: `runs/benchmark-squad500-qwen-answermean-20260604T1652Z/`
- SmolLM2: `runs/benchmark-squad500-smollm2-360m-answermean-20260604T1820Z/`

The model generations were not rerun after the labeler update. The v2 label pass only relabeled existing records.

## Label Summary

| Model | All Correct | All Incorrect | Test Correct | Test Incorrect | v2 Label Changes |
|---|---:|---:|---:|---:|---:|
| Qwen | 383 | 117 | 77 | 23 | 0 |
| SmolLM2 | 272 | 228 | 53 | 47 | 12 |

## Test Metrics

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.0979 | 0.1582 | 0.7800 | 0.7527 | 9 | 6 |
| Qwen | TCL-v0 probe confidence | 0.1528 | 0.1687 | 0.7900 | 0.7719 | 10 | 7 |
| Qwen | Validation-calibrated TCL-v0 | 0.2269 | 0.1963 | 0.7300 | 0.7719 | 0 | 0 |
| Qwen | Conservative TCL-v0 | 0.1090 | 0.1521 | 0.8000 | 0.8097 | 5 | 3 |
| SmolLM2 | Raw generation confidence | 0.2672 | 0.2849 | 0.5500 | 0.7523 | 20 | 2 |
| SmolLM2 | TCL-v0 probe confidence | 0.1898 | 0.1982 | 0.7500 | 0.8282 | 13 | 10 |
| SmolLM2 | Validation-calibrated TCL-v0 | 0.1338 | 0.1720 | 0.7500 | 0.8282 | 1 | 1 |
| SmolLM2 | Conservative TCL-v0 | 0.1275 | 0.1688 | 0.7600 | 0.8366 | 6 | 0 |

## Interpretation

The 500-example SQuAD diagnostic is more informative than the 100-example protocol check because the held-out test split has 100 examples and a healthier positive/negative balance.

For Qwen:

- conservative TCL-v0 improves Brier score, threshold accuracy, and AUC over raw generation confidence
- raw generation confidence still has the best ECE
- conservative TCL-v0 reduces high-confidence wrong answers, but does not eliminate them

For SmolLM2:

- TCL-v0 probe confidence improves ECE, Brier score, threshold accuracy, and AUC over raw generation confidence
- conservative TCL-v0 is the strongest overall score on ECE, Brier score, accuracy, and AUC
- validation calibration sharply reduces high-confidence wrong answers

The strongest cautious reading is:

```text
On SQuAD-500, frozen hidden states contain useful calibration signal, but the best way to combine that signal with raw confidence remains model-dependent.
```

## Labeler v2 Note

The original strict labeler was too harsh for some SQuAD-style extractive answers where the model produced a short sentence containing the gold span, for example `$1.2 billion` inside `Levi's Stadium cost $1.2 billion.`

The v2 rule still avoids fuzzy edit-distance matching, but allows exact normalized answer spans inside answer sentences and fixes numeric-phrase span matching. Spot checks preserved known dangerous negatives such as:

- `Boulder, Colorado` should not match `Colorado` for a river question
- `Apollo 11` should not match `Apollo 17`
- `Golden Anniversary` should not match `Super Bowl L`

## Claim Boundary

These are TCL-v0 confidence diagnostics only. They do not validate full TCL, do not prove generalization, and do not show that LLMs become truthful. They support the narrower hypothesis that frozen hidden states can contain useful correctness-calibration signal under tested conditions.

## Next Step

Run a targeted manual review of the SQuAD-500 held-out test split, especially high-confidence wrong cases and v2 label changes, before using these numbers in any public-facing research claim.
