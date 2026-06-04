# TCL-v0 Clean NQ-Open 100 Protocol Check

Status: protocol check, not broad validation

This run tested the improved benchmark protocol on a small NQ-Open subset before scaling to another 500-example benchmark.

Protocol:

- dataset: `google-research-datasets/nq_open`
- config: `nq_open`
- source split: `validation`
- prepared subset: 100 examples
- split: 65 train, 15 validation, 20 test
- prompt: `chat_short_factual_answer_v1`
- records preserve `raw_model_output` and cleaned `model_answer`
- correctness method: `strict_answer_segment_match_v1`

## Runs

- Qwen: `runs/benchmark-nqopen100-qwen-answermean-20260604T1153Z/`
- SmolLM2: `runs/benchmark-nqopen100-smollm2-360m-answermean-20260604T1200Z/`

## Test Label Summary

| Model | Test Correct | Test Incorrect | Reviewed High-Risk Cases | Reviewed Positive Labels | Label Changes |
|---|---:|---:|---:|---:|---:|
| Qwen | 1 | 19 | 1 | 1 | 0 |
| SmolLM2 | 1 | 19 | 1 | 1 | 0 |

## Test Metrics

| Model | Signal | ECE | Brier | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.4143 | 0.2164 | 0.6000 | 0 | 0 |
| Qwen | TCL-v0 probe confidence | 0.0551 | 0.0512 | 0.9500 | 0 | 0 |
| Qwen | Conservative TCL-v0 | 0.0551 | 0.0512 | 0.9500 | 0 | 0 |
| SmolLM2 | Raw generation confidence | 0.5005 | 0.3044 | 0.4000 | 1 | 0 |
| SmolLM2 | TCL-v0 probe confidence | 0.0450 | 0.0469 | 0.9500 | 0 | 0 |
| SmolLM2 | Conservative TCL-v0 | 0.0450 | 0.0469 | 0.9500 | 0 | 0 |

## Interpretation

The improved protocol works technically:

- no blank outputs in the reviewed test cases
- raw output is preserved separately from cleaned answer text
- positive labels and high-risk cases were easy to audit
- label review found no changes in the small held-out test split

The results still need caution:

- the held-out test split has only 20 examples
- each model has only 1 correct test answer
- metrics are therefore protocol-check metrics, not strong benchmark evidence

The clean protocol is ready to scale, but the next larger run should use a benchmark or prompt setup with more positive examples.

## Next Step

Run a larger clean benchmark only if it is likely to produce enough positives for meaningful probe training and validation. Otherwise, use a cleaner short-answer benchmark before scaling.
