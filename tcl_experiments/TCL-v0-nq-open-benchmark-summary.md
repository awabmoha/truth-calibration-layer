# TCL-v0 NQ-Open Benchmark Summary

Status: second benchmark diagnostic, not full TCL validation

This benchmark uses `google-research-datasets/nq_open`, config `nq_open`, validation split.

Prepared subset:

- 500 examples
- 325 train
- 75 validation
- 100 test
- correctness method: `strict_answer_segment_match_v1`

## Runs

- Qwen: `runs/benchmark-nqopen500-qwen-answermean-20260604T1049Z-retry2/`
- SmolLM2: `runs/benchmark-nqopen500-smollm2-360m-answermean-20260604T1118Z/`

Two earlier Qwen NQ-Open attempts stopped before recording examples because of schema/wiring bugs. They are local run traces only and are not benchmark results.

## Reviewed-Label Test Metrics

### Qwen NQ-Open 500

Manual review:

- high-risk wrong cases reviewed: 5
- automatic positive test labels reviewed: 6
- label changes: 2
- reviewed positives: 4/100

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4883 | 0.2757 | 0.8168 | 0.4800 | 0 | 0 |
| TCL-v0 probe confidence | 0.1348 | 0.1064 | 0.9165 | 0.8600 | 5 | 1 |
| Validation-calibrated TCL-v0 | 0.4538 | 0.2464 | 0.6063 | 0.5800 | 0 | 0 |
| Conservative TCL-v0 | 0.1018 | 0.0667 | 0.7128 | 0.9200 | 0 | 0 |

### SmolLM2 NQ-Open 500

Manual review:

- high-risk wrong cases reviewed: 7
- automatic positive test labels reviewed: 1
- label changes: 1
- reviewed positives: 2/100

| Signal | ECE | Brier | MCE | Accuracy at 0.5 | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---:|---:|---:|---:|---:|---:|
| Raw generation confidence | 0.4416 | 0.2376 | 0.8699 | 0.6200 | 3 | 0 |
| TCL-v0 probe confidence | 0.0135 | 0.0087 | 0.3905 | 0.9900 | 0 | 0 |
| Validation-calibrated TCL-v0 | 0.0788 | 0.0524 | 0.7712 | 0.9200 | 3 | 3 |
| Conservative TCL-v0 | 0.0164 | 0.0100 | 0.3905 | 0.9900 | 0 | 0 |

## Interpretation

NQ-Open is a harder and sparser benchmark under the current short-answer setup. The reviewed test split has very few positives:

- Qwen: 4/100 reviewed positives
- SmolLM2: 2/100 reviewed positives

This means NQ-Open is useful as a stress test, but the metrics should be interpreted cautiously.

The second benchmark still supports the central TCL-v0 direction:

- Hidden-state probe confidence improves calibration metrics over raw generation confidence.
- Conservative TCL-v0 remains safer than plain probe for Qwen because the plain probe produces high-confidence wrong answers.
- On SmolLM2 NQ-Open, the plain probe slightly beats conservative TCL-v0 on ECE/Brier, but conservative TCL-v0 remains nearly tied and also avoids high-confidence wrong answers.
- Validation-score calibration is not reliable in these small, sparse validation splits.

## Claim Boundary

Allowed:

- TCL-v0 has now been tested on two benchmark sources: TriviaQA and NQ-Open.
- Conservative TCL-v0 remains a strong and safer diagnostic variant.
- The results support continuing TCL-v0 experiments.

Not allowed:

- TCL is validated.
- TCL solves hallucination.
- TCL-v0 generalizes broadly.
- These sparse NQ-Open test metrics alone prove the method.

## Next Step

Improve the benchmark protocol for sparse open-domain QA:

- use a larger NQ-Open subset, or
- improve prompt/answer extraction so fewer answers are blank or truncated, or
- add a third benchmark with clearer short-answer labels.
