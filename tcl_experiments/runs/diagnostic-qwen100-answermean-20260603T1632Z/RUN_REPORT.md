# TCL-v0 Diagnostic Run Report

Run ID: `diagnostic-qwen100-answermean-20260603T1632Z`

Status: diagnostic run, not benchmark evidence

## Purpose

This run tested whether the TCL-v0 pipeline can record a larger factual QA diagnostic set and train a confidence-only probe from frozen LLM hidden states.

The run does not validate the full Truth Calibration Layer framework. It is an early diagnostic for the narrow TCL-v0 hypothesis:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

## Local Hardware Context

- Machine: Lenovo 82RN
- CPU: AMD Ryzen 7 5825U with Radeon Graphics
- CPU cores/logical processors: 8 cores / 16 logical processors
- RAM: about 6 GB
- GPU: integrated AMD Radeon graphics
- CUDA/NVIDIA GPU: not available

## Model and Data

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Dataset file: `data/diagnostic_questions_100.csv`
- Dataset name recorded in outputs: `diagnostic_questions_100`
- Number of questions: 100
- Hidden-state method: `answer_mean`
- Hidden-state layer: `-1` / final layer
- Max new tokens: 12
- Raw confidence method: geometric mean generated-token probability
- Correctness method: normalized exact match or accepted-answer substring match

## Output Files

- Run config: `records_answer_mean.config.json`
- Full records: `records_answer_mean.jsonl`
- Analysis summary: `analysis/summary.json`
- Method summary: `analysis/method_summary.csv`
- Per-method metrics: `analysis/answer_mean/metrics.json`
- Raw reliability bins: `analysis/answer_mean/raw_reliability_bins.csv`
- TCL-v0 reliability bins: `analysis/answer_mean/tcl_v0_reliability_bins.csv`
- Test predictions: `analysis/answer_mean/test_predictions.csv`
- Train/test split record: `analysis/answer_mean/splits.csv`

## Label Summary

- Total records: 100
- Correct labels: 91
- Incorrect labels: 9
- Overall model correctness on this diagnostic set: 0.91
- Incorrect IDs: `d006`, `d009`, `d023`, `d033`, `d040`, `d055`, `d084`, `d091`, `d100`

## Probe Split

- Train records: 65
- Test records: 35
- Positive rate across all records: 0.91

## Diagnostic Metrics

Raw generation confidence:

- Accuracy at 0.5 threshold: 0.7428571428571429
- ECE: 0.30154668930877443
- MCE: 0.6029596152236215
- Brier score: 0.18326885375917434
- AUC: 0.4791666666666667

TCL-v0 probe confidence:

- Accuracy at 0.5 threshold: 0.9142857142857143
- ECE: 0.08158904310122352
- MCE: 0.1978934338283317
- Brier score: 0.08609424034540128
- AUC: 0.5104166666666666

## Interpretation

The diagnostic run shows that the TCL-v0 pipeline can:

- Run a small instruction model locally on CPU.
- Record model answers, correctness labels, raw confidence, hidden states, and run metadata.
- Train a logistic-regression probe on frozen hidden-state vectors.
- Produce calibration metrics and reliability-bin outputs.

The probe had lower ECE and Brier score than raw generation confidence on this diagnostic split. This is interesting, but it should be treated only as a preliminary pipeline signal.

## Cautions

This run is not enough to claim TCL works.

Main limitations:

- The dataset is a locally curated diagnostic set, not a standard benchmark.
- The model answered 91 percent correctly, so the labels are imbalanced.
- The test split has only 35 examples.
- Correctness labeling uses simple string matching and still needs manual review for research-grade results.
- Only one hidden-state extraction method was tested.
- Only one model was tested.
- The AUC result is weak, so the probe is not yet a reliable ranker of correct versus incorrect examples.

Allowed claim from this run:

The TCL-v0 pipeline is functioning and produced an initial diagnostic where hidden-state probe confidence had lower ECE and Brier score than raw generation confidence on a small local factual QA set.

Not allowed:

- TCL has been validated.
- TCL makes the model truthful.
- TCL solves hallucination.
- The full four-dimensional TCL framework has been implemented.

## Recommended Next Step

Run the same 100-question diagnostic with at least one additional hidden-state method, preferably `answer_last`, then compare method summaries. After that, move from the local diagnostic CSV to a standard factual QA benchmark subset.

