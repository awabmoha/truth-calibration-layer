# TCL-v0 Benchmark Preparation

Status: benchmark preparation and first diagnostic completed

## Benchmark Source

Primary first benchmark:

- Dataset: `mandarjoshi/trivia_qa`
- Config: `rc.nocontext`
- Source split: `validation`

This source was checked through the Hugging Face Dataset Viewer API. The rows include:

- `question`
- `question_id`
- `answer.value`
- `answer.aliases`
- `answer.normalized_value`
- `answer.normalized_aliases`

## Prepared Subset Plan

The first benchmark subset should be small enough for the local laptop:

- 200 examples from TriviaQA validation.
- Fixed seed: `42`.
- Split plan:
  - train: 65 percent
  - validation: 15 percent
  - test: remaining examples

For the current `train_probe.py`, the recorded `train`, `validation`, and `test` splits can be used with:

```powershell
.\.venv\Scripts\python.exe scripts\train_probe.py --records <records.jsonl> --out-dir <analysis_dir> --split-mode record --calibrate
```

The `validation` split is used only for optional post-hoc calibration. The conservative confidence rule does not train on validation; it is computed as the minimum of raw confidence and probe confidence.

## Correctness Labeling

The benchmark run will still use automatic matching at first:

- normalized exact match
- word-boundary accepted-answer match for accepted answers of length 3 or more

Manual review is still required for research-grade claims. The preparation script writes a manual-review CSV template for this reason.

## Default TCL-v0 Settings

Use the best current diagnostic setting:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Hidden-state method: `answer_mean`
- Hidden-state layer: final layer (`-1`)
- Raw confidence: geometric mean generated-token probability
- Probe: logistic regression
- Current best score: conservative TCL-v0 confidence, `min(raw_generation_confidence, probe_confidence)`

## First Benchmark Diagnostic Result

The first 200-example TriviaQA diagnostic was run and recorded here:

```text
runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/
```

The strongest variant was conservative TCL-v0:

- ECE: 0.1347
- Brier score: 0.1257
- Accuracy at 0.5 threshold: 0.8250
- Wrong test examples with confidence >= 0.8: 0

Important caveat:

This is still a small diagnostic run. It should guide the next experiment, not be treated as full validation.

## Claim Boundary

Allowed after preparation and first diagnostic:

The project has a reproducible TriviaQA benchmark subset and an initial diagnostic showing that conservative TCL-v0 performed best on the tested split.

Not allowed:

- The result does not validate TCL broadly.
- The result does not prove the conservative rule generalizes.
- Larger benchmark runs and manual review are still required before paper-level empirical claims.
