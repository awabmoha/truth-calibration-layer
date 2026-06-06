# TCL-v0 Experiment Pipeline

This folder is the first practical step for the Truth Calibration Layer idea.
The goal is intentionally small:

> Test whether frozen LLM hidden states contain useful signal for predicting answer correctness.

TCL-v0 does not implement the full four-dimensional trust vector. It implements only the first dimension:
`hidden_state -> correctness/confidence score`.

Current best diagnostic variant:

```text
conservative TCL-v0 confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This conservative score is used because the plain probe can become overconfident on fluent wrong answers.

## What It Records

For each factual question, the pipeline records:

- dataset name
- question
- accepted answers
- model name
- prompt template
- model answer
- exact/substring correctness label
- raw generation confidence and its method
- hidden-state layer and extraction method
- hidden-state vector
- run metadata

Then it trains a simple probe on the hidden states and compares:

- raw generation confidence
- TCL-v0 probe confidence
- conservative TCL-v0 confidence
- calibrated TCL-v0 confidence, when requested

using accuracy, expected calibration error, maximum calibration error, Brier score, reliability bins, and high-confidence wrong-answer counts.

## Files

- `requirements.txt` - Python dependencies.
- `data/seed_questions.csv` - small starter factual QA set.
- `scripts/run_inference.py` - runs an LLM, extracts hidden states, and writes records.
- `scripts/train_probe.py` - trains TCL-v0 and writes calibration metrics, including conservative confidence.
- `scripts/prepare_triviaqa_subset.py` - prepares a reproducible TriviaQA subset and split files.
- `scripts/prepare_nq_open_subset.py` - prepares a reproducible NQ-Open subset and split files.
- `scripts/prepare_squad_subset.py` - prepares a reproducible context-grounded SQuAD subset and split files.
- `scripts/make_review_csv.py` - creates a manual-review CSV from recorded model outputs.
- `scripts/make_targeted_review_csv.py` - creates the predeclared targeted manual-review candidate set.
- `scripts/validate_targeted_review.py` - checks whether targeted manual review is complete before final decisions.
- `scripts/recompute_reviewed_metrics.py` - recomputes test metrics after manual label review.
- `scripts/verify_run_artifact.py` - checks returned run artifacts before interpreting TCL-v0 metrics.
- `scripts/package_run_artifact.py` - zips a verified run folder plus benchmark metadata for download.
- `scripts/import_run_artifact.py` - safely unpacks a downloaded run artifact and reruns verification locally.
- `scripts/summarize_extended_validation.py` - aggregates verified runs into a TCL-v0 decision note.
- `scripts/metrics.py` - shared calibration utilities.
- `scripts/check_runtime.py` - reports Python, PyTorch, Transformers, and CUDA/GPU availability.
- `scripts/run_free_gpu_squad_dry_run.sh` - one-command SQuAD dry run for Kaggle/Colab GPU notebooks.
- `scripts/run_free_gpu_triviaqa_dry_run.sh` - one-command TriviaQA dry run for Kaggle/Colab GPU notebooks.
- `scripts/build_tcl_v0_writeup_docx.py` - rebuilds the TCL-v0 research writeup DOCX from the Markdown source.
- `scripts/build_tcl_v0_writeup_pdf.py` - rebuilds the standalone TCL-v0 research writeup PDF from the Markdown source.

## Quick Start

Use this only after agreeing to run a smoke test. The tiny model and seed questions are for checking that the pipeline works; they are not evidence for TCL.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_inference.py --model sshleifer/tiny-gpt2 --limit 20 --hidden-state-method answer_mean
.\.venv\Scripts\python.exe scripts\train_probe.py --records runs\<run_id>\records_answer_mean.jsonl --out-dir runs\<run_id>\analysis
```

`sshleifer/tiny-gpt2` is only for a fast smoke test. For meaningful evidence, use a stronger instruction model later.

## Hidden-State Methods

`scripts/run_inference.py` supports:

- `answer_mean` - mean-pooled generated-answer hidden states.
- `answer_last` - final generated-token hidden state.
- `prompt_answer_mean` - mean-pooled prompt plus generated-answer hidden states.

The preferred starting method is `answer_mean`, because TCL-v0 is trying to predict correctness of the produced answer.

## Benchmark Diagnostic

The first benchmark diagnostic used:

- Dataset: TriviaQA `rc.nocontext`, validation subset.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Hidden-state method: `answer_mean`.
- Best current diagnostic score: conservative TCL-v0 confidence.

Main benchmark reports live in:

```text
runs/benchmark-triviaqa200-qwen-answermean-20260603T1700Z/
runs/benchmark-triviaqa500-qwen-answermean-20260604T0932Z/
runs/benchmark-triviaqa500-smollm2-360m-answermean-20260604T1018Z/
runs/benchmark-squad500-qwen-answermean-20260604T1652Z/
runs/benchmark-squad500-smollm2-360m-answermean-20260604T1820Z/
```

Key reports:

- `RUN_REPORT.md`
- `FAILURE_ANALYSIS.md`
- `CALIBRATION_REPORT.md`
- `CONSERVATIVE_CONFIDENCE_REPORT.md`
- `MANUAL_REVIEW_TEST_REPORT.md`
- `HIGH_RISK_REVIEW_REPORT.md`
- `EXTENDED_MANUAL_REVIEW_REPORT.md`
- `TCL-v0-cross-model-comparison.md`
- `TCL-v0-labeling-rule-update.md`
- `TCL-v0-nq-open-benchmark-summary.md`
- `TCL-v0-benchmark-protocol-update.md`
- `TCL-v0-clean-nq-open-100-summary.md`
- `TCL-v0-squad-context-100-summary.md`
- `TCL-v0-squad-context-500-summary.md`
- `TCL-v0-squad500-manual-review-report.md`

## Claim Boundary

TCL-v0 is a confidence-only probe experiment. A successful run may support the narrow claim that frozen hidden states improve calibration under tested conditions. It does not show that TCL makes a model truthful, solves hallucination, or validates the full four-dimensional TCL framework.

Current cautious claim:

Conservative TCL-v0 produced the best metrics on the 200-example and 500-example TriviaQA diagnostics. The pattern appears across two small CPU-runnable models on TriviaQA. NQ-Open has also been tested with the stricter label rule; it is useful but sparse under the current short-answer setup. SQuAD with context is now the cleaner local benchmark: on SQuAD-500, Qwen showed mixed but useful Brier/accuracy/AUC gains, while SmolLM2 showed stronger calibration gains under conservative TCL-v0. This is still not broad validation.
