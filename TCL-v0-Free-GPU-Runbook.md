# TCL-v0 Free GPU Runbook

Status: execution guide for Colab/Kaggle dry runs

This runbook explains how to try TCL-v0 Extended Validation on free GPU platforms before paying for cloud compute. Free GPU should be treated as an execution test and possibly a modest validation run, not guaranteed full extended validation.

## 1. Free GPU Options

Current recommendation:

```text
Use Kaggle Notebooks first. Use Google Colab Free as fallback.
```

Reason:

Kaggle is better for a reproducible run with saved notebook artifacts. Colab Free is useful for quick checks, but its free GPU access is more variable and less guaranteed.

### Google Colab Free

Use for:

- quick GPU pipeline checks
- SQuAD-200 or SQuAD-1000 dry runs
- confirming that `run_inference.py` uses CUDA correctly

Limitations:

- GPU access is not guaranteed
- sessions can disconnect
- runtime length and GPU type vary
- usually not reliable enough for the full extended-validation matrix

### Kaggle Notebooks

Use for:

- more structured free notebook runs
- saving outputs as notebook artifacts
- SQuAD-1000 and possibly TriviaQA-1000 if runtime quota allows

Limitations:

- weekly GPU quota
- session length limit
- internet and persistence settings must be checked

## 2. Free GPU Strategy

The first free-GPU target should be:

```text
TCL-v0 Extended Validation Dry Run
```

Recommended dry-run matrix:

- benchmark: SQuAD with context
- examples: 200 first, then 1000 if stable
- model: `Qwen/Qwen2.5-1.5B-Instruct` if the GPU can handle it
- fallback model: `Qwen/Qwen2.5-0.5B-Instruct`
- hidden-state method: `answer_mean`
- device: CUDA
- dtype: float16

Do not start with the full two-benchmark, two-model matrix on free GPU. First prove that the cloud notebook can finish one clean run and preserve outputs.

## 3. Colab/Kaggle Setup Commands

Recommended path:

```text
Open notebooks/tcl_v0_kaggle_free_gpu_dry_run.ipynb in Kaggle first.
```

That notebook already contains the setup, runtime check, SQuAD-200 dry run, SQuAD-1000 escalation, and output-saving checklist. The commands below are the same workflow written out manually for debugging or Colab fallback.

Run these commands in a notebook cell.

```bash
git clone https://github.com/awabmoha/truth-calibration-layer.git
cd truth-calibration-layer/tcl_experiments
python -m pip install -r requirements.txt
python scripts/check_runtime.py
```

The runtime check should show:

```text
"cuda_available": true
```

If CUDA is false, stop. The notebook is not using GPU.

## 3.1 Recommended One-Command Dry Run

After setup, run:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 200 Qwen/Qwen2.5-0.5B-Instruct
```

If that completes and the outputs are saved correctly, try:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 1000 Qwen/Qwen2.5-1.5B-Instruct
```

If the 1.5B model fails due to memory, use:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 1000 Qwen/Qwen2.5-0.5B-Instruct
```

## 4. Prepare Benchmark Data

Start with SQuAD-200 to verify the full pipeline quickly.

```bash
python scripts/prepare_squad_subset.py --limit 200
```

Then, if stable, prepare SQuAD-1000.

```bash
python scripts/prepare_squad_subset.py --limit 1000
```

Expected files:

```text
data/benchmarks/squad/squad_validation_200.csv
data/benchmarks/squad/squad_validation_200_splits.csv
data/benchmarks/squad/squad_validation_1000.csv
data/benchmarks/squad/squad_validation_1000_splits.csv
```

## 5. Run GPU Inference

Small dry run:

```bash
python scripts/run_inference.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset squad_validation_200 \
  --questions data/benchmarks/squad/squad_validation_200.csv \
  --splits data/benchmarks/squad/squad_validation_200_splits.csv \
  --run-id freegpu-squad200-qwen05-answermean \
  --hidden-state-method answer_mean \
  --device cuda \
  --torch-dtype float16 \
  --max-new-tokens 12
```

Stronger dry run, if GPU memory allows:

```bash
python scripts/run_inference.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset squad_validation_1000 \
  --questions data/benchmarks/squad/squad_validation_1000.csv \
  --splits data/benchmarks/squad/squad_validation_1000_splits.csv \
  --run-id freegpu-squad1000-qwen15-answermean \
  --hidden-state-method answer_mean \
  --device cuda \
  --torch-dtype float16 \
  --max-new-tokens 12
```

If the 1.5B model fails due to memory, use the 0.5B model on SQuAD-1000 instead.

## 6. Train Probe And Compute Metrics

For SQuAD-200:

```bash
python scripts/train_probe.py \
  --records runs/freegpu-squad200-qwen05-answermean/records_answer_mean.jsonl \
  --out-dir runs/freegpu-squad200-qwen05-answermean/analysis \
  --split-mode record \
  --calibrate
```

For SQuAD-1000:

```bash
python scripts/train_probe.py \
  --records runs/freegpu-squad1000-qwen15-answermean/records_answer_mean.jsonl \
  --out-dir runs/freegpu-squad1000-qwen15-answermean/analysis \
  --split-mode record \
  --calibrate
```

## 7. Manual Review CSV

Create a first review CSV from the records:

```bash
python scripts/make_review_csv.py \
  --records runs/freegpu-squad1000-qwen15-answermean/records_answer_mean.jsonl \
  --out runs/freegpu-squad1000-qwen15-answermean/manual_review_all.csv
```

For the final extended-validation decision, manual review should focus on high-confidence wrong cases and strong raw-vs-TCL disagreement cases. This broader review CSV is only the first artifact.

## 7.1 Artifact Verification

The one-command dry-run script automatically verifies the returned artifact. If you need to rerun verification manually, use:

```bash
python scripts/verify_run_artifact.py \
  --run-dir runs/<run_id> \
  --method answer_mean \
  --min-records 200 \
  --require-manual-review \
  --require-calibrated \
  --out-json runs/<run_id>/artifact_verification.json \
  --out-md runs/<run_id>/ARTIFACT_VERIFICATION.md
```

This check confirms that records, probe metrics, reliability bins, test predictions, and review files exist before any result is interpreted.

## 8. Save Outputs Before The Session Ends

Free notebooks can disconnect. Before closing the session, save:

```text
runs/<run_id>/records_answer_mean.jsonl
runs/<run_id>/records_answer_mean.config.json
runs/<run_id>/analysis/
runs/<run_id>/manual_review_all.csv
runs/<run_id>/artifact_verification.json
runs/<run_id>/ARTIFACT_VERIFICATION.md
data/benchmarks/squad/*_1000*
```

On Colab, copy the folder to Google Drive or download it as a zip.

On Kaggle, save the notebook output as an artifact.

## 9. How To Interpret A Free-GPU Result

A free-GPU dry run can prove:

- CUDA execution works
- stronger model loading works or fails
- SQuAD-1000 pipeline can finish or cannot finish
- the metrics path still works outside the local laptop

It cannot prove full extended validation by itself unless the predeclared matrix is completed and manually reviewed.

## 10. Decision After Free-GPU Dry Run

After the dry run, choose one:

- Free GPU is enough for the full SQuAD-1000 plus TriviaQA-1000 matrix.
- Free GPU is enough only for script validation, and paid GPU is needed for the real run.
- Free GPU is too unstable, so pause and plan paid/borrowed compute.

The decision should be recorded before running more experiments.
