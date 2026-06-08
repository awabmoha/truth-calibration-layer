# TCL-v0 Kaggle Ablation Runbook

Status: manual execution guide for the next Kaggle checkpoint.

Use this only after reading `TCL-v0-Ablation-Plan.md`.

## Goal

Run a small ablation smoke test before any expensive 1,000-example ablation.

The smoke test checks:

- Gemma-safe chat-template fallback
- layer-position extraction: `early_middle`, `middle`, `final`
- pooling methods: `answer_mean`, `answer_last`, `prompt_answer_mean`
- raw-only calibration baselines
- hidden-only, raw-plus-hidden, and probe-score-plus-raw ablation analysis

## Kaggle Setup

Use one notebook session.

Recommended first target:

```text
model: /kaggle/input/models/google/gemma/transformers/2b-it/3
benchmark: TriviaQA rc.nocontext
size: 100 examples smoke test
```

Do not start with 1,000 examples. First prove the ablation code runs end to end.

## Step 1: Clone Or Upload Repo

In Kaggle:

```bash
git clone https://github.com/awabmoha/truth-calibration-layer.git
cd truth-calibration-layer/tcl_experiments
```

If you are using an uploaded copy instead of git clone, `cd` into:

```bash
/kaggle/working/truth-calibration-layer/tcl_experiments
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Kaggle may show CUDA package warnings. If `torch.cuda.is_available()` is true and the runtime check passes, continue.

The ablation analysis requires:

```text
numpy
pandas
scikit-learn
torch
transformers
```

These are listed in `tcl_experiments/requirements.txt`.

## Step 3: Check Runtime

```bash
python scripts/check_runtime.py
```

Save the output in the notebook.

## Step 4: Prepare TriviaQA-1000

If the prepared files are not already present:

```bash
python scripts/prepare_triviaqa_subset.py --limit 1000 --seed 42
```

Expected files:

```text
data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_1000.csv
data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_1000_splits.csv
```

## Step 5: Run 100-Example Ablation Smoke Test

For Kaggle-hosted Gemma 2B-it:

```bash
export MODEL="/kaggle/input/models/google/gemma/transformers/2b-it/3"
export DATASET="triviaqa_rc_nocontext_validation_1000"
export QUESTIONS="data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_1000.csv"
export SPLITS="data/benchmarks/triviaqa/triviaqa_rc_nocontext_validation_1000_splits.csv"
export RUN_PREFIX="ablation-smoke-gemma-triviaqa100"
export LIMIT=100
export DEVICE="auto"
export DTYPE="auto"

bash scripts/run_free_gpu_ablation_smoke.sh
```

Expected output folder:

```text
runs/ablation-smoke-gemma-triviaqa100-ablation_summary/
```

Key files:

```text
ablation_metrics.json
ablation_summary.csv
test_predictions/
```

## Step 6: Save Artifacts

Before closing Kaggle, save:

```text
runs/ablation-smoke-gemma-triviaqa100-ablation_summary/
runs/ablation-smoke-gemma-triviaqa100-*/
```

If possible, zip them:

```bash
zip -r runs/ablation-smoke-gemma-triviaqa100_artifact.zip \
  runs/ablation-smoke-gemma-triviaqa100-ablation_summary \
  runs/ablation-smoke-gemma-triviaqa100-*
```

Download the zip and give it back to Codex.

## Step 7: Interpret Smoke Test

The smoke test is successful if:

- all 9 run folders are created
- `ablation_metrics.json` exists
- `ablation_summary.csv` has rows
- the JSON `errors` list is empty

If the test split has only one class, the smoke test can still prove the pipeline runs, but it is not evidence. In that case, run a larger smoke test with `LIMIT=200` before the 1,000-example run.

## Step 8: Only If Smoke Test Works

Run the same ablation with:

```bash
export LIMIT=1000
export RUN_PREFIX="ablation-gemma-triviaqa1000"
bash scripts/run_free_gpu_ablation_smoke.sh
```

This will be much slower because it runs 9 inference passes:

```text
3 pooling methods x 3 layer positions
```

If time is limited, use a smaller locked subset:

```text
answer_mean x early_middle/middle/final
```

or:

```text
final layer x answer_mean/answer_last/prompt_answer_mean
```

## What To Send Back

Send Codex:

- the final `ablation_summary.csv`
- `ablation_metrics.json`
- the artifact zip if available
- any traceback if a cell fails

For the updated default smoke test, the artifact path is:

```text
runs/ablation-smoke-gemma-triviaqa100_artifact.zip
```

## Stop Conditions

Stop and send the error if:

- the model cannot load
- train or validation has only one class
- any ablation run produces empty generations
- CUDA runs out of memory
- the smoke test takes too long

Do not run the 1,000-example ablation until the smoke test completes successfully.
