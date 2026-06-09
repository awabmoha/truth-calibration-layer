# TCL-v0 Kaggle Ablation Runbook

Status: reproducible Kaggle execution protocol for the next ablation checkpoint.

This protocol assumes `TCL-v0-Ablation-Plan.md` has been reviewed and accepted as the governing experiment plan.

## Goal

Run a small ablation smoke test before any higher-cost 1,000-example ablation.

The smoke test checks:

- Gemma-safe chat-template fallback
- layer-position extraction: `early_middle`, `middle`, `final`
- pooling methods: `answer_mean`, `answer_last`, `prompt_answer_mean`
- raw-only calibration baselines
- hidden-only, raw-plus-hidden, and probe-score-plus-raw ablation analysis

## Kaggle Setup

The protocol should be executed in a single Kaggle notebook session.

Recommended first target:

```text
model: /kaggle/input/models/google/gemma/transformers/2b-it/3
benchmark: TriviaQA rc.nocontext
size: 100 examples smoke test
```

The 1,000-example ablation is intentionally deferred until the smoke test completes end to end.

## Step 1: Clone Or Upload Repository

For a fresh Kaggle notebook:

```bash
git clone https://github.com/awabmoha/truth-calibration-layer.git
cd truth-calibration-layer/tcl_experiments
```

For an uploaded repository copy, use:

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

Record the output in the notebook log.

## Step 4: Prepare TriviaQA-1000

If the prepared files are not already present, create them with:

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

Before closing the Kaggle session, preserve:

```text
runs/ablation-smoke-gemma-triviaqa100-ablation_summary/
runs/ablation-smoke-gemma-triviaqa100-*/
```

The preferred artifact bundle can be created with:

```bash
zip -r runs/ablation-smoke-gemma-triviaqa100_artifact.zip \
  runs/ablation-smoke-gemma-triviaqa100-ablation_summary \
  runs/ablation-smoke-gemma-triviaqa100-*
```

The resulting zip should be downloaded and retained for local import, verification, and review.

## Step 7: Interpret Smoke Test

The smoke test is successful if:

- all 9 run folders are created
- `ablation_metrics.json` exists
- `ablation_summary.csv` has rows
- the JSON `errors` list is empty

If the test split has only one class, the smoke test can still demonstrate that the pipeline runs, but it is not evidence for the research claim. In that case, a larger smoke test with `LIMIT=200` should be completed before the 1,000-example run.

## Step 8: Full Ablation Escalation

This step should be run only after the smoke test completes successfully.

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

If notebook time is limited, use a smaller locked subset:

```text
answer_mean x early_middle/middle/final
```

or:

```text
final layer x answer_mean/answer_last/prompt_answer_mean
```

## Artifacts For Local Review

The following artifacts are required for local review:

- the final `ablation_summary.csv`
- `ablation_metrics.json`
- the artifact zip if available
- any traceback produced by a failed notebook cell

For the updated default smoke test, the artifact path is:

```text
runs/ablation-smoke-gemma-triviaqa100_artifact.zip
```

## Stop Conditions

Execution should stop if:

- the model cannot load
- train or validation has only one class
- any ablation run produces empty generations
- CUDA runs out of memory
- the smoke test takes too long

The 1,000-example ablation should not be run until the smoke test completes successfully.
