# TCL-v0 Free GPU Decision

Status: current recommendation for no-cost extended-validation dry runs

## Decision

Use **Kaggle Notebooks first** for the TCL-v0 Extended Validation free-GPU dry run.

Use **Google Colab Free** as the fallback.

## Why Kaggle First

Kaggle is the better first free option for this project because:

- it is notebook-based and easy to run without local GPU hardware
- it supports GPU notebooks
- it has a clearer weekly GPU-quota pattern than Colab Free
- notebook outputs can be saved as artifacts
- it is better suited to running a reproducible script from top to bottom

Public Kaggle-oriented documentation reports a 30 hour per week GPU limit, and NVIDIA's Kaggle engineering writeup describes Kaggle Notebooks as no-cost GPU-access notebooks with P100 GPUs and quota management.

## Why Not Colab First

Colab Free is useful, but less reliable for our purpose.

Google's own Colab FAQ says Colab is free and can provide GPU/TPU access, but also says resources are not guaranteed, not unlimited, and usage limits fluctuate. It also says free notebooks can run for at most 12 hours depending on availability and usage patterns.

That makes Colab good for quick script checks, but riskier for 1,000-example inference runs that need stable output preservation.

## Free-GPU Target

Do not try the whole extended-validation matrix first.

Start with:

```text
SQuAD-200 -> Qwen2.5-0.5B -> GPU pipeline check
```

If successful:

```text
SQuAD-1000 -> Qwen2.5-1.5B if memory allows
```

Fallback:

```text
SQuAD-1000 -> Qwen2.5-0.5B
```

## Success Condition For The Free-GPU Dry Run

The free-GPU route is good enough to continue if:

- `scripts/check_runtime.py` reports CUDA available
- SQuAD-200 inference completes
- probe training completes
- analysis metrics are written
- output artifacts can be saved/downloaded
- SQuAD-1000 can complete without repeated disconnects

## Failure Condition For The Free-GPU Dry Run

Free GPU is not enough if:

- CUDA is unavailable
- the notebook disconnects before SQuAD-200 finishes
- model downloads repeatedly fail
- SQuAD-1000 cannot finish
- output artifacts cannot be reliably saved

If this happens, use free GPU only for script validation and move serious extended validation to paid/borrowed GPU.

## Exact First Run

The first run should use the Kaggle-ready notebook:

```text
notebooks/tcl_v0_kaggle_free_gpu_dry_run.ipynb
```

Inside that notebook, the first helper-script run is:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 200 Qwen/Qwen2.5-0.5B-Instruct
```

Then, only if stable:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 1000 Qwen/Qwen2.5-1.5B-Instruct
```

If 1.5B fails:

```bash
bash scripts/run_free_gpu_squad_dry_run.sh 1000 Qwen/Qwen2.5-0.5B-Instruct
```

## Source Notes

- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- Kaggle GPU notebook discussion/source trail:
  - https://developer.nvidia.com/blog/?p=29393
  - https://www.datacamp.com/tutorial/tutorial-kaggle-datasets-tutorials-kaggle-notebooks
