# Notebooks

This folder contains notebook protocols for running TCL-v0 validation checks outside the local CPU-only machine.

## Current Recommendation

Kaggle is the recommended free-GPU environment. Colab Free remains a fallback for short runtime checks.

## Notebook Index

- `tcl_v0_kaggle_free_gpu_dry_run.ipynb` - original free-GPU execution check, centered on Qwen SQuAD/TriviaQA dry runs.
- `tcl_v0_kaggle_reviewed_model_runs.ipynb` - reusable reviewed-run notebook for Qwen, Phi-3.5 Mini, and Kaggle-hosted Gemma 2B-it.
- `tcl_v0_kaggle_ablation_smoke.ipynb` - next-checkpoint ablation smoke notebook for layer, pooling, and raw-only baseline comparisons.

These notebooks are execution protocols, not completed validation results. Results should be interpreted only after artifact verification, targeted manual review where required, and comparison against the relevant predeclared plan.
