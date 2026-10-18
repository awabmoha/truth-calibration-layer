# Notebooks

This folder contains notebook protocols for running TCL-v0 validation checks outside the local CPU-only machine.

## Current Notebook

`tcl_phase2_kaggle_runs.ipynb` is the current notebook. Set `TRAIN_DATASET` to `triviaqa` or `squad` (one session each, which can run in parallel), run on Kaggle with GPU T4 x2 and Internet on, and download the zip it writes to `/kaggle/working/`. `tcl_phase1_kaggle_runs.ipynb` produced the Phase 1 records; the other notebooks belong to the archived TCL-v0 work.

## Current Recommendation

Kaggle is the recommended free-GPU environment. Colab Free remains a fallback for short runtime checks.

## Notebook Index

- `tcl_v0_kaggle_free_gpu_dry_run.ipynb` - original free-GPU execution check, centered on Qwen SQuAD/TriviaQA dry runs.
- `tcl_v0_kaggle_reviewed_model_runs.ipynb` - reusable reviewed-run notebook for Qwen, Phi-3.5 Mini, and Kaggle-hosted Gemma 2B-it.
- `tcl_v0_kaggle_ablation_smoke.ipynb` - next-checkpoint ablation smoke notebook for layer, pooling, and raw-only baseline comparisons.
- `tcl_phase2_kaggle_runs.ipynb` - Phase 2 trust-head training (arms A-D, lambda selection, three seeds) for one training dataset per session. Protocol: `docs/phase2/TCL-Phase2-Protocol.md`.
- `tcl_phase1_kaggle_runs.ipynb` - Phase 1 Kaggle runs (inference v2: answer-span confidence, layer sweep features, P(True), semantic entropy) for one model per session. Protocol: `docs/phase1/TCL-Phase1-Kaggle-Protocol.md`.

These notebooks are execution protocols, not completed validation results. Results should be interpreted only after artifact verification, targeted manual review where required, and comparison against the relevant predeclared plan.
