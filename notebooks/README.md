# Notebooks

Kaggle notebooks for every GPU step of the program. My laptop has no GPU, so model inference and training run here; analysis runs locally.

## Current Notebook

`tcl_phase4a_kaggle.ipynb` is the latest notebook (Phase 4a, Qwen2.5-7B in float16 on GPU T4 x2): set `TASKS` (any of `pool`, `popqa`, `grounding`, `relabel`) and `TAG`; `LIMIT = 50` with `ONLY_SPLITS = "test"` is the smoke test. `tcl_phase3_popqa_kaggle.ipynb` ran Phase 3 PopQA inference. Set `MODEL_TAG` to `qwen` or `phi`; `N_QUESTIONS = 1500` runs batch 1, and 3000 or 4500 add batches only if the size rule needs them. `tcl_phase2_kaggle_runs.ipynb` covers Phases 2 and 2b (both complete). Set `TRAIN_DATASET` to `triviaqa` or `squad` and `PART` to `"1"`, `"2"`, `"3"`, or a list such as `"2,3"` (parts 2 and 3 need `LAMBDA` from part 1). `LM_TARGET = "reference"` reproduces Phase 2; `"reference_eot"` with `ARMS = "C,D"` reproduces Phase 2b. Run on Kaggle with GPU T4 x2 and Internet on, and download the zip each session writes to `/kaggle/working/`. `tcl_phase1_kaggle_runs.ipynb` produced the Phase 1 records; the other notebooks belong to the archived TCL-v0 work.

## Current Recommendation

Kaggle is the recommended free-GPU environment. Colab Free remains a fallback for short runtime checks.

## Notebook Index

- `tcl_v0_kaggle_free_gpu_dry_run.ipynb` - original free-GPU execution check, centered on Qwen SQuAD/TriviaQA dry runs.
- `tcl_v0_kaggle_reviewed_model_runs.ipynb` - reusable reviewed-run notebook for Qwen, Phi-3.5 Mini, and Kaggle-hosted Gemma 2B-it.
- `tcl_v0_kaggle_ablation_smoke.ipynb` - next-checkpoint ablation smoke notebook for layer, pooling, and raw-only baseline comparisons.
- `tcl_phase4a_kaggle.ipynb` - Phase 4a runs on Qwen2.5-7B (float16 across two T4s): the Phase 1 pool with semantic entropy on test, PopQA, the grounding sets with the chosen labeler, and the Phase 3 grounding relabel. Protocol: `docs/phase4/TCL-Phase4a-Protocol.md`.
- `tcl_phase4a_labeler_kaggle.ipynb` - Phase 4a grounding-labeler selection on the 100 Phase 3 audit answers (no 7B answer used).
- `tcl_phase3_grounding_kaggle.ipynb` - Phase 3 grounding runs: inference with the passage in the prompt and the abstention line, then NLI grounding labels, for one model per session (datasets can be split across sessions). Protocol: `docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md`.
- `tcl_phase3_popqa_kaggle.ipynb` - Phase 3 PopQA inference (same settings as Phase 1) for one model per session, pinned to the commit that drew the PopQA sample. Protocol: `docs/phase3/TCL-Phase3-Protocol.md`.
- `tcl_phase2_kaggle_runs.ipynb` - Phase 2 trust-head training (arms A-D, lambda selection, three seeds), one part of one training dataset per session, two trainings at a time on the two GPUs. Protocol: `docs/phase2/TCL-Phase2-Protocol.md`.
- `tcl_phase1_kaggle_runs.ipynb` - Phase 1 Kaggle runs (inference v2: answer-span confidence, layer sweep features, P(True), semantic entropy) for one model per session. Protocol: `docs/phase1/TCL-Phase1-Kaggle-Protocol.md`.

These notebooks are execution protocols, not completed validation results. Results should be interpreted only after artifact verification, targeted manual review where required, and comparison against the relevant predeclared plan.
