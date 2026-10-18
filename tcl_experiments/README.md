# TCL Experiments

Code, question sets, run records, and analysis outputs for the TCL research program. The governing plan is [../TCL-Research-Program.md](../TCL-Research-Program.md); every experiment here maps to a hypothesis and a protocol in [../docs/](../docs/).

This is a research pipeline, not a packaged library.

## Current Pipeline (Phase 1, used again in Phase 2)

| Step | Script | Runs on |
|---|---|---|
| Build question sets with fixed splits (TriviaQA; SQuAD split by Wikipedia article; NQ-Open) | `scripts/prepare_phase1_kaggle_data.py` | CPU |
| Inference: answer-span labels and confidence, hidden states at 25/50/100% depth, P(True), semantic entropy; resumable | `scripts/run_inference_v2.py` | GPU (Kaggle) |
| Re-analysis of the TCL-v0 records (local Phase 1: H1, H2) | `scripts/run_phase1_analysis.py` | CPU |
| Analysis of the Kaggle runs (H1-R, H2-R, baselines, layer sweep, label-audit sample) | `scripts/run_phase1_kaggle_analysis.py` | CPU |
| Correctness labeling rules (shared) | `scripts/labeling.py` | - |
| Runtime check (Python, PyTorch, CUDA) | `scripts/check_runtime.py` | any |

The GPU step is driven by [../notebooks/tcl_phase1_kaggle_runs.ipynb](../notebooks/tcl_phase1_kaggle_runs.ipynb), which pins the exact code commit for each run.

## Data

| Path | Contents |
|---|---|
| `data/phase1_kaggle/` | Phase 1 question sets: `triviaqa.csv` (4,500), `squad.csv` (4,500, article-disjoint splits), `nq_open.csv` (3,610), each with a `split` column and a metadata file |
| `data/benchmarks/` | TCL-v0 subsets (100-1,000 questions) with split and manual-review files |
| `data/seed_questions.csv`, `data/diagnostic_questions_100.csv` | Small smoke-test sets |

## Run Records

| Path | Contents |
|---|---|
| `runs/phase1_kaggle/<model>/<dataset>/` | Phase 1 inference outputs: `records.jsonl` and `config.json` (committed), `hidden_chunk_*.npz` (local only, excluded by `.gitignore`) |
| `runs/phase1_kaggle_analysis/` | Phase 1 decisions, metrics, test predictions, layer sweep, and label audit |
| `runs/phase1_local/` | Local Phase 1 re-analysis of the TCL-v0 records |
| `runs/freegpu-*`, `runs/benchmark-*`, `runs/post_cloud_decision_*`, `runs/ablation_*` | TCL-v0 runs and checkpoints (June 2026) |
| `runs/phase1_kaggle_download/` | Raw Kaggle zips and logs (local only, excluded by `.gitignore`) |

## Reproduce Phase 1

From the repository root:

```bash
pip install -r requirements.txt
python tcl_experiments/scripts/prepare_phase1_kaggle_data.py
```

Run the Kaggle notebook once per model, unzip each output into `tcl_experiments/runs/phase1_kaggle/<model>/`, then:

```bash
python tcl_experiments/scripts/run_phase1_kaggle_analysis.py
```

To check the inference script without a GPU, use tiny models on a few questions:

```bash
python tcl_experiments/scripts/run_inference_v2.py --model HuggingFaceTB/SmolLM2-135M-Instruct --model-tag smol --dataset triviaqa --questions tcl_experiments/data/phase1_kaggle/triviaqa.csv --out-dir tcl_experiments/runs/smoke --limit 5 --only-splits test --nli-model cross-encoder/nli-deberta-v3-xsmall --se-samples 4 --device cpu
```

## TCL-v0 Pipeline (Archived)

The June 2026 TCL-v0 pipeline is kept so its runs remain reproducible. Its conclusions are superseded by Phase 1.

| Group | Scripts |
|---|---|
| Inference and probe | `run_inference.py`, `train_probe.py`, `metrics.py` |
| Question sets | `prepare_triviaqa_subset.py`, `prepare_squad_subset.py`, `prepare_nq_open_subset.py` |
| Manual review | `make_review_csv.py`, `make_targeted_review_csv.py`, `assist_targeted_review.py`, `validate_targeted_review.py`, `recompute_reviewed_metrics.py`, `relabel_records.py` |
| Baselines and ablations | `run_raw_calibration_baselines.py`, `run_ablation_analysis.py` |
| Cloud artifacts | `package_run_artifact.py`, `import_run_artifact.py`, `verify_run_artifact.py`, `run_post_cloud_pipeline.py`, `validate_run_tracker.py` |
| Decision notes | `summarize_extended_validation.py`, `write_extended_validation_decision_note.py` |
| GPU shell runners | `run_free_gpu_squad_dry_run.sh`, `run_free_gpu_triviaqa_dry_run.sh`, `run_free_gpu_ablation_smoke.sh` |
| Paper builds | `build_tcl_v0_writeup_docx.py`, `build_tcl_v0_writeup_pdf.py` |

The `TCL-v0-*.md` summaries in this folder are TCL-v0 benchmark notes from June 2026, also archived.
