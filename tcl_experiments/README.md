# TCL Experiments

Code, question sets, run records, and analysis outputs for the TCL research program. The governing plan is [../TCL-Research-Program.md](../TCL-Research-Program.md); every experiment here maps to a hypothesis and a protocol in [../docs/](../docs/).

This is a research pipeline, not a packaged library.

## Phase 1 Pipeline (its records are reused in Phases 2 and 3)

| Step | Script | Runs on |
|---|---|---|
| Build question sets with fixed splits (TriviaQA; SQuAD split by Wikipedia article; NQ-Open) | `scripts/prepare_phase1_kaggle_data.py` | CPU |
| Inference: answer-span labels and confidence, hidden states at 25/50/100% depth, P(True), semantic entropy; resumable | `scripts/run_inference_v2.py` | GPU (Kaggle) |
| Re-analysis of the TCL-v0 records (local Phase 1: H1, H2) | `scripts/run_phase1_analysis.py` | CPU |
| Analysis of the Kaggle runs (H1-R, H2-R, baselines, layer sweep, label-audit sample) | `scripts/run_phase1_kaggle_analysis.py` | CPU |
| Correctness labeling rules (shared) | `scripts/labeling.py` | - |
| Runtime check (Python, PyTorch, CUDA) | `scripts/check_runtime.py` | any |

The GPU step is driven by [../notebooks/tcl_phase1_kaggle_runs.ipynb](../notebooks/tcl_phase1_kaggle_runs.ipynb), which pins the exact code commit for each run.

## Phase 3 Pipeline

| Step | Script | Runs on |
|---|---|---|
| Draw the PopQA sample (seed 2030, three batches of 1,500) | `scripts/prepare_phase3_popqa.py` | CPU |
| PopQA inference with the Phase 1 settings | `scripts/run_inference_v2.py`, driven by [../notebooks/tcl_phase3_popqa_kaggle.ipynb](../notebooks/tcl_phase3_popqa_kaggle.ipynb) | GPU (Kaggle) |
| Draw the PopQA label-audit sample (seed 2031) | `scripts/draw_phase3_audit.py` | CPU |
| 3a design selection (`select`, validation splits only) and 3b scoring of every arm (`score`) | `scripts/run_phase3_heads.py` | CPU |
| H4-T decision, paired bootstrap CIs, secondary metrics, refusal-excluded results | `scripts/run_phase3_analysis.py` | CPU |
| H4 (self-consistency): one-pass consistency heads, combiner with raw confidence, scores | `scripts/run_phase3_h4.py` | CPU |
| H4 consistency-label audit sample (seed 2032) | `scripts/draw_phase3_h4_audit.py` | CPU |
| H4 decision: learnability, non-redundancy, AURC gain with bootstrap CIs | `scripts/run_phase3_h4_analysis.py` | CPU |
| Grounding data: MRQA samples (SQuAD, NQ, HotpotQA, RelationExtraction) with distractor passages | `scripts/prepare_phase3_grounding.py` | CPU |
| Grounding inference (passage in the prompt, `--abstain-instruction`) and NLI grounding labels | `scripts/run_inference_v2.py`, `scripts/label_grounding.py` | GPU (Kaggle) |
| H4 grounding and correctness audit sample (seed 2035) | `scripts/draw_phase3_grounding_audit.py` | CPU |
| H4 (grounding): one-pass grounding heads, combiner, scores; decision | `scripts/run_phase3_h4_grounding.py`, `scripts/run_phase3_h4_grounding_analysis.py` | CPU |

## Phase 4a Pipeline

| Step | Script | Runs on |
|---|---|---|
| Choose the grounding labeler on the Phase 3 audit set | `scripts/select_grounding_labeler.py`, driven by [../notebooks/tcl_phase4a_labeler_kaggle.ipynb](../notebooks/tcl_phase4a_labeler_kaggle.ipynb) | GPU (Kaggle) |
| 7B inference (`--device-map auto`, float16 on two GPUs), grounding labels, Phase 3 relabel | `scripts/run_inference_v2.py`, `scripts/label_grounding.py --labeler`, driven by [../notebooks/tcl_phase4a_kaggle.ipynb](../notebooks/tcl_phase4a_kaggle.ipynb) | GPU (Kaggle) |
| The three label audits (seeds 2040-2042) | `scripts/draw_phase4a_audits.py` | CPU |
| Scoring and decisions: the Phase 3 scripts with `--runs runs/phase4a --models qwen7b` | `run_phase3_heads.py score`, `run_phase3_analysis.py`, `run_phase3_h4.py`, `run_phase3_h4_analysis.py`, `run_phase3_h4_grounding.py`, `run_phase3_h4_grounding_analysis.py` | CPU |
| Scale comparison and weak-confidence pattern (ungated) | `scripts/phase4a_ungated.py` | CPU |
| Final reporting analyses after closing the program: minimum detectable effects, corrected weak-confidence pattern, baseline choice | `scripts/final_analyses.py` | CPU |

## Phase 2 and 2b Pipeline

| Step | Script | Runs on |
|---|---|---|
| Labeling rule v3 and its tests and validation | `scripts/labeling.py` (`answer_match_v3`), `scripts/test_labeling_v3.py`, `scripts/validate_labeling_v3.py` | CPU |
| Train one arm (A-D) for one seed: trust head, LoRA, LM loss (`--lm-target reference` for Phase 2, `reference_eot` for 2b), scoring of fixed and regenerated answers, perplexity | `scripts/train_trust_head.py` | GPU (Kaggle) |
| Run one part of a training dataset's trainings, two at a time on two GPUs, with lambda selection | `scripts/run_phase2_part.py` | GPU (Kaggle) |
| Decisions (H3-T, H3-J), paired bootstrap CIs, generation bar, reference baselines, label-audit sample | `scripts/run_phase2_analysis.py` | CPU |

The GPU steps are driven by [../notebooks/tcl_phase2_kaggle_runs.ipynb](../notebooks/tcl_phase2_kaggle_runs.ipynb).

## Data

| Path | Contents |
|---|---|
| `data/phase3_grounding/` | Grounding question sets from MRQA with distractor passages (seeds 2033, 2034) and metadata |
| `data/phase3/` | The Phase 3 PopQA sample (4,500 test questions in three batches) and its metadata |
| `data/phase2/` | The fixed WikiText-2 perplexity sample (200 paragraphs) |
| `data/phase1_kaggle/` | Phase 1 question sets: `triviaqa.csv` (4,500), `squad.csv` (4,500, article-disjoint splits), `nq_open.csv` (3,610), each with a `split` column and a metadata file |
| `data/benchmarks/` | TCL-v0 subsets (100-1,000 questions) with split and manual-review files |
| `data/seed_questions.csv`, `data/diagnostic_questions_100.csv` | Small smoke-test sets |

## Run Records

| Path | Contents |
|---|---|
| `runs/phase1_kaggle/<model>/<dataset>/` | Phase 1 inference outputs: `records.jsonl` and `config.json` (committed), `hidden_chunk_*.npz` (local only, excluded by `.gitignore`) |
| `runs/phase1_kaggle_analysis/` | Phase 1 decisions, metrics, test predictions, layer sweep, and label audit |
| `runs/phase3a/` | Phase 3a design selection: every setting's validation AUROC, the chosen setting, run log |
| `runs/phase3_popqa/<model>/popqa/` | PopQA inference outputs (records and config committed; hidden states local only) |
| `runs/phase3_audit/` | PopQA label-audit samples and reviewed labels (100 per model) |
| `runs/phase3b/`, `runs/phase3b_analysis/` | Phase 3b test scores per arm and seed, manifest with label counts, H4-T decision, details with every CI, run logs |
| `runs/phase3_h4_audit/`, `runs/phase3_h4/`, `runs/phase3_h4_analysis/` | H4 self-consistency: label audit with my hand counts, test scores per arm and seed, decision and details, run logs |
| `runs/phase3_grounding/`, `runs/phase3_h4g_audit/`, `runs/phase3_h4g/`, `runs/phase3_h4g_analysis/` | H4 grounding: answers and NLI labels per model and dataset (hidden states kept local), label audit with my labels, test scores per arm and seed, decision and details, run logs |
| `runs/phase4a/` | Phase 4a 7B answers and labels (`phase1_kaggle/`, `phase3_popqa/`, `phase3_grounding/` under `qwen7b/`; hidden states local only), the Phase 3 answers relabeled (`phase3_relabel/`), and the labeler selection |
| `runs/phase4a_audit/` | The three Phase 4a label audits with my labels |
| `runs/phase4a_h4t*/`, `runs/phase4a_h4c*/`, `runs/phase4a_h4g*/` | Phase 4a scores, decisions, and details for H4-T, self-consistency, and grounding, with run logs |
| `runs/phase4a_relabel_h4g*/`, `runs/phase4a_ungated/` | Ungated: the Phase 3 grounding analysis with the fixed label; scale comparison and weak-confidence pattern |
| `runs/final_analyses/` | Minimum detectable effects (`mde.csv`), the corrected weak-confidence pattern, and the baseline-choice table |
| `runs/phase2/`, `runs/phase2b/` | Phase 2 and 2b trainings: per arm and seed, test scores on fixed answers, regenerated answers, metrics (with optimizer step counts), training log |
| `runs/phase2_analysis/`, `runs/phase2b_analysis/` | Phase 2 and 2b decisions, details with every CI, per-arm metrics, run log, and the Phase 2 label audit |
| `runs/phase2b_audit/` | The Phase 2b label audit (100 regenerated answers) |
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
