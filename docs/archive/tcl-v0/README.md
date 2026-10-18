# TCL-v0 Archive (June 2026)

These are the records of TCL-v0, the first empirical diagnostic of TCL: a logistic probe on frozen final-layer hidden states, tested on Qwen, Phi, and Gemma with 1,000 questions per run.

They are kept for history. Their conclusions were re-tested in Phase 1 ([../../phase1/](../../phase1/)), which found that:
- the 200-question test sets were too small to detect the effects in question,
- the SQuAD subset came from only 2 Wikipedia articles, so train and test questions shared passages,
- the headline metric (high-confidence wrong answers under `min(raw, probe)`) could also be achieved by recalibrating raw confidence alone,
- the June 9 ablation checkpoint scored against automatic rather than reviewed labels.

| Group | Files |
|---|---|
| Plans | `TCL-v0-experiment-plan.md`, `TCL-v0-Extended-Validation-Plan.md`, `TCL-v0-Ablation-Plan.md`, `TCL-v0-Roadmap.md`, `TCL-v0-to-Full-TCL-Gap.md` |
| Results | `TCL-v0-results-summary.md`, `TCL-v0-evidence-report-v1.md`, `TCL-v0-evidence-report-v2.md`, `TCL-v0-Baseline-Checkpoint.md`, `TCL-v0-Reviewed-AnswerMean-Ablation-Checkpoint.md` |
| Compute and runs | `TCL-v0-Free-GPU-Decision.md`, `TCL-v0-Free-GPU-Runbook.md`, `TCL-v0-Kaggle-Ablation-Runbook.md`, `TCL-v0-Kaggle-Phi-Run-Log.md`, `TCL-v0-Kaggle-Gemma-Run-Log.md`, `TCL-v0-Extended-Validation-Run-Tracker.csv` |
| Repository state | `RELEASE_NOTES.md` (v0.1.x), `readme-high-confidence-errors.svg` (former README figure) |
