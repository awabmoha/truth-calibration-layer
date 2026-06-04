# Current State: TCL and TCL-v0

Status: private/shareable research snapshot, not full TCL validation

This repository contains two connected pieces of work:

- Truth Calibration Layer (TCL), a theory-stage framework for confidence-aware language models.
- TCL-v0, an early empirical diagnostic that tests one narrow part of the idea.

The full TCL framework has not been implemented or validated. TCL-v0 is a confidence-only probe experiment:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

## What Is Ready

The repository has been cleaned into one canonical package:

- `TCL-Theory-Paper-Theoretical-Framework.docx` is the current theory paper.
- `TCL-v0-research-writeup.md`, `.docx`, and `.pdf` are the empirical companion note.
- `TCL-v0-evidence-report-v1.md` summarizes the current evidence.
- `TCL-v0-results-summary.md` keeps the longer diagnostic history.
- `tcl_experiments/` contains scripts, benchmark subsets, run records, and reproducible report builders.

Old duplicate theory drafts, the one-off JS document builder, local virtual environments, logs, and temporary files were removed from the cleaned repository.

## Current Empirical Signal

TCL-v0 has preliminary evidence that frozen hidden states can improve answer-confidence calibration under tested conditions.

The current strongest local result is SQuAD-500 using two small CPU-runnable instruction models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

The strongest practical variant is conservative TCL-v0:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule uses the hidden-state probe as a confidence-lowering signal, while preventing it from raising confidence above raw generation confidence.

On SQuAD-500:

- Qwen improved Brier score, threshold accuracy, AUC, and high-confidence error counts under conservative TCL-v0, while raw generation confidence still had better ECE.
- SmolLM2 improved ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts under conservative TCL-v0.

TriviaQA-500 supports the same general direction across the same two models. NQ-Open was useful as a stress test but too sparse on the local CPU setup for strong conclusions.

## What This Does Not Claim

This snapshot does not claim that:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes across all models, datasets, or tasks.
- The full four-dimensional TCL trust vector has been implemented.
- The current results are publication-grade without broader manual review and stronger compute.

## Why This Is Still Useful

The work has moved past pure theory in one narrow but meaningful way: TCL-v0 creates a reproducible bridge from the theory paper to an empirical question.

The current evidence supports continuing the research direction, especially the hypothesis that hidden states may contain useful correctness-calibration signal. It does not yet support strong claims about general truthfulness or full-system reliability.

## Recommended Next Step

The next serious experiment should be planned before running:

- use larger instruction models if compute allows
- run larger benchmark subsets
- include one context-grounded QA benchmark and one open-domain QA benchmark
- predeclare train/validation/test splits
- predeclare manual-review rules
- compare raw, probe, calibrated, and conservative confidence scores
- report ECE, MCE, Brier score, reliability bins, accuracy, AUC, and high-confidence wrong-answer counts

Until that follow-up is complete, the safest public framing is:

```text
TCL is a theory-stage framework. TCL-v0 provides preliminary evidence that frozen hidden states can support confidence calibration under tested conditions, but it does not validate full TCL.
```
