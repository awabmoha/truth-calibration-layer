# Current State: TCL and TCL-v0

Status: shareable research checkpoint, not full TCL validation

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
- `TCL-v0-evidence-report-v2.md` summarizes the reviewed extended-validation checkpoint.
- `TCL-v0-Roadmap.md` records the next-step roadmap for baselines, ablations, generalization, API cleanup, and scope control.
- `TCL-v0-Reviewed-AnswerMean-Ablation-Checkpoint.md` records the local answer-mean ablation against raw-only calibration and feature-fusion baselines.
- `TCL-v0-results-summary.md` keeps the longer diagnostic history.
- `tcl_experiments/` contains scripts, benchmark subsets, run records, and reproducible report builders.

Old duplicate theory drafts, the one-off JS document builder, local virtual environments, logs, and temporary files were removed from the cleaned repository.

## Current Empirical Signal

TCL-v0 has reviewed preliminary evidence that frozen hidden states can improve answer-confidence calibration under tested QA conditions.

As of the June 7, 2026 checkpoint, the reviewed extended-validation set includes:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- Kaggle-hosted Gemma 2B-it
- SQuAD validation with context
- TriviaQA `rc.nocontext`
- six 1,000-example runs, with 200 held-out test examples per run
- targeted manual review completed for all six runs

The strongest practical variant is conservative TCL-v0:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule uses the hidden-state probe as a confidence-lowering signal, while preventing it from raising confidence above raw generation confidence.

The most consistent practical signal is not that TCL-v0 wins every calibration metric. The clearest repeated finding is that conservative TCL-v0 reduces high-confidence wrong-answer counts, especially on TriviaQA and the Gemma exploratory runs.

The formal Qwen/Phi gate decision was `mixed_continue_cautiously`. The exploratory Qwen/Phi/Gemma decision was `supports_continuing_tcl_v0`.

The answer-mean ablation checkpoint has now been run locally on the existing reviewed records. In the weighted comparison, hidden conservative TCL-v0 beat the best raw-only baseline on Brier score in 5 of 6 runs, ECE in 4 of 6 runs, and AUC in 5 of 6 runs. It did not beat raw-only calibration on high-confidence wrong-answer count, because raw-only calibrators can suppress confidence directly.

## What This Does Not Claim

This snapshot does not claim that:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 is a general truth detector or factuality verifier.
- TCL-v0 generalizes across all models, datasets, or tasks.
- The full four-dimensional TCL trust vector has been implemented.
- The current results are publication-grade without stronger baselines, ablations, and broader tasks.

## Why This Is Still Useful

The work has moved past pure theory in one narrow but meaningful way: TCL-v0 creates a reproducible bridge from the theory paper to an empirical question.

The current evidence supports continuing the research direction, especially the hypothesis that hidden states may contain useful correctness-calibration signal. It does not yet support strong claims about general truthfulness or full-system reliability.

## Recommended Next Step

The next serious experiment is the remaining GPU ablation:

- run ablations for hidden layer and hidden-state pooling on Kaggle GPU
- compare raw-only, hidden-only, raw-plus-hidden, learned fusion, and conservative fusion on the newly extracted records
- test at least one cross-dataset or prompt-shift setting
- keep manual-review rules predeclared
- report ECE, MCE, Brier score, reliability bins, accuracy, AUC, and high-confidence wrong-answer counts

Until that follow-up is complete, the safest public framing is:

```text
TCL is a theory-stage framework. TCL-v0 provides reviewed preliminary evidence that frozen hidden states can support confidence calibration under tested QA conditions, but it does not validate full TCL.
```
