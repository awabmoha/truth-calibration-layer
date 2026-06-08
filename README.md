# Truth Calibration Layer (TCL)

Truth Calibration Layer is an independent research project exploring whether language models can become more confidence-aware by adding a calibration layer around model behavior.

Current status: **theory-stage framework plus early TCL-v0 diagnostics**.

> **Scope warning:** despite the name, TCL-v0 is not a general truth detector. In this repository, "truth calibration" currently means estimating whether a generated answer is likely correct in bounded QA benchmark settings. TCL-v0 does not verify factuality in open-ended generation, solve hallucination, or make deployment-ready reliability decisions.

This repository should not be read as a completed implementation of full TCL. The main paper proposes TCL as a theoretical framework and research hypothesis. The practical work here is an early prototype direction called TCL-v0:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

## Research Goal

Modern LLMs often produce fluent answers without reliable self-awareness of whether those answers are correct. TCL asks whether model-internal signals, especially hidden states, can support better confidence calibration than raw generation confidence alone.

The immediate practical question is narrow:

```text
Can frozen LLM hidden states predict answer correctness better than raw softmax/generation confidence?
```

## Repository Contents

- `TCL-Theory-Paper-Theoretical-Framework.docx` - final theory-stage paper.
- `TCL-v0-experiment-plan.md` - practical plan for the first TCL-v0 experiments.
- `TCL-v0-Extended-Validation-Plan.md` - predeclared gate for the next stronger TCL-v0 validation stage.
- `TCL-v0-Free-GPU-Decision.md` - recommendation for using free GPU resources before paid cloud.
- `TCL-v0-Free-GPU-Runbook.md` - Colab/Kaggle execution guide for free-GPU dry runs.
- `notebooks/` - Kaggle/Colab notebook helpers for free-GPU dry runs.
- `TCL-v0-results-summary.md` - current diagnostic results and claim boundaries.
- `TCL-v0-evidence-report-v1.md` - consolidated preliminary evidence report.
- `TCL-v0-evidence-report-v2.md` - current reviewed extended-validation evidence report.
- `TCL-v0-Baseline-Checkpoint.md` - raw-only calibration baseline checkpoint for the reviewed six-run evidence.
- `TCL-v0-to-Full-TCL-Gap.md` - explicit mapping from TCL-v0 evidence to the unvalidated full TCL architecture.
- `TCL-v0-Ablation-Plan.md` - predeclared next experiment for baselines, hidden-layer/pooling ablations, and raw-vs-hidden comparisons.
- `TCL-v0-Roadmap.md` - next-step roadmap covering baselines, ablations, broader tasks, API cleanup, and risk controls.
- `TCL-v0-research-writeup.md` - short TCL-v0 method/results writeup.
- `TCL-v0-research-writeup.docx` - generated DOCX version of the TCL-v0 writeup.
- `TCL-v0-research-writeup.pdf` - standalone PDF version of the TCL-v0 writeup.
- `TCL-v0-Kaggle-Phi-Run-Log.md` - Kaggle Phi-3.5 Mini run log and reviewed results.
- `TCL-v0-Kaggle-Gemma-Run-Log.md` - Kaggle Gemma 2B-it run log and reviewed results.
- `RELEASE_NOTES.md` - current-state note for private sharing or public-readiness review.
- `requirements.txt` - root convenience file that installs `tcl_experiments/requirements.txt`.
- `tcl_experiments/` - scripts, datasets, benchmark prep, and diagnostic run records.

Older draft papers and the original one-off document builder were removed from the cleaned repository. The tracked package now keeps one canonical theory paper and one TCL-v0 empirical companion writeup.

## Current TCL-v0 Design

TCL-v0 records:

- question
- accepted/correct answer
- model answer
- correctness label
- raw generation confidence
- hidden-state vector
- TCL-v0 probe confidence

Then it compares raw confidence against TCL-v0 confidence using:

- ECE
- MCE
- Brier score
- reliability bins
- accuracy
- high-confidence error counts

## Current Diagnostic Findings

These are **early diagnostic findings**, not broad validation.

The current best variant is conservative TCL-v0:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule was added because a plain hidden-state probe improved some average metrics but could become dangerously overconfident on fluent wrong answers. The conservative version prevents TCL-v0 from raising confidence above the model's raw generation confidence.

The strongest current local result is SQuAD-500:

- models: `Qwen/Qwen2.5-0.5B-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct`
- dataset: SQuAD `plain_text`, validation subset with context included
- split: 325 train, 75 validation, 100 test
- hidden-state method: `answer_mean`
- correctness method: `strict_answer_segment_match_v2`

On SQuAD-500, Qwen showed mixed but useful TCL-v0 gains in Brier score, threshold accuracy, AUC, and high-confidence error counts, while raw generation confidence still had better ECE. SmolLM2 showed stronger calibration gains under conservative TCL-v0 across ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts.

TriviaQA 500 also supports the same direction:

- models: `Qwen/Qwen2.5-0.5B-Instruct` and `HuggingFaceTB/SmolLM2-360M-Instruct`
- dataset: TriviaQA `rc.nocontext`, validation subset
- split: 325 train, 75 validation, 100 test
- hidden-state method: `answer_mean`
- correctness method for those runs: `strict_answer_segment_match_v1`

Across both small CPU-runnable models on TriviaQA, conservative TCL-v0 improved calibration metrics over raw generation confidence while producing zero wrong held-out test examples with confidence >= 0.8 under the stricter label rule.

TCL-v0 has also been tested on NQ-Open and SQuAD. NQ-Open is much sparser under the current short-answer setup. SQuAD with context is a cleaner next benchmark because the small local models produce enough correct and incorrect answers for probe diagnostics.

The benchmark protocol has since been improved to preserve raw outputs separately from cleaned answers and to use chat-template prompting when available.

A clean 100-example NQ-Open protocol check confirmed the improved pipeline works, but the held-out test split remained sparse with only one correct answer per model.

Current cautious interpretation:

```text
Frozen hidden states appear to contain useful calibration signal, but that signal must be constrained and tested more broadly.
```

## Current Extended-Validation Checkpoint

As of June 7, 2026, TCL-v0 has a reviewed free-GPU validation checkpoint across:

- models: `Qwen/Qwen2.5-0.5B-Instruct`, `microsoft/Phi-3.5-mini-instruct`, and Kaggle-hosted Gemma 2B-it
- benchmarks: SQuAD validation with context and TriviaQA `rc.nocontext`
- scale: 1,000 examples per run, with 200 held-out test examples
- manual review: targeted review completed for all six 1,000-example runs

The original predeclared two-model stopping gate used Qwen and Phi:

```text
decision: mixed_continue_cautiously
stopping-rule choice: continue TCL-v0 research
```

The later Gemma exploratory extension strengthened the pattern:

```text
decision: supports_continuing_tcl_v0
```

The strongest repeated finding is not that TCL-v0 wins every calibration metric in every setting. The clearest result is that conservative TCL-v0 repeatedly reduces high-confidence wrong answers, especially on open-domain TriviaQA and the Gemma runs.

After adding raw-only calibration baselines, the interpretation is stricter: raw-only post-hoc calibration explains part of the high-confidence-error reduction. Hidden-state conservative TCL-v0 remains competitive and often improves AUC/ranking, but future claims must compare against temperature scaling, Platt/logistic raw calibration, isotonic regression, and raw-only conservative fusion.

Current best-position statement:

```text
TCL remains a theory-stage framework, but TCL-v0 now has reviewed preliminary evidence across multiple models and benchmarks that frozen hidden-state probes can reduce dangerous overconfidence in answer correctness estimates.
```

## Claim Boundaries

What this project can currently claim:

- TCL is a theoretical framework and research hypothesis.
- TCL-v0 is an early confidence-only diagnostic prototype around frozen models.
- Reviewed TCL-v0 diagnostics suggest hidden states contain useful calibration signal under tested QA settings.
- TCL-v0 provides empirical motivation for a future joint token-and-trust training objective.
- Conservative TCL-v0 is the strongest current diagnostic variant in this package.
- The most consistent practical gain is reduction of high-confidence wrong-answer counts.

What this project does **not** claim:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 is a general factuality, hallucination, reasoning, safety, or value-judgment verifier.
- TCL-v0 generalizes beyond the tested model and dataset.
- The full four-dimensional TCL trust vector has been implemented.
- TCL-v0 is ready for deployment or user-facing reliability decisions.

## Distance From Full TCL

TCL-v0 is not on its own a direct validation of the full TCL architecture. It tests one supporting assumption:

```text
frozen hidden states contain answer-correctness calibration signal
```

That assumption is necessary for the broader theory to be plausible, but it is not sufficient to validate full TCL.

Full TCL would require at least:

- a model trained with a joint objective for next-token prediction and trust-score prediction
- an integrated trust head or architectural module, not only an external post-hoc probe
- multiple trust dimensions beyond answer correctness, such as reasoning validity, grounding/provenance, and epistemic uncertainty
- comparisons against raw confidence, raw-only post-hoc calibration, and frozen-probe baselines

So the current best interpretation is:

```text
TCL-v0 provides empirical motivation for TCL, not architectural validation of TCL.
```

## Main Current Criticisms

The strongest critique of the current checkpoint is not that the idea failed. It is that the evidence is still narrow:

- the current practical evidence is mostly short-answer QA correctness calibration
- stronger calibration baselines are still missing, including temperature scaling, Platt scaling, isotonic regression, and simple learned raw-plus-probe combinations
- task diversity is limited; TCL-v0 has not yet been tested on code, multi-step reasoning, long-form factuality, adversarial prompts, or subjective/safety responses
- the conservative rule reduces overconfidence but cannot raise confidence when the base model is underconfident
- the repository is still closer to a research log than a reusable Python package
- the project name can over-promise unless the scope warning is kept visible

## Local Experiment Notes

The local machine used for these diagnostics is CPU-only and suited for small models.

Observed practical setup:

- CPU: AMD Ryzen 7 5825U
- RAM: about 6 GB available during testing
- GPU: integrated AMD Radeon, no CUDA
- local model used: `Qwen/Qwen2.5-0.5B-Instruct`

For larger experiments, a stronger GPU environment may be needed.

## Reproducing the Prototype

From the experiment folder:

```powershell
cd tcl_experiments
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

From the repository root, the equivalent dependency entry point is:

```powershell
pip install -r requirements.txt
```

The root `requirements.txt` delegates to `tcl_experiments/requirements.txt`, where the experiment dependencies are maintained.

The current scripts are in `tcl_experiments/scripts/`, including reproducible builders for the TCL-v0 writeup DOCX/PDF artifacts.

Important: do not treat a successful run as full TCL validation. Each run should be documented as a TCL-v0 diagnostic only.

## Next Step

Follow `TCL-v0-Ablation-Plan.md` before running any new scale experiment. The next technical checkpoint should test whether hidden-state TCL-v0 still adds value beyond raw-only calibration under layer, pooling, and feature ablations.

## Author

Independent research project by Awab Mohamed.
