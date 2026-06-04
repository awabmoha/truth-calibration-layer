# Truth Calibration Layer (TCL)

Truth Calibration Layer is an independent research project exploring whether language models can become more confidence-aware by adding a calibration layer around model behavior.

Current status: **theory-stage framework plus early TCL-v0 diagnostics**.

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
- `TCL-v0-results-summary.md` - current diagnostic results and claim boundaries.
- `tcl_experiments/` - scripts, datasets, benchmark prep, and diagnostic run records.
- `build_tcl_doc.js` - document build script used for the paper package.
- Older paper versions are kept for traceability.

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

On the current 500-example TriviaQA diagnostic:

- model: `Qwen/Qwen2.5-0.5B-Instruct`
- dataset: TriviaQA `rc.nocontext`, validation subset
- split: 325 train, 75 validation, 100 test
- hidden-state method: `answer_mean`
- extended manual review: all 26 automatic positive test labels reviewed, 3 false positives corrected

Using reviewed held-out labels, conservative TCL-v0 improved ECE, Brier score, MCE, and 0.5-threshold accuracy over raw generation confidence, while producing zero wrong test examples with confidence >= 0.8.

Current cautious interpretation:

```text
Frozen hidden states appear to contain useful calibration signal, but that signal must be constrained and tested more broadly.
```

## Claim Boundaries

What this project can currently claim:

- TCL is a theoretical framework and research hypothesis.
- TCL-v0 is an early confidence-only diagnostic prototype.
- Initial diagnostics suggest hidden states may contain useful calibration signal.
- Conservative TCL-v0 is the strongest current diagnostic variant in this package.

What this project does **not** claim:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes beyond the tested model and dataset.
- The full four-dimensional TCL trust vector has been implemented.

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

The current scripts are in `tcl_experiments/scripts/`.

Important: do not treat a successful run as full TCL validation. Each run should be documented as a TCL-v0 diagnostic only.

## Next Step

The next recommended step is testing whether the conservative TCL-v0 pattern holds on a second model or a second benchmark source.

## Author

Independent research project by Awab Mohamed.
