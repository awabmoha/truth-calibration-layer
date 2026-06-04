# TCL-v0 Evidence Report v1

Status: preliminary evidence report, not full TCL validation

## Research Question

TCL-v0 asks one narrow question:

```text
Can frozen LLM hidden states predict answer correctness better than raw generation confidence?
```

This is not the full Truth Calibration Layer. Full TCL remains a theoretical framework and research hypothesis. TCL-v0 only tests whether hidden-state vectors contain useful correctness-calibration signal.

## Practical Target

The current target is not to prove that TCL makes models truthful. The target is to reach a cautious, evidence-backed statement:

```text
Across small open-source LLMs and multiple QA benchmarks, frozen hidden states show preliminary evidence of containing useful correctness-calibration signal, especially when combined conservatively with raw generation confidence.
```

## Method

TCL-v0 records:

- question
- context when available
- accepted answers
- model answer
- correctness label
- raw generation confidence
- final-layer hidden-state vector
- probe confidence from a small logistic-regression probe

Current default:

- hidden-state extraction: `answer_mean`
- probe: logistic regression
- safer combined score: `min(raw_generation_confidence, tcl_v0_probe_confidence)`
- calibration metrics: ECE, MCE, Brier score, reliability bins, accuracy, AUC, high-confidence wrong counts

## Evidence So Far

### TriviaQA

TriviaQA was the first useful benchmark. Under strict reviewed labels, conservative TCL-v0 improved calibration over raw confidence for both small CPU-runnable models:

- Qwen 500: conservative TCL-v0 improved ECE, Brier score, and accuracy, with zero wrong test examples at confidence >= 0.8.
- SmolLM2 500: conservative TCL-v0 strongly improved ECE, Brier score, accuracy, and high-confidence error counts.

Interpretation:

TriviaQA supports the core TCL-v0 direction, but it is open-domain and answer-labeling is imperfect.

### NQ-Open

NQ-Open acted as a harder stress test. The pipeline worked, but the local small models produced very few correct held-out answers.

Interpretation:

NQ-Open is useful as a stress test, but it is too sparse at this local scale to be the main evidence source.

### SQuAD With Context

SQuAD is currently the cleanest local benchmark because the context gives small models enough correct and incorrect examples.

SQuAD-500 v2-label results:

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.0979 | 0.1582 | 0.7800 | 0.7527 | 9 | 6 |
| Qwen | Conservative TCL-v0 | 0.1090 | 0.1521 | 0.8000 | 0.8097 | 5 | 3 |
| SmolLM2 | Raw generation confidence | 0.2672 | 0.2849 | 0.5500 | 0.7523 | 20 | 2 |
| SmolLM2 | Conservative TCL-v0 | 0.1275 | 0.1688 | 0.7600 | 0.8366 | 6 | 0 |

Manual review of SQuAD-500 high-confidence wrong candidates found:

- Qwen: 1 likely false-negative label out of 14 candidates
- SmolLM2: 1 likely false-negative label out of 27 candidates

Interpretation:

SQuAD-500 is the strongest current support for TCL-v0. It shows that hidden states can add useful correctness signal, but the gain is model-dependent. Qwen improves on Brier, accuracy, AUC, and high-confidence wrong counts but not ECE. SmolLM2 improves strongly across the main metrics.

## What We Can Claim Now

Allowed cautious claim:

```text
TCL-v0 has preliminary evidence that frozen hidden states can improve confidence calibration over raw generation confidence under tested conditions.
```

More precise claim:

```text
Conservative TCL-v0 is currently the strongest practical variant because it uses hidden-state probe signal while limiting the probe's ability to raise confidence above raw generation confidence.
```

## What We Cannot Claim

Not allowed:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes to all models or tasks.
- The full four-dimensional TCL trust vector has been implemented.
- The current results are publication-grade without broader manual review and stronger compute.

## Why We Stop Running More Local Benchmarks For Now

The local laptop is CPU-only, and SQuAD-500 took several hours for two small models. More local benchmark runs would add cost faster than they add clarity.

The next valuable step is not another small local run. The current consolidation is now mostly complete, so the remaining work is public-readiness and stronger follow-up planning:

- review the standalone TCL-v0 PDF as the empirical companion artifact
- prepare a short release note if the repository will be shared
- predeclare the next stronger benchmark/model experiment
- prepare a stronger future experiment plan for GPU-scale models

## Current Research Writeup

The evidence has now been consolidated into a short TCL-v0 research writeup:

```text
TCL-v0: Frozen Hidden States as Confidence Signals for Answer Correctness
```

Writeup file:

- `TCL-v0-research-writeup.md`
- `TCL-v0-research-writeup.docx`
- `TCL-v0-research-writeup.pdf`

The writeup is currently a standalone empirical companion note to the theory paper. It should remain separate unless future evidence becomes strong enough to justify a formal empirical appendix.

## Bottom Line

We are not looping.

We have reached the first meaningful checkpoint: TCL-v0 has enough early evidence to justify a formal writeup and a better-compute follow-up experiment. The evidence is not final, but it is no longer just an idea.
