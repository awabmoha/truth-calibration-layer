# TCL-v0: Frozen Hidden States as Confidence Signals for Answer Correctness

Status: preliminary TCL-v0 research writeup, not full TCL validation

## Abstract

Truth Calibration Layer (TCL) is a theory-stage framework for confidence-aware language models. This writeup reports the first practical diagnostic, TCL-v0, which tests a narrower hypothesis: whether frozen language-model hidden states contain useful signal for predicting answer correctness. TCL-v0 records model answers, raw generation confidence, hidden-state vectors, and correctness labels, then trains a lightweight logistic-regression probe to estimate correctness confidence. Early diagnostics across TriviaQA, NQ-Open, and SQuAD suggest that hidden states can contain useful calibration signal, especially when probe confidence is combined conservatively with raw generation confidence. These results are preliminary and do not validate full TCL.

## 1. Motivation

Large language models can produce fluent answers that sound confident even when they are wrong. Raw token probabilities are useful but incomplete: they measure generation likelihood, not necessarily factual correctness.

TCL-v0 asks whether internal model states can provide an additional confidence signal. The central question is:

```text
Can frozen LLM hidden states predict answer correctness better than raw generation confidence?
```

This is only one practical slice of the broader TCL theory. It does not implement the full four-dimensional TCL trust vector.

## 2. Positioning

TCL-v0 sits between three nearby research directions:

- Neural network calibration: prior work shows that modern neural networks can be miscalibrated and that post-hoc calibration methods such as temperature scaling can help.
- Language-model self-knowledge: prior work asks whether models can estimate what they know, often through verbalized confidence, multiple-choice confidence, or auxiliary heads.
- Hallucination and truthfulness evaluation: benchmarks such as TruthfulQA test whether models repeat common falsehoods, while uncertainty methods such as semantic uncertainty estimate whether generations are meaningfully stable.

TCL-v0 is different in one narrow way: it does not ask the model to verbalize confidence, sample many generations, or change the base model. It freezes the model and tests whether a small probe on hidden states can predict whether the produced answer is correct.

## 3. Contributions

This package contributes:

- a theory-to-practice bridge from TCL to a minimal empirical test
- a reproducible TCL-v0 pipeline for QA calibration diagnostics
- comparisons between raw generation confidence and hidden-state probe confidence
- a conservative score that bounds probe confidence by raw generation confidence
- early evidence across TriviaQA, NQ-Open, and SQuAD
- claim boundaries that keep the results separate from full TCL validation

## 4. Method

For each QA example, TCL-v0 records:

- question
- context, when the benchmark provides one
- accepted answers
- model answer
- correctness label
- raw generation confidence
- hidden-state vector
- split and run metadata

Current implementation:

- base model remains frozen
- hidden-state method: `answer_mean`
- hidden-state layer: final transformer layer
- probe: standardized logistic regression
- raw confidence: geometric mean probability of generated tokens
- conservative score: `min(raw_generation_confidence, tcl_v0_probe_confidence)`

The conservative score was introduced because a plain hidden-state probe can become overconfident on fluent wrong answers. Conservative TCL-v0 allows the hidden-state probe to lower confidence but prevents it from raising confidence above raw generation confidence.

## 5. Models And Benchmarks

Local experiments used small CPU-runnable instruction models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Benchmarks:

- TriviaQA `rc.nocontext`
- NQ-Open
- SQuAD `plain_text` with context

NQ-Open was useful as a stress test but too sparse at this scale. SQuAD-500 is currently the cleanest local benchmark because the provided context gives small models a healthier mix of correct and incorrect answers.

## 6. Evaluation

TCL-v0 compares:

- raw generation confidence
- plain TCL-v0 probe confidence
- validation-calibrated TCL-v0 confidence
- conservative TCL-v0 confidence

Metrics:

- Expected Calibration Error
- Maximum Calibration Error
- Brier score
- reliability bins
- accuracy at threshold 0.5
- AUC when both classes are present
- wrong answers with confidence >= 0.8 and >= 0.9

## 7. Main Result: SQuAD-500

SQuAD-500 used 325 train examples, 75 validation examples, and 100 held-out test examples. Labels were relabeled with `strict_answer_segment_match_v2`, which allows exact normalized answer spans inside short answer sentences while still avoiding fuzzy edit-distance matching.

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.0979 | 0.1582 | 0.7800 | 0.7527 | 9 | 6 |
| Qwen | Conservative TCL-v0 | 0.1090 | 0.1521 | 0.8000 | 0.8097 | 5 | 3 |
| SmolLM2 | Raw generation confidence | 0.2672 | 0.2849 | 0.5500 | 0.7523 | 20 | 2 |
| SmolLM2 | Conservative TCL-v0 | 0.1275 | 0.1688 | 0.7600 | 0.8366 | 6 | 0 |

Interpretation:

- Qwen: conservative TCL-v0 improves Brier score, threshold accuracy, AUC, and high-confidence error counts, but raw generation confidence has better ECE.
- SmolLM2: conservative TCL-v0 improves ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts.

This is a useful but not universal win. The result supports the idea that hidden states contain calibration signal, while showing that the value of the signal depends on model and score construction.

## 8. Manual Review

A targeted SQuAD-500 review examined held-out test examples that were labeled incorrect but received high confidence from at least one score.

| Model | Candidate Cases | Manual Correct | Genuine Wrong or Incomplete |
|---|---:|---:|---:|
| Qwen | 14 | 1 | 13 |
| SmolLM2 | 27 | 1 | 26 |

The review found that most high-confidence wrong cases were real failures, not label artifacts. Two likely false negatives were found, but a sensitivity check did not change the overall interpretation.

## 9. Broader Evidence

TriviaQA supports the same general direction: conservative TCL-v0 improved calibration metrics across Qwen and SmolLM2 under stricter reviewed labels.

NQ-Open showed that the pipeline can run on a harder benchmark, but the small local models produced too few correct held-out examples for strong conclusions.

Together, the current evidence supports a cautious statement:

```text
Frozen hidden states appear to contain useful correctness-calibration signal under tested conditions, but the signal must be constrained and tested more broadly.
```

## 10. Limitations

Current limitations:

- small CPU-runnable models only
- limited benchmark sizes
- automatic labels still require manual audit
- SQuAD is context-grounded and may not represent open-domain generation
- the probe may learn dataset-specific patterns
- no GPU-scale models have been tested
- no full TCL trust vector has been implemented

The current results should be treated as preliminary diagnostics, not validation.

## 11. Claim Boundaries

Allowed:

- TCL remains a theoretical framework and research hypothesis.
- TCL-v0 is an early confidence-only diagnostic.
- Frozen hidden states show preliminary evidence of correctness-calibration signal.
- Conservative TCL-v0 is the strongest current practical variant in this package.

Not allowed:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes to all models, datasets, or tasks.
- The four-dimensional TCL trust vector has been implemented or validated.

## 12. Next Experiment Plan

The next meaningful step is not more local CPU runs. The next step is a stronger GPU-scale experiment:

- use larger instruction models
- run at least 1,000 to 5,000 examples per benchmark
- include SQuAD or another context-grounded QA benchmark
- include at least one open-domain benchmark
- preserve fixed train/validation/test splits
- perform a predeclared manual-review protocol
- compare raw, probe, calibrated, and conservative scores
- optionally test layer-wise probes and a small MLP after logistic regression is stable

## 13. Related Work

Calibration of neural networks is the immediate methodological background. Guo et al. showed that modern neural networks can be poorly calibrated and that temperature scaling is a strong post-hoc baseline. TCL-v0 uses calibration metrics from this tradition, but the intervention is different: it tests hidden-state probes rather than only rescaling output probabilities.

Language-model self-knowledge is also closely related. Kadavath et al. studied whether language models know what they know, including confidence-related behavior and answerability. TCL-v0 shares the focus on correctness awareness, but uses frozen hidden-state vectors and a small external probe instead of relying only on model verbalization or task-specific heads.

Truthfulness benchmarks such as TruthfulQA frame the broader problem: models can generate fluent answers that reproduce common falsehoods. TCL-v0 does not solve truthfulness, but it targets a nearby practical question: whether internal representations can help estimate when a generated answer is likely correct.

Semantic uncertainty methods estimate uncertainty by grouping semantically equivalent generations. TCL-v0 takes a different route: instead of sampling multiple outputs, it uses the hidden state of a single generated answer.

## 14. References

- Guo, Pleiss, Sun, and Weinberger. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
- Kadavath et al. "Language Models (Mostly) Know What They Know." 2022. https://arxiv.org/abs/2207.05221
- Lin, Hilton, and Evans. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022. https://arxiv.org/abs/2109.07958
- Kuhn, Gal, and Farquhar. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." 2023. https://arxiv.org/abs/2302.09664

## 15. Bottom Line

TCL-v0 has moved from idea to early evidence. The current data does not prove TCL, but it does justify a stronger follow-up experiment.

The cleanest current conclusion is:

```text
TCL-v0 provides preliminary evidence that frozen hidden states can improve answer-confidence calibration under tested conditions, especially when used as a conservative companion to raw generation confidence.
```
