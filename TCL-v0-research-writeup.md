# TCL-v0: Frozen Hidden States as Confidence Signals for Answer Correctness

**Author:** Awab Mohamed  
**Project:** Truth Calibration Layer (TCL)  
**Status:** Preliminary empirical note; not full TCL validation  

## Abstract

Truth Calibration Layer (TCL) is a theory-stage framework for confidence-aware language models. This note reports TCL-v0, a narrow empirical diagnostic that asks whether frozen language-model hidden states contain useful signal for estimating answer correctness. TCL-v0 leaves the base model unchanged, extracts final-layer hidden states from generated answers, trains a lightweight logistic-regression probe, and compares probe-based confidence against raw generation confidence. Early diagnostics across TriviaQA, NQ-Open, and SQuAD suggest that hidden states can improve correctness calibration under tested conditions, especially when probe confidence is conservatively combined with raw generation confidence. The strongest current local result is SQuAD-500: conservative TCL-v0 improves Brier score, threshold accuracy, AUC, and high-confidence error counts for Qwen2.5-0.5B, and improves ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts for SmolLM2-360M. These findings are preliminary and do not validate the full TCL framework.

## 1. Research Question

Large language models often generate fluent answers even when those answers are wrong. Raw token probabilities can be useful, but they primarily measure the likelihood of the generated text under the model, not whether the answer is factually correct.

TCL-v0 asks:

```text
Can frozen LLM hidden states predict answer correctness better than raw generation confidence?
```

This question is intentionally narrower than full TCL. TCL-v0 does not implement the proposed four-dimensional trust vector. It only tests whether hidden states can support a correctness-confidence score.

## 2. Contributions

This work contributes:

- a minimal empirical bridge from the TCL theory paper to a practical diagnostic
- a reproducible pipeline for recording answers, confidence, hidden states, labels, and calibration metrics
- a comparison between raw generation confidence and hidden-state probe confidence
- a conservative TCL-v0 confidence rule:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

- early diagnostics across TriviaQA, NQ-Open, and SQuAD
- an explicit claim boundary separating TCL-v0 evidence from full TCL validation

## 3. Positioning

TCL-v0 is related to neural network calibration, language-model self-knowledge, hallucination evaluation, and semantic uncertainty.

Calibration work shows that neural networks can be poorly calibrated and that post-hoc methods such as temperature scaling can improve confidence estimates. TCL-v0 uses standard calibration metrics from this tradition, but it does not merely rescale output probabilities. It asks whether internal hidden states contain a separate correctness signal.

Language-model self-knowledge work studies whether models know what they know. TCL-v0 shares that motivation, but avoids verbalized confidence and does not fine-tune the base model. Instead, it trains a small external probe on frozen hidden states.

Truthfulness benchmarks such as TruthfulQA motivate the broader problem: models can imitate common falsehoods while sounding plausible. TCL-v0 does not solve truthfulness, but it studies a nearby operational question: can internal representations help estimate whether a generated answer is correct?

Semantic uncertainty methods estimate uncertainty by sampling or grouping semantically equivalent generations. TCL-v0 instead uses one generated answer and its hidden-state representation.

## 4. Method

For each QA example, TCL-v0 records:

- question
- context, when available
- accepted answers
- model answer
- correctness label
- raw generation confidence
- hidden-state vector
- split and run metadata

The current implementation uses:

- frozen base model
- final transformer layer
- `answer_mean` hidden-state pooling
- standardized logistic regression probe
- geometric mean generated-token probability as raw confidence
- conservative score using `min(raw, probe)`

The conservative rule was introduced after early runs showed that a plain hidden-state probe can become overconfident on fluent wrong answers. The conservative rule lets the probe lower confidence, but prevents it from raising confidence above the model's raw generation confidence.

## 5. Experimental Setup

### Models

The local experiments used small CPU-runnable instruction models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

These models were chosen because the local machine is CPU-only and cannot comfortably run larger 7B or 8B models.

### Benchmarks

The experiments used:

- TriviaQA `rc.nocontext`
- NQ-Open
- SQuAD `plain_text` with context

TriviaQA provided the first meaningful benchmark signal. NQ-Open acted as a harder stress test but was too sparse for strong local conclusions. SQuAD-500 is currently the cleanest local benchmark because context-grounding gives the small models enough correct and incorrect held-out examples.

### Metrics

The evaluation compares:

- raw generation confidence
- TCL-v0 probe confidence
- validation-calibrated TCL-v0 confidence
- conservative TCL-v0 confidence

Metrics:

- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier score
- reliability bins
- accuracy at threshold 0.5
- AUC when both classes are present
- wrong answers with confidence >= 0.8 and >= 0.9

## 6. Main Result: SQuAD-500

SQuAD-500 used:

- 325 train examples
- 75 validation examples
- 100 held-out test examples

Labels were relabeled with `strict_answer_segment_match_v2`, which allows exact normalized answer spans inside short answer sentences while avoiding fuzzy edit-distance matching.

| Model | Signal | ECE | Brier | Accuracy at 0.5 | AUC | Wrong >= 0.8 | Wrong >= 0.9 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen | Raw generation confidence | 0.0979 | 0.1582 | 0.7800 | 0.7527 | 9 | 6 |
| Qwen | Conservative TCL-v0 | 0.1090 | 0.1521 | 0.8000 | 0.8097 | 5 | 3 |
| SmolLM2 | Raw generation confidence | 0.2672 | 0.2849 | 0.5500 | 0.7523 | 20 | 2 |
| SmolLM2 | Conservative TCL-v0 | 0.1275 | 0.1688 | 0.7600 | 0.8366 | 6 | 0 |

### Interpretation

For Qwen, conservative TCL-v0 improves Brier score, threshold accuracy, AUC, and high-confidence error counts. Raw generation confidence still has better ECE.

For SmolLM2, conservative TCL-v0 improves ECE, Brier score, threshold accuracy, AUC, and high-confidence error counts.

The result is positive but not uniform. It supports the hypothesis that hidden states contain calibration signal, while showing that the best confidence construction depends on the model.

## 7. Manual Review

A targeted SQuAD-500 review examined held-out test examples labeled incorrect but assigned high confidence by at least one score.

| Model | Candidate Cases | Manual Correct | Genuine Wrong or Incomplete |
|---|---:|---:|---:|
| Qwen | 14 | 1 | 13 |
| SmolLM2 | 27 | 1 | 26 |

Most reviewed high-confidence wrong cases were real model failures, not label artifacts. Two likely false negatives were found, but a sensitivity check did not change the overall interpretation.

## 8. Broader Evidence

TriviaQA supports the same general direction. Under stricter reviewed labels, conservative TCL-v0 improved calibration metrics for both Qwen and SmolLM2 on the 500-example TriviaQA diagnostics.

NQ-Open confirmed that the pipeline can run on a harder open-domain benchmark, but the small local models produced too few correct held-out answers for strong conclusions.

The current cross-benchmark interpretation is:

```text
Frozen hidden states appear to contain useful correctness-calibration signal under tested conditions, but the signal must be constrained and tested more broadly.
```

## 9. Limitations

Current limitations:

- only small CPU-runnable models were tested
- benchmark sizes remain modest
- automatic labels still require manual audit
- SQuAD is context-grounded and does not fully represent open-domain generation
- the probe may learn dataset-specific answer patterns
- no larger GPU-scale models have been tested
- no full TCL trust vector has been implemented

The results should be read as preliminary diagnostics, not validation.

## 10. Claim Boundaries

Allowed claim:

```text
TCL-v0 provides preliminary evidence that frozen hidden states can improve answer-confidence calibration under tested conditions, especially when used as a conservative companion to raw generation confidence.
```

Not allowed:

- TCL is validated.
- TCL makes LLMs truthful.
- TCL solves hallucination.
- TCL-v0 generalizes to all models, datasets, or tasks.
- The four-dimensional TCL trust vector has been implemented or validated.

## 11. Future Work

The next meaningful experiment should move beyond local CPU diagnostics:

- use larger instruction models
- run 1,000 to 5,000 examples per benchmark
- include one context-grounded QA benchmark
- include one open-domain QA benchmark
- use fixed train/validation/test splits
- predeclare manual-review rules
- compare raw, probe, calibrated, and conservative confidence scores
- test layer-wise probes after the logistic-regression baseline is stable
- optionally compare logistic regression with a small MLP probe

## 12. Related Work

Guo et al. studied calibration in modern neural networks and showed that temperature scaling is a strong post-hoc calibration method. TCL-v0 uses calibration metrics from this tradition but evaluates hidden-state probes rather than only output-probability rescaling.

Kadavath et al. studied whether language models know what they know. TCL-v0 shares the interest in correctness awareness, but uses frozen hidden-state vectors and a small external probe.

Lin, Hilton, and Evans introduced TruthfulQA to evaluate whether models mimic human falsehoods. TCL-v0 does not directly evaluate truthfulness in that sense, but addresses the related confidence problem: estimating whether an answer should be trusted.

Kuhn, Gal, and Farquhar proposed semantic uncertainty for natural language generation. TCL-v0 differs by avoiding multiple sampled generations and instead probing the hidden state of a single generated answer.

## 13. References

- Guo, Pleiss, Sun, and Weinberger. "On Calibration of Modern Neural Networks." ICML 2017. https://arxiv.org/abs/1706.04599
- Kadavath et al. "Language Models (Mostly) Know What They Know." 2022. https://arxiv.org/abs/2207.05221
- Lin, Hilton, and Evans. "TruthfulQA: Measuring How Models Mimic Human Falsehoods." ACL 2022. https://arxiv.org/abs/2109.07958
- Kuhn, Gal, and Farquhar. "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." 2023. https://arxiv.org/abs/2302.09664

## 14. Conclusion

TCL-v0 has moved from theory to early empirical evidence. The current data does not prove TCL, but it justifies a stronger follow-up experiment with larger models, larger benchmark samples, and a predeclared manual-review protocol.

The clean conclusion is:

```text
Frozen hidden states show preliminary evidence of containing useful answer-correctness calibration signal, especially when constrained by raw generation confidence.
```
