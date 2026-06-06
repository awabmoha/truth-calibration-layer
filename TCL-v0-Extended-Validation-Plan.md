# TCL-v0 Extended Validation Plan

Status: planned validation gate, not an experiment run

This document defines the next research step after the first TCL-v0 local checkpoint. Its purpose is to prevent open-ended experimentation by setting a clear question, scope, success criteria, failure criteria, and stopping point before any new benchmark runs.

## 1. Stage Name

The next stage should be called:

```text
TCL-v0 Extended Validation
```

It should not be called TCL-v1 yet.

Reason:

TCL-v0 still tests only one narrow idea:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

TCL-v1 should be reserved for a larger step toward the full Truth Calibration Layer framework, such as testing additional trust-vector dimensions, claim-level decomposition, retrieval/provenance signals, or a more integrated calibration architecture.

## 2. Main Goal

The goal is to decide whether the TCL-v0 signal is strong enough to justify moving toward TCL-v1.

Core question:

```text
Does hidden-state-based confidence continue to help when tested on stronger models, larger datasets, and stricter evaluation rules?
```

The goal is not to prove that TCL makes models truthful. The goal is to test whether frozen hidden states contain reusable correctness-calibration signal beyond raw generation confidence.

## 3. Decision Gate

This stage ends with one of three decisions:

1. Continue TCL-v0 research.
2. Move toward a TCL-v1 design.
3. Revise or weaken the TCL hypothesis.

The stage should not continue indefinitely. Once the predefined experiments are complete and reviewed, the decision must be written down.

## 4. Hypothesis

Primary hypothesis:

```text
Frozen hidden states contain learnable signal about answer correctness that can improve confidence calibration over raw generation confidence under controlled evaluation.
```

Operational hypothesis:

```text
Conservative TCL-v0 confidence will outperform raw generation confidence on most primary calibration and safety metrics across at least two models and two benchmark types.
```

## 5. Scope

This stage tests:

- frozen base models
- final-layer hidden states
- `answer_mean` hidden-state pooling as the default
- logistic-regression probe as the main probe
- raw generation confidence as the baseline
- conservative TCL-v0 confidence as the main TCL-v0 variant

This stage does not test:

- full TCL
- the four-dimensional trust vector
- model fine-tuning
- claim decomposition
- retrieval-grounded truth verification
- deployment readiness

## 6. Recommended Models

Use at least two models.

Minimum practical model set:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

Preferred stronger model set, if compute allows:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-3B-Instruct`

GPU/cloud target, if available later:

- one 7B or 8B instruction model

Model rule:

The base model must stay frozen. TCL-v0 is testing hidden-state probes, not base-model training.

## 7. Recommended Benchmarks

Use at least two benchmark types:

### Context-Grounded QA

Purpose:

Give small and medium models enough correct and incorrect answers for meaningful calibration.

Candidate:

- SQuAD validation split with context

### Open-Domain QA

Purpose:

Stress-test the method when answers must come from model knowledge rather than provided context.

Candidates:

- TriviaQA `rc.nocontext`
- NQ-Open

Recommendation:

Use SQuAD plus TriviaQA first. NQ-Open can remain a stress test, but earlier local results were too sparse for strong conclusions at small scale.

## 8. Sample Sizes

Minimum extended-validation run:

- 1,000 examples per benchmark
- fixed train/validation/test split
- recommended split: 65 percent train, 15 percent validation, 20 percent test

Preferred run:

- 3,000 to 5,000 examples per benchmark
- same fixed split rule

Local fallback:

- If compute is still CPU-only, do not run another large local benchmark unless it answers a specific question that existing SQuAD-500 and TriviaQA-500 results do not answer.

## 9. Signals To Compare

Every run should compare:

- raw generation confidence
- TCL-v0 probe confidence
- validation-calibrated TCL-v0 confidence
- conservative TCL-v0 confidence

Main conservative rule:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

Rationale:

Earlier TCL-v0 diagnostics showed that a plain probe can become overconfident on fluent wrong answers. The conservative rule allows the probe to lower confidence, but prevents it from raising confidence above the model's raw generation confidence.

## 10. Metrics

Primary metrics:

- Brier score
- Expected Calibration Error (ECE)
- high-confidence wrong-answer counts at confidence >= 0.8 and >= 0.9

Secondary metrics:

- Maximum Calibration Error (MCE)
- reliability bins
- accuracy at threshold 0.5
- AUC when both classes are present
- accuracy-coverage or risk-coverage curve if implemented

Reporting rule:

Report every metric for every model and benchmark. Do not report only the metrics where TCL-v0 wins.

## 11. Labeling Rules

Automatic labels must record the exact method used.

Recommended default:

- exact normalized answer match
- exact normalized answer-span match inside short answer sentences
- no broad fuzzy edit-distance matching

Manual review must be predeclared before running.

Manual-review targets:

- all held-out test examples where any confidence score is >= 0.8 and the automatic label is incorrect
- all cases where raw confidence and conservative TCL-v0 strongly disagree
- a small random sample of low-confidence correct and low-confidence incorrect examples

Manual-review output:

- candidate id
- question
- accepted answer
- model answer
- automatic label
- manual label
- reviewer note
- whether the case changes the aggregate interpretation

## 12. Success Criteria

TCL-v0 extended validation is successful if conservative TCL-v0 beats raw generation confidence on most primary criteria across at least two models and two benchmark types.

Strong support:

- lower Brier score than raw confidence
- lower ECE or clearly better reliability bins
- fewer wrong answers at confidence >= 0.8 and >= 0.9
- AUC equal to or better than raw confidence
- gains remain after manual review

Moderate support:

- conservative TCL-v0 improves Brier score and high-confidence wrong counts, but ECE is mixed
- results are consistent on one benchmark type and mixed on another
- the probe helps some models but not all

No support:

- raw confidence beats conservative TCL-v0 on most primary metrics
- gains appear only on one small model or one dataset
- improvements disappear after manual review
- TCL-v0 introduces high-confidence wrong answers that raw confidence avoids

## 13. Failure Criteria

The TCL-v0 hypothesis should be weakened if:

- hidden-state probe confidence repeatedly overfits validation data
- conservative TCL-v0 fails to improve Brier score or high-confidence wrong counts
- results depend on one benchmark's labeling artifacts
- manual review overturns most of the apparent improvement
- stronger models do not show the signal seen in smaller local models

Failure does not mean the full TCL theory is impossible. It means the current hidden-state-probe path is not strong enough as stated.

## 14. What Would Justify TCL-v1

Moving toward TCL-v1 is justified only if extended validation shows that:

- hidden states repeatedly contain useful correctness-calibration signal
- conservative TCL-v0 improves safety-relevant confidence behavior
- gains survive stronger models and larger benchmark subsets
- label review does not explain away the result
- failure cases are understood well enough to design the next layer

If those conditions hold, TCL-v1 can explore:

- additional trust-vector dimensions
- claim-level scoring
- layer-wise or multi-layer probes
- small MLP probes after logistic regression is stable
- retrieval/provenance-aware confidence features
- abstention or review-routing policies

## 15. Stopping Rule

The extended-validation stage stops after:

1. The planned model and benchmark matrix is completed, or compute limits make completion impossible.
2. Metrics are reported for raw, probe, calibrated, and conservative confidence.
3. Manual review is completed under the predeclared rules.
4. A short decision note is written.

The decision note must choose one:

- continue TCL-v0
- move toward TCL-v1
- revise the hypothesis

## 16. Immediate Next Step

Before running anything, decide the compute path:

- local CPU-only fallback
- borrowed/cloud GPU
- remote notebook environment

If compute is limited, the next action should be planning or environment selection, not another local benchmark run.

## 17. Safe Public Framing

Until extended validation is complete, the safest framing remains:

```text
TCL is a theory-stage framework. TCL-v0 provides preliminary evidence that frozen hidden states can support confidence calibration under tested conditions, but it does not validate full TCL.
```
