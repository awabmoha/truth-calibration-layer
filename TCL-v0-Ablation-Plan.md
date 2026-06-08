# TCL-v0 Ablation Plan

Status: predeclared next-experiment plan.

This plan should be treated as the next technical checkpoint after the v0.1.1 raw-baseline correction and the v0.1.2 updated empirical writeup.

## Purpose

The current evidence shows that frozen hidden states may contain answer-correctness calibration signal. However, raw-only calibration explains a meaningful part of the earlier high-confidence wrong-answer reduction.

The next experiment must therefore answer a stricter question:

```text
Does hidden-state TCL-v0 add value beyond raw-only calibration, and which design choices are responsible for that value?
```

This is not a scale-up experiment. It is a mechanism and ablation experiment.

## Non-Goals

This experiment does not validate full TCL.

It does not test:

- joint token-and-trust training
- an integrated trust head
- the full four-dimensional trust vector
- open-ended factuality verification
- hallucination solving
- deployment readiness

It only tests whether frozen hidden states add useful answer-correctness calibration signal beyond stronger post-hoc baselines.

## Primary Research Questions

1. Does `min(raw, hidden_state_probe)` beat `min(raw, raw_only_calibrated)` on Brier score, ECE, AUC, or high-confidence wrong-answer counts?
2. Does hidden-state signal improve ranking/AUC beyond raw-only monotonic transformations?
3. Which hidden-state layer is most useful?
4. Which pooling method is most useful?
5. Does combining raw confidence and hidden states improve over hidden-only or raw-only features?
6. Does the method remain useful on both context-grounded QA and open-domain QA?

## Datasets

Use the same benchmark families as the reviewed checkpoint:

- SQuAD validation with context
- TriviaQA `rc.nocontext`

Use 1,000 examples per benchmark first.

Do not run 3,000-example scale-up until this ablation checkpoint is complete and interpreted.

## Models

Minimum model set:

- Gemma 2B-it, because it produced the strongest exploratory signal
- one comparison model from the existing reviewed set: Qwen 0.5B or Phi-3.5 Mini

Preferred model set if Kaggle time allows:

- Gemma 2B-it
- Phi-3.5 Mini
- Qwen 0.5B

Do not add a new model until the ablation script and baseline table are stable.

## Splits

Use the same split policy as the reviewed checkpoint:

```text
65 percent train
15 percent validation
20 percent test
seed = 42
```

The validation split is used for:

- temperature scaling
- Platt/logistic raw calibration
- isotonic raw calibration
- probe calibration
- model-selection decisions, if needed

The test split must remain untouched until the final reported metrics.

## Signals To Compare

Every run must report:

- raw generation confidence
- temperature-scaled raw confidence
- Platt/logistic calibrated raw confidence
- isotonic calibrated raw confidence
- hidden-state probe confidence
- calibrated hidden-state probe confidence
- `min(raw, hidden_state_probe)`
- `min(raw, raw_platt_confidence)`
- `min(raw, raw_isotonic_confidence)`
- learned fusion of raw confidence plus hidden-state probe confidence, if implemented

The required critical comparison is:

```text
min(raw_generation_confidence, hidden_state_probe_confidence)
vs
min(raw_generation_confidence, raw_only_calibrated_confidence)
```

## Hidden-State Ablations

### Layer Choice

Test at least:

- final layer
- middle layer
- early-middle layer

If exact layer indices differ across models, use proportional positions:

```text
25 percent depth
50 percent depth
last layer
```

### Pooling Method

Test:

- `answer_mean`
- `answer_last`
- `prompt_answer_mean`

Optional if easy:

- question-only mean
- context-only mean for SQuAD

Default current method remains `answer_mean`; it should not be assumed best.

## Feature Ablations

Train and compare:

- raw confidence only
- hidden states only
- raw confidence plus hidden states
- hidden-state probe score plus raw confidence

The key question is whether hidden states add value after raw confidence is already available.

## Probe Ablations

Required:

- logistic regression with `class_weight="balanced"`
- logistic regression without class weighting as a sensitivity check

Optional:

- small MLP probe

Do not introduce an MLP unless the logistic-regression ablations are complete. The logistic probe remains the interpretability baseline.

## Metrics

Primary metrics:

- Brier score
- ECE
- wrong answers with confidence >= 0.8
- wrong answers with confidence >= 0.9
- AUC

Secondary metrics:

- MCE
- reliability bins
- accuracy at threshold 0.5
- calibration curve plots, if generated
- risk-coverage or selective prediction curve, if implemented

Report every metric. Do not report only wins.

## Manual Review Rule

Use the same targeted manual-review rule as the reviewed checkpoint:

- high-confidence wrong examples
- raw-vs-conservative disagreement examples
- uncertain or suspicious automatic labels
- small sanity sample if feasible

Manual review must be completed before final interpretation.

If manual review is not completed, the result must be labeled:

```text
automatic-label ablation only
```

## Success Criteria

The ablation supports continuing hidden-state TCL-v0 if:

- hidden-state methods beat raw-only baselines on AUC in most completed model/dataset runs
- hidden-state methods beat raw-only baselines on at least one calibration metric in most completed runs
- the win is not only a reduction in high-confidence errors from suppressing confidence
- results remain visible across at least two model/dataset combinations

The ablation weakens TCL-v0 if:

- raw-only calibration matches or beats hidden-state methods on AUC, Brier, and ECE
- `min(raw, raw_only_calibrated)` matches `min(raw, hidden_probe)` on high-confidence wrong-answer counts
- hidden-state wins appear only on one model or one benchmark
- probe results are unstable across layer or pooling choices

## Decision Outcomes

After this checkpoint, write one decision note with one of:

1. **continue_tcl_v0_hidden_state_path** - hidden states survive strong raw-only baselines.
2. **revise_to_posthoc_calibration_baseline** - most gains are explained by raw-only calibration.
3. **needs_more_generalization_testing** - signal exists but is unstable across models/tasks.
4. **pause_tcl_v0_empirical_path** - hidden-state evidence is too weak or too inconsistent.

## Kaggle Execution Checklist

Before running:

- confirm GPU is enabled
- confirm the model path or Hugging Face access works
- run `scripts/check_runtime.py`
- prepare SQuAD-1000 and TriviaQA-1000 with fixed splits
- run one 50-example smoke test
- confirm both correct and incorrect labels exist in train, validation, and test

During running:

- save records JSONL
- save test predictions
- save baseline calibration outputs
- save ablation summaries
- package artifacts before closing the session

After running:

- import artifacts locally
- verify artifact manifests
- run raw-only baselines
- run targeted manual review
- recompute reviewed metrics
- write the ablation decision note

## Reporting Template

Each final ablation report should include:

- model
- benchmark
- sample size
- split counts
- positive rate by split
- hidden layer setting
- pooling setting
- probe type and regularization
- class weighting
- raw-only baseline metrics
- hidden-state metrics
- raw-only conservative fusion metrics
- hidden conservative fusion metrics
- manual-review status
- decision outcome

## Claim Boundary

Even if this ablation succeeds, the allowed claim remains narrow:

```text
Frozen hidden states add answer-correctness calibration signal beyond raw-only post-hoc calibration under tested QA settings.
```

It still would not validate full TCL.
