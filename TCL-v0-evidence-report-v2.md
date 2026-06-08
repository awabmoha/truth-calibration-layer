# TCL-v0 Evidence Report v2

Status: reviewed extended-validation evidence report, not full TCL validation

Generated from the June 7, 2026 checkpoint.

## Research Question

TCL-v0 asks a narrow empirical question:

```text
Can frozen LLM hidden states support better answer-correctness confidence than raw generation confidence?
```

This report does not validate the full Truth Calibration Layer. Full TCL remains a theory-stage framework. TCL-v0 only tests one practical path:

```text
frozen LLM hidden states -> small probe -> correctness/confidence score
```

## Current Position

The project has moved beyond pure theory. TCL-v0 now has reviewed preliminary evidence across multiple models and benchmarks.

Best current claim:

```text
Frozen hidden states appear to contain useful answer-correctness calibration signal under tested QA settings, especially for reducing high-confidence wrong answers.
```

The evidence supports continuing TCL-v0 research. It does not justify claiming full TCL validation, model truthfulness, hallucination solving, or deployment readiness.

## Scope Warning

Despite the project name, this report is about confidence calibration for answer correctness in bounded QA benchmark settings. It is not evidence that TCL-v0 can detect truth in the broad sense of factuality, hallucination, reasoning validity, safety, subjective judgment, or value alignment.

The phrase "truth calibration" should therefore be read narrowly here:

```text
calibrating confidence that a generated answer is correct under a benchmark label
```

## Gap To Full TCL

TCL-v0 validates only a supporting assumption for the broader theory:

```text
frozen hidden states can contain answer-correctness calibration signal
```

It does not validate the full TCL architectural proposal.

The gap is structural:

- TCL-v0 uses frozen models; full TCL requires training the model with a trust objective.
- TCL-v0 uses an external logistic probe; full TCL would require an integrated trust head or architectural module.
- TCL-v0 tests one dimension, answer correctness; full TCL proposes a multi-dimensional trust vector.
- TCL-v0 evaluates short-answer QA; full TCL would need broader factuality, reasoning, grounding, and uncertainty tests.
- TCL-v0 compares post-hoc scores; full TCL must compare a jointly trained model against post-hoc calibration baselines.

Therefore, this evidence should be read as empirical motivation:

```text
TCL-v0 motivates future joint token-and-trust training. It does not prove that such training works.
```

## Method Summary

Each run records:

- question
- context when available
- accepted answers
- model answer
- automatic correctness label
- raw generation confidence
- final-layer hidden-state vector
- TCL-v0 probe confidence

Default configuration:

- hidden-state method: `answer_mean`
- probe: logistic regression
- baseline: raw generation confidence
- main TCL-v0 variant: conservative confidence

Conservative rule:

```text
conservative_confidence = min(raw_generation_confidence, tcl_v0_probe_confidence)
```

This rule was chosen because the plain probe can become overconfident. The conservative score allows the probe to lower confidence, but prevents it from raising confidence above raw generation confidence.

## Formal Four-Run Gate

The predeclared extended-validation gate was completed with:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- SQuAD validation with context
- TriviaQA `rc.nocontext`
- 1,000 examples per run
- 200 held-out test examples per run
- targeted manual review completed for all four runs

Decision:

```text
mixed_continue_cautiously
```

Stopping-rule choice:

```text
continue TCL-v0 research
```

Interpretation:

Conservative TCL-v0 showed useful signal, but the formal gate was not strong enough to justify moving directly to TCL-v1.

## Formal Gate Results

### Qwen SQuAD-1000

Reviewed result: mixed.

```text
raw brier: 0.189956
conservative brier: 0.195644
raw ece: 0.119559
conservative ece: 0.152570
raw wrong >= 0.8: 30
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 11
conservative wrong >= 0.9: 2
```

Conservative TCL-v0 reduced high-confidence wrong answers, but raw confidence had better Brier score and ECE.

### Qwen TriviaQA-1000

Reviewed result: strong support.

```text
raw brier: 0.255366
conservative brier: 0.131989
raw ece: 0.344028
conservative ece: 0.099740
raw wrong >= 0.8: 9
conservative wrong >= 0.8: 1
raw wrong >= 0.9: 3
conservative wrong >= 0.9: 0
```

### Phi SQuAD-1000

Reviewed result: mixed.

```text
raw brier: 0.029361
conservative brier: 0.046439
raw ece: 0.009211
conservative ece: 0.040488
raw wrong >= 0.8: 5
conservative wrong >= 0.8: 4
raw wrong >= 0.9: 3
conservative wrong >= 0.9: 2
```

Conservative TCL-v0 reduced high-confidence wrong answers, but raw confidence had better Brier score and ECE.

### Phi TriviaQA-1000

Reviewed result: strong support.

```text
raw brier: 0.278213
conservative brier: 0.198426
raw ece: 0.296857
conservative ece: 0.152008
raw wrong >= 0.8: 42
conservative wrong >= 0.8: 9
raw wrong >= 0.9: 17
conservative wrong >= 0.9: 3
```

## Gemma Exploratory Extension

After the formal gate, Gemma 2B-it was added as an exploratory third model using Kaggle's local model path:

```text
/kaggle/input/models/google/gemma/transformers/2b-it/3
```

Gemma required a notebook-local prompt patch because its chat template rejected system-role messages. The patch retried with a user-only chat message.

The Gemma extension is not a replacement for the formal gate. It is additional exploratory evidence.

Exploratory six-run decision:

```text
supports_continuing_tcl_v0
```

### Gemma SQuAD-1000

Reviewed result: strong support.

```text
raw accuracy@0.5: 0.66
conservative accuracy@0.5: 0.835
raw brier: 0.281062
conservative brier: 0.134160
raw ece: 0.238979
conservative ece: 0.121125
raw wrong >= 0.8: 62
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 33
conservative wrong >= 0.9: 5
raw auc: 0.534425
conservative auc: 0.883021
```

### Gemma TriviaQA-1000

Reviewed result: strong support.

```text
raw accuracy@0.5: 0.26
conservative accuracy@0.5: 0.755
raw brier: 0.473393
conservative brier: 0.197273
raw ece: 0.546427
conservative ece: 0.176408
raw wrong >= 0.8: 76
conservative wrong >= 0.8: 10
raw wrong >= 0.9: 19
conservative wrong >= 0.9: 2
raw auc: 0.711928
conservative auc: 0.760785
```

## Main Pattern

The strongest repeated pattern is:

```text
conservative TCL-v0 reduces high-confidence wrong answers.
```

This is the most important result for the TCL idea because dangerous overconfidence is closer to the central problem than ordinary accuracy alone.

The pattern is strongest on:

- TriviaQA
- Gemma 2B-it
- high-confidence wrong-answer counts

The pattern is mixed on:

- SQuAD for Qwen and Phi
- Brier/ECE when raw confidence is already strong

## Interpretation

The results support the idea that hidden states carry correctness-relevant signal. They also show why TCL-v0 must be constrained:

- plain probe confidence can help but may become overconfident
- calibrated confidence can reduce high-confidence wrong answers but can harm other metrics
- conservative confidence is the most stable variant so far

This supports continuing TCL-v0 research, not jumping to TCL-v1.

## Critical Limitations And Risks

The current evidence has important limits:

- the practical work is still mostly short-answer QA correctness calibration
- SQuAD and TriviaQA are useful starting points, but they are narrower than open-ended factuality or multi-sentence generation
- stronger standard calibration baselines are not yet implemented, including temperature scaling, Platt scaling, isotonic regression, and learned raw-plus-probe fusion
- the conservative rule can reduce overconfidence, but it cannot fix base-model underconfidence because it never raises confidence above the raw score
- TCL-v0 does not change the underlying model answer or reasoning process
- correctness labels can still be noisy, even after targeted manual review
- the current repository is organized as a research record and experiment scaffold, not as a clean reusable library

The main misuse risk is over-trust. A user could see lower high-confidence error counts and treat TCL-v0 as a hallucination fix. That would be incorrect. TCL-v0 currently reshapes confidence estimates; it does not verify truth independently.

## Baseline And Ablation Gaps

The strongest next critique to answer is whether hidden states add value beyond simpler calibration methods. A first CPU-only baseline checkpoint has now been added in `TCL-v0-Baseline-Checkpoint.md`.

Required next baselines:

- raw generation confidence
- temperature scaling
- Platt/logistic calibration on raw confidence
- isotonic regression
- learned fusion of raw confidence and probe confidence
- conservative fusion

Required ablations:

- hidden layer choice
- hidden-state pooling method
- probe family
- raw-only vs hidden-only vs raw-plus-hidden features
- conservative min vs learned fusion

Initial baseline result: raw-only calibration explains a meaningful part of the earlier high-confidence-error reductions. Hidden-state TCL-v0 remains competitive, especially on AUC/ranking and several Brier comparisons, but high-confidence-error reduction alone is no longer enough evidence for hidden-state-specific calibration gains.

Until broader ablations are run, TCL-v0 should be described as promising but incomplete.

## Claim Boundaries

Allowed:

- TCL is a theory-stage framework.
- TCL-v0 is an early empirical diagnostic.
- Reviewed TCL-v0 evidence suggests hidden states can improve confidence calibration under tested QA conditions.
- Conservative TCL-v0 is currently the strongest practical variant.
- The clearest benefit is reducing high-confidence wrong answers.

Not allowed:

- TCL is validated.
- TCL makes models truthful.
- TCL solves hallucination.
- TCL-v0 generalizes to all tasks or models.
- The full four-dimensional TCL trust vector has been implemented.
- The system is deployment-ready.

## Recommended Next Step

Stop running new experiments until this checkpoint is frozen and summarized.

Immediate work:

- update README and result summaries
- follow `TCL-v0-Roadmap.md`
- keep this report as the current empirical checkpoint
- package the project state
- optionally commit the checkpoint

Next experiment, only after documentation and baselines are clean:

```text
Gemma 2B-it on TriviaQA-1000 baseline-and-ablation checkpoint
compare raw, temperature scaling, Platt/logistic raw calibration, isotonic, hidden-state probe, and conservative fusion
same targeted manual-review rules
```

Reason:

TriviaQA is where TCL-v0 repeatedly shows the clearest practical value, and Gemma produced the strongest reviewed exploratory signal. But the next useful result is not merely a bigger dataset; it is proving whether hidden-state TCL-v0 still matters after stronger baselines are included.

## Bottom Line

TCL-v0 is now a real empirical foothold for the broader TCL theory. The evidence is preliminary, but it is no longer just a theoretical sketch.

Best final wording:

```text
TCL remains a theory-stage framework, but TCL-v0 now has reviewed preliminary evidence across multiple models and QA benchmarks that frozen hidden-state probes can reduce dangerous overconfidence in answer correctness estimates.
```
