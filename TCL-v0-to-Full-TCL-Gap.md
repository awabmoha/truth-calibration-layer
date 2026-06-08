# TCL-v0 To Full TCL Gap

Status: scope-mapping note.

This note explains why TCL-v0 is useful but does not validate the full Truth Calibration Layer proposal.

## The Full TCL Proposal

The broader TCL theory proposes a model-level reliability framework with:

- a joint training objective for next-token prediction and trust-score prediction
- a multi-dimensional trust vector
- architectural integration into the model, such as a trust head or calibration module
- evaluation of correctness, reasoning validity, grounding/provenance, and epistemic uncertainty

That is a training and architecture claim.

## What TCL-v0 Actually Tests

TCL-v0 is narrower:

```text
frozen LLM hidden states -> external probe -> answer-correctness confidence
```

It uses:

- frozen models
- one trust dimension: answer correctness
- short-answer QA benchmarks
- a post-hoc probe, not a model trained to emit trust scores
- calibration metrics and high-confidence error counts

That is a diagnostic claim, not an architecture validation claim.

## What TCL-v0 Can Justify

TCL-v0 can support this statement:

```text
Hidden states in frozen LLMs contain calibration-relevant signal for answer correctness. This motivates the TCL hypothesis that a joint token-and-trust training objective may be worth testing.
```

This is meaningful because full TCL depends on hidden representations carrying useful reliability information. If hidden states had no signal, the broader TCL direction would be weaker.

## What TCL-v0 Cannot Justify

TCL-v0 cannot show that:

- a jointly trained TCL model will work
- a four-dimensional trust vector is learnable
- a model can reliably estimate reasoning validity or provenance
- post-hoc probing is equivalent to architectural integration
- TCL solves hallucination or broad factuality

The current evidence is empirical motivation, not validation.

## What Would Begin To Validate Full TCL

A future TCL-v1 should minimally include:

- a small or mid-size model fine-tuned with a trust-score prediction head
- a joint objective combining token prediction and trust-score prediction
- at least two trust dimensions, for example correctness and grounding/provenance
- comparison against raw confidence, raw-only calibration, and frozen-probe TCL-v0
- evaluation under distribution shift or a second task family

A future result would be stronger if the jointly trained model improved calibration without relying only on confidence suppression.

## Best Current Wording

Use this wording publicly:

```text
TCL-v0 validates a supporting assumption for TCL: frozen hidden states contain answer-correctness calibration signal under tested QA settings. It does not validate the full TCL architectural proposal. Full validation requires joint token-and-trust training, multiple trust dimensions, and comparisons against post-hoc calibration baselines.
```
