# Manual Review: Held-Out Test Split

Run: `benchmark-triviaqa200-qwen-answermean-20260603T1700Z`

Dataset: TriviaQA `rc.nocontext` validation subset, 200 examples

Model: `Qwen/Qwen2.5-0.5B-Instruct`

Hidden-state method: `answer_mean`

## Scope

This review checked the 40 held-out test examples only. The goal was to verify whether the automatic correctness labels were obviously wrong before treating the benchmark diagnostic as evidence for TCL-v0 planning.

## Result

- Reviewed examples: 40
- Auto/manual agreement: 40/40
- Label changes: 0
- Manual correct labels: 7
- Manual incorrect labels: 33

The completed audit file is:

`manual_review_test_completed.csv`

## Notes

No obvious accepted-answer label corrections were found in the held-out test split. The high-confidence plain-probe errors remain genuine model errors under the benchmark answers.

One abbreviation-like case was preserved as incorrect: the model answered `ISL`, while the accepted answer list gives `IS`. This was kept unchanged under the benchmark label rather than treated as a semantic match.

## Interpretation

Because the held-out test labels were unchanged, the existing test metrics do not need to be recomputed for this run.

This supports the current cautious interpretation:

- Full TCL remains a theoretical framework and research hypothesis.
- TCL-v0 is only a diagnostic prototype.
- On this small benchmark diagnostic, plain hidden-state probing improved some metrics but produced dangerous high-confidence errors.
- The conservative TCL-v0 score, `min(raw_generation_confidence, probe_confidence)`, is the strongest current diagnostic variant because it improved ECE/Brier/accuracy while avoiding high-confidence wrong answers on this test split.

## Next Step

Run the same conservative TCL-v0 diagnostic on a larger TriviaQA subset, preferably 500 examples, then manually review only the highest-risk cases first:

- wrong answers with confidence >= 0.8
- examples where raw confidence and TCL-v0 confidence strongly disagree
- a small random sample from the held-out test split
