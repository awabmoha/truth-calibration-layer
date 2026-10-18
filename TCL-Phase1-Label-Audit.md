# TCL Phase 1 Label Audit

Status: assisted pass complete; my review of 31 flagged rows pending. October 18, 2026.

Protocol: `TCL-Phase1-Kaggle-Protocol.md`, Section 3: "I will audit a random sample of 100 test rows per retained pairing and report the label error rate."
Sample: `tcl_experiments/runs/phase1_kaggle_analysis/label_audit_sample.csv`, drawn with seed 2026 from labels only, before any metric was computed.
Audit file: `tcl_experiments/runs/phase1_kaggle_analysis/label_audit_assisted.csv`.

## Method

Assisted audit. For each of the 600 rows, a proposed label was written with a short reason. Every row whose correct label is a judgment call is flagged for my own review. I fill in `manual_label` for the flagged rows; unflagged rows keep the proposed label.

**Rubric.** A row is correct (1) if the answer span states an accepted answer or an unambiguous equivalent: an alias, a spelling or transliteration variant, a singular or plural form, an abbreviation (UK), or a paraphrase with the same meaning. A row is incorrect (0) if it names a different entity or value, refuses, gives several conflicting candidates, or answers with something broader or vaguer than the question asks for. When the gold answer is itself wrong or broken, I judge the answer on its facts and flag it.

## Provisional Results

| Pairing | Rows | Labels changed | Changed, unflagged | Flagged | Wrongly marked incorrect | Wrongly marked correct | Provisional error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen TriviaQA | 100 | 0 | 0 | 1 | 0 | 0 | 0% |
| Qwen SQuAD | 100 | 14 | 8 | 9 | 14 | 0 | 14% |
| Phi TriviaQA | 100 | 3 | 2 | 4 | 2 | 1 | 3% |
| Phi NQ-Open | 100 | 13 | 6 | 13 | 12 | 1 | 13% |
| Gemma TriviaQA | 100 | 1 | 0 | 1 | 1 | 0 | 1% |
| Gemma SQuAD | 100 | 7 | 5 | 3 | 4 | 3 | 7% |
| **Total** | **600** | **38** | **21** | **31** | **33** | **5** | **6.3%** |

"Changed, unflagged" are confident corrections (21 of 600, 3.5%). The rest of the changes are judgment calls awaiting my review, so the final error rate per pairing lies between those two numbers.

## Error Types

**The automatic rule is too strict (33 false negatives).** It marks correct answers wrong when they differ from the gold string in form only:
- singular vs plural: "pathogens" / "pathogen", "bone" / "bones", "Oxygen mask" / "oxygen masks"
- spelling and transliteration: "vaporization" / "vaporisation", "Abdel" / "Abdul"
- articles and conjunctions: "mid-sixties" / "the mid-sixties", a country list missing "and"
- abbreviations and units: "UK", "6.04 mL" / "6.04 milliliters", "O2" / "dioxygen"
- synonyms: "photoelectric" / "optical" smoke detector, "hypodermis" / "subcutis"

**Two smaller sources of false positives (5).**
- Refusals that repeat words from the question can match the gold answer: "The context does not specify whether atmospheric oxygen levels are going up, down..." was marked correct for the gold answer "down".
- Partial matches: "4.5" matched the gold answer "4" as a digit; "Intracellular and interstitial" matched on "interstitial" alone.

**Gold-answer problems (flagged).** SQuAD gives "1862" as Bismarck's birth year (he was born in 1815). One NQ-Open gold answer is a broken span ("), starting on"). Several NQ-Open answers are time-dependent (the current Defence Secretary, the most recent Deadpool film).

## What This Means

- **TriviaQA labels are reliable** (0-3% errors). Its accepted-answer lists include many aliases.
- **SQuAD and NQ-Open labels are noisy** (7-14%), almost entirely in one direction: correct answers marked wrong.
- The noise affects every method's scores on the same rows, so it does not obviously favor the probe or raw confidence. But it is not random, so I cannot rule out that it shifts the comparison. For example, plural and spelling variants may be concentrated in particular answer types. Two of the four H1-R wins are on TriviaQA (low noise) and two are on Gemma SQuAD and Phi NQ-Open (higher noise). The Qwen SQuAD loss is on the noisiest pairing.
- I will **not** re-run the Phase 1 analysis with corrected labels. That would be a new test after seeing the results. The recorded H1-R and H2-R decisions stand, with this audit attached as a stated limitation.

## Consequences for Phase 2

The labeling rule must be improved before Phase 2 data is labeled, and the improved rule must be fixed before any Phase 2 results:
1. Normalize singular and plural forms, spelling variants, number words, units, and common abbreviations.
2. Never mark a detected refusal as correct, even if it repeats gold words.
3. Do not let a gold number match inside a different number ("4" in "4.5").
4. Validate the new rule on this audit: it should reproduce my final labels on the 600 audited rows with clearly fewer errors than the current rule.

## Review Instructions (for me)

Open `label_audit_assisted.csv`, filter `flag_for_review = True` (31 rows), read `question`, `accepted_answers`, `answer`, and `reason`, and enter 0 or 1 in `manual_label`. Then the final per-pairing error rates are computed from `manual_label` where present and `proposed_label` otherwise.
