# TCL Phase 1 Label Audit

Status: complete. October 18, 2026. All 600 labels, including the 31 judgment calls, were decided by the assistant under the rubric below; I have not reviewed them by hand.

Protocol: `docs/phase1/TCL-Phase1-Kaggle-Protocol.md`, Section 3: "I will audit a random sample of 100 test rows per retained pairing and report the label error rate."
Sample: `tcl_experiments/runs/phase1_kaggle_analysis/label_audit_sample.csv`, drawn with seed 2026 from labels only, before any metric was computed.
Audit file: `tcl_experiments/runs/phase1_kaggle_analysis/label_audit_assisted.csv`.

## Method

Assisted audit. For each of the 600 rows, a proposed label was written with a short reason, and every judgment call was flagged (31 rows). I then chose to have the assistant make the final call on the flagged rows too, under one consistent rule, instead of reviewing them myself. The final label is `manual_label` for flagged rows and `proposed_label` otherwise; `manual_label_source` records that these calls are not human-reviewed. This deviates from the protocol's wording ("I will audit"), and I report it as such.

**Rubric.** A row is correct (1) if the answer span states an accepted answer or an unambiguous equivalent: an alias, a spelling or transliteration variant, a singular or plural form, an abbreviation (UK), or a paraphrase with the same meaning. A row is incorrect (0) if it names a different entity or value, refuses, gives several conflicting candidates, or answers with something broader or vaguer than the question asks for. An answer that adds a wrong item to a correct one ("Cheese and wine", "courts and Parliament") is incorrect; a different name for the same thing ("Beat Generation" for "Beatnik") is correct. When the gold answer is itself wrong or broken, the answer is judged on its facts.

## Results

| Pairing | Rows | Automatic label wrong | Correct answers marked wrong | Wrong answers marked correct | Error rate |
|---|---:|---:|---:|---:|---:|
| Qwen TriviaQA | 100 | 0 | 0 | 0 | 0% |
| Qwen SQuAD | 100 | 14 | 14 | 0 | 14% |
| Phi TriviaQA | 100 | 4 | 2 | 2 | 4% |
| Phi NQ-Open | 100 | 14 | 12 | 2 | 14% |
| Gemma TriviaQA | 100 | 1 | 1 | 0 | 1% |
| Gemma SQuAD | 100 | 7 | 4 | 3 | 7% |
| **Total** | **600** | **40** | **33** | **7** | **6.7%** |

21 of the 40 corrections were unambiguous (e.g. "pathogens" vs "pathogen"); the other 19 are judgment calls made under the rubric above.

## Error Types

**The automatic rule is too strict (33 false negatives).** It marks correct answers wrong when they differ from the gold string in form only:
- singular vs plural: "pathogens" / "pathogen", "bone" / "bones", "Oxygen mask" / "oxygen masks"
- spelling and transliteration: "vaporization" / "vaporisation", "Abdel" / "Abdul"
- articles and conjunctions: "mid-sixties" / "the mid-sixties", a country list missing "and"
- abbreviations and units: "UK", "6.04 mL" / "6.04 milliliters", "O2" / "dioxygen"
- synonyms: "photoelectric" / "optical" smoke detector, "hypodermis" / "subcutis"

**Three smaller sources of false positives (7).**
- Refusals that repeat words from the question can match the gold answer: "The context does not specify whether atmospheric oxygen levels are going up, down..." was marked correct for the gold answer "down".
- Partial matches: "4.5" matched the gold answer "4" as a digit; "Intracellular and interstitial" matched on "interstitial" alone.
- Answers that add a wrong item to a correct one: "Cheese and wine", "Canadian courts and Parliament".

**Gold-answer problems (flagged).** SQuAD gives "1862" as Bismarck's birth year (he was born in 1815). One NQ-Open gold answer is a broken span ("), starting on"). Several NQ-Open answers are time-dependent (the current Defence Secretary, the most recent Deadpool film).

## What This Means

- **TriviaQA labels are reliable** (0-4% errors). Its accepted-answer lists include many aliases.
- **SQuAD and NQ-Open labels are noisy** (7-14%), almost entirely in one direction: correct answers marked wrong.
- The noise affects every method's scores on the same rows, so it does not obviously favor the probe or raw confidence. But it is not random, so I cannot rule out that it shifts the comparison. For example, plural and spelling variants may be concentrated in particular answer types. Two of the four H1-R wins are on TriviaQA (low noise) and two are on Gemma SQuAD and Phi NQ-Open (higher noise). The Qwen SQuAD loss is on the noisiest pairing.
- I will **not** re-run the Phase 1 analysis with corrected labels. That would be a new test after seeing the results. The recorded H1-R and H2-R decisions stand, with this audit attached as a stated limitation.

## Consequences for Phase 2

The labeling rule must be improved before Phase 2 data is labeled, and the improved rule must be fixed before any Phase 2 results:
1. Normalize singular and plural forms, spelling variants, number words, units, and common abbreviations.
2. Never mark a detected refusal as correct, even if it repeats gold words.
3. Do not let a gold number match inside a different number ("4" in "4.5").
4. Validate the new rule on this audit: it should reproduce my final labels on the 600 audited rows with clearly fewer errors than the current rule.
