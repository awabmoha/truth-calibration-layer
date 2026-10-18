# Labeling Rule v3

Status: frozen for Phase 2. October 18, 2026.

Code: `is_correct_v3` in `tcl_experiments/scripts/labeling.py` (rule v2 is kept unchanged next to it, so Phase 1 results stay reproducible).
Tests: `tcl_experiments/scripts/test_labeling_v3.py` (53 cases, all pass).
Validation: `tcl_experiments/scripts/validate_labeling_v3.py`, outputs in `tcl_experiments/runs/labeling_v3_validation/`.

## Why

The Phase 1 label audit (`docs/phase1/TCL-Phase1-Label-Audit.md`) found the automatic rule wrong on 40 of 600 test answers (6.7%), almost always marking correct answers wrong because they differed from the gold answer in form only. Every Phase 2 result depends on the labels, so I fixed the rule before collecting any Phase 2 data.

## What v3 Does

Both the answer and each accepted answer are normalized the same way, then compared.

| Problem in v2 | v3 behavior | Example |
|---|---|---|
| Plurals | singular and plural forms match | "pathogens" = "pathogen" |
| UK/US spelling | common variants match | "vaporization" = "vaporisation" |
| Articles and "and" | ignored | "mid-sixties" = "the mid-sixties"; lists with or without "and" |
| Abbreviations | UK, US/USA, units, percent | "UK" = "United Kingdom"; "6.04 mL" = "6.04 milliliters"; "3600 RPM" = "3600 revolutions per minute" |
| Number formats | number words, thousands separators, "6.0", ranges | "Twelve" = "12"; "2,240" = "2240"; "12-15" = "12 to 15" |
| Accents and honorifics | stripped | "Bogotá" = "Bogota"; "Pangloss" = "Dr Pangloss" |
| Transliteration | one word in a multi-word name may differ by one letter (same length, 5+ letters) | "Gamal Abdel Nasser" = "Gamal Abdul Nasser" |
| Digits inside decimals | decimals are single tokens | "4.5" no longer matches "4" |
| Refusals | only text before a refusal phrase can count | "does not specify whether ... up, down" is wrong; "political figures. The context does not specify the names" is right |
| Question echoes | a gold answer that appears in the question counts only if the answer uses it more often than the question does (not applied to numbers or either/or questions) | "The answer is snapdragon" for "another name for the snapdragon" is wrong |
| Bracketed asides in gold | dropped only when they hold a number or short symbol | "1.5 m (4 ft 11 in)" also accepts "1.5 m"; "Retriever (Labrador)" is kept whole |

## Validation

**On the 600 audited Phase 1 answers:**

| Pairing | v2 errors | v3 errors |
|---|---:|---:|
| Qwen TriviaQA | 0 | 0 |
| Qwen SQuAD | 14 | 7 |
| Phi TriviaQA | 4 | 4 |
| Phi NQ-Open | 14 | 13 |
| Gemma TriviaQA | 1 | 1 |
| Gemma SQuAD | 7 | 2 |
| **Total** | **40 (6.7%)** | **27 (4.5%)** |

v3 changes one audited label that v2 had right: "Professional Golfers' Association (PGA)" for "What sport is the concern of the PGA?". TriviaQA accepts "Golfer" for that question, so v3 follows the accepted answers; my audit label judged that the answer does not name the sport.

**This number is optimistic.** I developed v3 while looking at the audit set, so the true error rate on new data is likely higher than 4.5%.

**On answers I had not audited**, v3 changes 510 of 37,830 Phase 1 labels (1.3%), mostly from incorrect to correct on SQuAD and from correct to incorrect on Gemma's TriviaQA answers (restated questions and refusals). I reviewed random samples of 60 changed labels after each round of fixes; in the last fully reviewed round, about 49 of 60 changes (about 82%) were improvements, and the clear patterns behind the wrong ones were then fixed.

## Known Limitations

A string rule cannot judge meaning. The remaining errors are of four kinds:
- **Synonyms and paraphrases**: "O2" vs "dioxygen", "photoelectric" vs "optical" smoke detector, "Luminescence" vs "secretions that luminesce".
- **Partial answers**: "ten times" vs "ten times their own weight", "1989" vs "6 December 1989".
- **Wrong or broken gold answers**: SQuAD's "1862" for Bismarck's birth year, broken spans in NQ-Open, and incomplete alias lists (e.g. no plain "Tanganyika").
- **Containment**: a gold answer inside a longer, different name ("United Nations Convention on the Rights of the Child" contains "the United Nations").

## Commitments for Phase 2

1. All Phase 2 labels use v3 (`answer_match_v3`), frozen as of this document. Any later change is a logged amendment.
2. I will audit a fresh random sample of Phase 2 test answers, drawn from labels only, to measure v3's error rate on data it was not developed on. That number, not the 4.5% above, is the one I will report.
3. Phase 1 results are not re-labeled or re-run; they stay on rule v2 as recorded.
