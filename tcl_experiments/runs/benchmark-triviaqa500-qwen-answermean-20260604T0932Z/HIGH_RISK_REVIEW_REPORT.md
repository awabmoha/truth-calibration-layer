# High-Risk Manual Review

Run ID: `benchmark-triviaqa500-qwen-answermean-20260604T0932Z`

Scope: wrong held-out test answers with high TCL-v0 probe confidence.

## Result

- Reviewed high-risk cases: 5
- Label changes: 0
- All reviewed cases appear to be genuine model errors.
- Conservative TCL-v0 had 0 wrong test examples with confidence >= 0.8.
- Conservative TCL-v0 had 0 wrong test examples with confidence >= 0.9.

## Reviewed Cases

| ID | Question | Model Answer Summary | Accepted Answer Summary | Review |
|---|---|---|---|---|
| `triviaqa-rc.nocontext-validation-183` | Which was the longest moon landing? | Apollo 11 | Apollo 17 | unchanged incorrect |
| `triviaqa-rc.nocontext-validation-194` | The Black Hills lie between which two rivers? | Missouri and Big Sioux | Belle Fourche and Cheyenne | unchanged incorrect |
| `triviaqa-rc.nocontext-validation-349` | What is the alcoholic ingredient of Irish coffee? | coffee beans / no whiskey answer | whiskey / whisky | unchanged incorrect |
| `triviaqa-rc.nocontext-validation-413` | The Sign Of Four was written by which author? | Agatha Christie | Arthur Conan Doyle | unchanged incorrect |
| `triviaqa-rc.nocontext-validation-445` | Macbeth belonged to which royal house or dynasty? | House of Scots | Dunkeld / Canmore | unchanged incorrect |

## Interpretation

The high-risk review supports the same caution from the 200-example run: the plain hidden-state probe can be overconfident on fluent wrong answers. The conservative TCL-v0 rule remains the safer current diagnostic variant because it limits probe confidence by raw generation confidence.
