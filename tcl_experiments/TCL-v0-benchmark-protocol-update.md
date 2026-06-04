# TCL-v0 Benchmark Protocol Update

Status: protocol cleanup, not full benchmark rerun

## Purpose

After the NQ-Open benchmark, the main weakness was not the TCL-v0 probe. The weakness was benchmark measurement quality:

- some model outputs were blank but had high token confidence
- some outputs echoed prompt/instruction text
- some outputs were long sentence fragments instead of short answers
- some numeric answers were missed because accepted answers used different units or formatting

This update improves the benchmark protocol before scaling further.

## Changes

### Prompting

Future inference now uses chat-template prompting when the tokenizer provides a chat template.

Prompt template name:

```text
chat_short_factual_answer_v1
```

Fallback prompt remains available for models without chat templates.

### Answer Recording

Future records now distinguish:

- `raw_model_output`: the exact decoded generation
- `model_answer`: the cleaned primary answer used for correctness labeling

This keeps the audit trail while avoiding label decisions on prompt echoes or formatting artifacts.

### Confidence Handling

If answer extraction produces an empty cleaned answer, `raw_generation_confidence` is set to `0.0`. This prevents blank generations from becoming high-confidence answers.

### Labeling

The strict labeler now handles numeric-unit variants more safely.

Examples:

- `54 Mbps` can match accepted `54 Mbit/s`
- `18 years old` can match accepted `18`
- `Apollo 11` does not match accepted `Apollo 17`

## Smoke Test

Two 20-example Qwen NQ-Open protocol smoke runs were recorded:

- `runs/protocol-smoke-nqopen-qwen-v2/`
- `runs/protocol-smoke-nqopen-qwen-chat/`

The chat-template prompt produced shorter, cleaner answers and avoided blank outputs in the smoke sample. It did not solve model factual accuracy, which remains a separate limitation.

## Interpretation

This update improves measurement quality. It does not create new benchmark evidence by itself and does not validate full TCL.

The next clean benchmark should use:

- chat-template prompting when available
- cleaned primary answer extraction
- raw output preservation
- strict correctness method
- targeted manual review of positives and high-risk cases

## Next Step

Run a smaller clean benchmark rerun first, likely 100 examples on NQ-Open, before spending laptop time on another 500-example run.
