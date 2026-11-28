# TCL Phase 2b Protocol: One Rerun With the End-of-Turn Token

Status: locked before any Phase 2b training. Written November 20, 2026.

**Outcome** (added November 28, 2026; the protocol text above and below is unchanged): H3-T and H3-J failed again; the joint-training claim is dropped ([checkpoint](TCL-Phase2b-Checkpoint.md)).

This protocol fixes the single follow-up that my program plan allows after Phase 2. Phase 2 ([checkpoint](TCL-Phase2-Checkpoint.md)) failed H3-T and H3-J, and the result fell closest to the plan's "H3-T fails, C < A" row: *investigate once; if it still fails, drop the joint claim and publish the negative result.* Phase 2b is that one investigation. Changes after this point are logged as amendments (Section 9) with the reason; the decision rules in Section 6 cannot change once any Phase 2b test result has been computed.

## 1. Why a Rerun

After the Phase 2 analysis I found a flaw in my implementation of the LM loss. Its target was the reference answer **without the end-of-turn token** that closes an assistant answer in the chat format (`<|im_end|>` for Qwen2.5). The model was trained to produce the answer but never to stop after it. Arms C and D, which have an LM loss, went from a median of 2 words per answer to about 9-20, and failed the generation-quality bar.

This flaw touches only the arms that use the LM loss. It may also have affected the representations their trust heads read, so the Phase 2 transfer comparison involving C and D may not reflect TCL-J as designed. Phase 2b tests TCL-J with the flaw fixed.

## 2. The One Change

The LM target becomes the reference answer **followed by the chat template's end-of-turn token**, found by rendering an assistant turn with the model's own chat template (`end_of_turn_ids` in `train_trust_head.py`, option `--lm-target reference_eot`). For Qwen2.5-0.5B-Instruct this is `<|im_end|>`.

**Nothing else changes** from the Phase 2 protocol: model, data and splits, rule v3 labels, base-model answers, trust head, layer, pooling, LoRA settings, optimizer, learning rates, batch size and micro-batches, epochs, checkpoint selection, lambda grid and selection rule, seeds, evaluation, metrics, statistics, and generation-quality bar.

## 3. Arms

| Arm | Phase 2b | Reason |
|---|---|---|
| A: frozen head | **Reused** from Phase 2 | Has no LM loss; its code path is unchanged |
| B: trust-only LoRA | **Reused** from Phase 2 | Has no LM loss; its code path is unchanged |
| C: joint LoRA (TCL-J) | **Rerun** | Uses the LM loss |
| D: LM-only LoRA, then head | **Rerun** | Uses the LM loss |

**Lambda for C** is selected again by the Phase 2 rule (validation AUROC, seed 0, from {0.1, 0.5, 1.0}), because the LM loss it weighs has changed. The chosen lambda is then run with seeds 1 and 2.

**Trainings:** per training dataset, 3 (lambda selection) + 2 (C seeds 1-2) + 3 (D) = 8; 16 in all.

**Runs directory:** `tcl_experiments/runs/phase2b/<train dataset>/`, holding the new C and D runs, plus unchanged copies of the Phase 2 A and B runs so that the Phase 2 analysis script can run on it unmodified (`run_phase2_analysis.py --in-dir runs/phase2b --out-dir runs/phase2b_analysis`).

## 4. Compute

Kaggle T4 x2, two trainings at a time (`run_phase2_part.py --lm-target reference_eot --arms C,D`), two sessions per training dataset: part 1 (lambda selection, 3 trainings) and parts 2-3 together (C seeds 1-2 and D seeds 0-2, 5 trainings). Estimate from Phase 2 timings: about 12-14 session hours. Before the real runs, a 20-question test checks that the end-of-turn token is in the LM target and that regenerated answers stop.

## 5. Also Reported (No Decision Depends on These)

- Median and mean answer length in words of each arm's regenerated answers, against the base model's 2 words.
- The Phase 2 and Phase 2b results side by side for C and D.

## 6. Decision Rules

Exactly the Phase 2 rules (protocol Section 8), applied to the Phase 2b runs:

- **H3-T passes** if `d(C, A)` has a CI lower bound above zero in **both** transfer directions, and C meets the generation-quality bar.
- **H3-J passes** if `d(C, B)` has a CI lower bound above zero in both directions, and C meets the bar.
- The tie reading and the attribution to LM fine-tuning (`d(C, D)`) are read as in Phase 2.

**What follows, fixed now:**

| Phase 2b outcome | What I do |
|---|---|
| H3-T passes | The joint claim survives, conditional on the corrected LM target. I report Phase 2 and Phase 2b together, stating that the pass came on the second, preregistered attempt after a disclosed implementation fix, and I continue to Phase 3 with TCL-J. |
| H3-T fails | I drop the claim that joint training improves transfer and publish the negative result (Phase 2 and 2b together). TCL continues under the plan's revised Phase 3, which I will redesign around transfer before any new data. |

**This is the only rerun.** Whatever Phase 2b shows, I do not run a Phase 2c on H3 with further fixes; any later work on joint training is a new hypothesis with its own protocol.

## 7. Label Audit

No new audit of base-model labels: the fixed answers and their labels are the same as in Phase 2, which audited them. I audit a random sample of 50 regenerated test answers per dataset from Phase 2b arm C, seed 0 (seed 2029), before reading any Phase 2b metric, because the rule's error rate depends on answer length (Phase 2 audit).

## 8. Code

- `tcl_experiments/scripts/train_trust_head.py`: option `--lm-target reference_eot` (default `reference` reproduces Phase 2 exactly); the chosen setting and token ids are recorded in each run's `metrics.json`.
- `tcl_experiments/scripts/run_phase2_part.py`: options `--lm-target`, `--arms`, and comma-separated `--part`.
- `notebooks/tcl_phase2_kaggle_runs.ipynb`: pinned to the commit that contains this protocol and code.
- Analysis: `run_phase2_analysis.py`, unchanged.

## 9. Amendments

None yet.
