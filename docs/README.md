# Documentation Guide

## Reading Order

1. **[../README.md](../README.md)**: what TCL is and where the research stands.
2. **[../TCL-Research-Program.md](../TCL-Research-Program.md)**: the master plan, with hypotheses H1-H6, the locked evaluation protocol, phase gates, pivots, and the change log.
3. **[../paper/TCL-Theory-Paper-v2.md](../paper/TCL-Theory-Paper-v2.md)**: the theory, including the joint objective, the trust vector, and why joint training could help.
4. **[phase1/TCL-Phase1-Kaggle-Checkpoint.md](phase1/TCL-Phase1-Kaggle-Checkpoint.md)**: the main Phase 1 results.
5. **[phase2/TCL-Phase2-Checkpoint.md](phase2/TCL-Phase2-Checkpoint.md)** and **[phase2/TCL-Phase2b-Checkpoint.md](phase2/TCL-Phase2b-Checkpoint.md)**: the Phase 2 results and the rerun.
6. **[../paper/acl-phase2/](../paper/acl-phase2/)**: the Phase 2 negative-result paper.

## Phase 1 (closed)

Documents in the order they were written. Each protocol was committed before the results it governs.

| Document | Role |
|---|---|
| [TCL-Phase1-Protocol.md](phase1/TCL-Phase1-Protocol.md) | Protocol for re-analyzing the TCL-v0 records (H1, H2) |
| [TCL-Phase1-Local-Checkpoint.md](phase1/TCL-Phase1-Local-Checkpoint.md) | Results: H1 and H2 fail on TCL-v0 records; defects found |
| [TCL-Phase1-Kaggle-Protocol.md](phase1/TCL-Phase1-Kaggle-Protocol.md) | Protocol for the corrected replication (H1-R, H2-R), with amendments |
| [TCL-Phase1-Kaggle-Checkpoint.md](phase1/TCL-Phase1-Kaggle-Checkpoint.md) | Results: H1-R passes at the threshold, H2-R fails; baselines; layer sweep |
| [TCL-Phase1-Label-Audit.md](phase1/TCL-Phase1-Label-Audit.md) | Label error rates on 600 audited test answers |

## Phase 2 (closed: H3 fails)

| Document | Role |
|---|---|
| [TCL-Labeling-Rule-v3.md](phase2/TCL-Labeling-Rule-v3.md) | Correctness labeling rule for Phase 2, with validation on the Phase 1 audit |
| [TCL-Phase2-Protocol.md](phase2/TCL-Phase2-Protocol.md) | Locked protocol for H3-T (transfer) and H3-J (joint objective): data, arms, training, evaluation, decision rules |
| [TCL-Phase2-Checkpoint.md](phase2/TCL-Phase2-Checkpoint.md) | Results: H3-T and H3-J fail; joint training helps in-distribution but does not transfer; label audit; setup problems found |
| [TCL-Phase2b-Protocol.md](phase2/TCL-Phase2b-Protocol.md) | Locked protocol for the one rerun of arms C and D with the end-of-turn token in the LM target |
| [TCL-Phase2b-Checkpoint.md](phase2/TCL-Phase2b-Checkpoint.md) | Results: the fix works, H3-T and H3-J fail again; the joint-transfer claim is dropped |

## Phase 3 (H4-T and H4 self-consistency done)

| Document | Role |
|---|---|
| [TCL-Phase3-Protocol.md](phase3/TCL-Phase3-Protocol.md) | Locked protocol for H4-T: can a head trained on a mixture of datasets, given raw confidence, beat raw confidence on a held-out dataset (including the new PopQA)? |
| [TCL-Phase3-Checkpoint.md](phase3/TCL-Phase3-Checkpoint.md) | Results: H4-T fails (1 of 6 pairs); mixture training helps but not enough; TCL's correctness dimension becomes calibrated raw confidence |
| [TCL-Phase3-H4-Consistency-Protocol.md](phase3/TCL-Phase3-H4-Consistency-Protocol.md) | Locked protocol for H4 (self-consistency): can a one-pass head predict how consistently the model answers, and does that add to raw confidence on a held-out dataset? |
| [TCL-Phase3-H4-Consistency-Checkpoint.md](phase3/TCL-Phase3-H4-Consistency-Checkpoint.md) | Results: self-consistency is learnable from one pass on every pair, but adds to raw confidence on 3 of 6 (4 needed); H4 fails by one pair |

## Archive

[archive/tcl-v0/](archive/tcl-v0/) holds the June 2026 TCL-v0 work: experiment plans, evidence reports, runbooks, and Kaggle run logs. Each file carries an "Archived" note. Their conclusions are superseded by Phase 1, which re-tested them on larger, corrected data.
