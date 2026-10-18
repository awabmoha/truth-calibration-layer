# Documentation Guide

## Reading Order

1. **[../README.md](../README.md)**: what TCL is and where the research stands.
2. **[../TCL-Research-Program.md](../TCL-Research-Program.md)**: the master plan, with hypotheses H1-H6, the locked evaluation protocol, phase gates, pivots, and the change log.
3. **[../paper/TCL-Theory-Paper-v2.md](../paper/TCL-Theory-Paper-v2.md)**: the theory, including the joint objective, the trust vector, and why joint training could help.
4. **[phase1/TCL-Phase1-Kaggle-Checkpoint.md](phase1/TCL-Phase1-Kaggle-Checkpoint.md)**: the main Phase 1 results.

## Phase 1 (closed)

Documents in the order they were written. Each protocol was committed before the results it governs.

| Document | Role |
|---|---|
| [TCL-Phase1-Protocol.md](phase1/TCL-Phase1-Protocol.md) | Protocol for re-analyzing the TCL-v0 records (H1, H2) |
| [TCL-Phase1-Local-Checkpoint.md](phase1/TCL-Phase1-Local-Checkpoint.md) | Results: H1 and H2 fail on TCL-v0 records; defects found |
| [TCL-Phase1-Kaggle-Protocol.md](phase1/TCL-Phase1-Kaggle-Protocol.md) | Protocol for the corrected replication (H1-R, H2-R), with amendments |
| [TCL-Phase1-Kaggle-Checkpoint.md](phase1/TCL-Phase1-Kaggle-Checkpoint.md) | Results: H1-R passes at the threshold, H2-R fails; baselines; layer sweep |
| [TCL-Phase1-Label-Audit.md](phase1/TCL-Phase1-Label-Audit.md) | Label error rates on 600 audited test answers |

## Phase 2 (in progress)

| Document | Role |
|---|---|
| [TCL-Labeling-Rule-v3.md](phase2/TCL-Labeling-Rule-v3.md) | Correctness labeling rule for Phase 2, with validation on the Phase 1 audit |
| [TCL-Phase2-Protocol.md](phase2/TCL-Phase2-Protocol.md) | Locked protocol for H3-T (transfer) and H3-J (joint objective): data, arms, training, evaluation, decision rules |

## Archive

[archive/tcl-v0/](archive/tcl-v0/) holds the June 2026 TCL-v0 work: experiment plans, evidence reports, runbooks, and Kaggle run logs. Each file carries an "Archived" note. Their conclusions are superseded by Phase 1, which re-tested them on larger, corrected data.
