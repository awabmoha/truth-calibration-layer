"""Compare labeling rules v2 and v3 against the Phase 1 audited labels.

1. Error rates of v2 and v3 on the 600 audited test answers, per pairing.
2. How many labels v3 changes on all Phase 1 Kaggle records, with a random
   sample of changed rows for review.

Run: python tcl_experiments/scripts/validate_labeling_v3.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from labeling import is_correct, is_correct_v3

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs" / "phase1_kaggle_analysis" / "label_audit_assisted.csv"
RECORDS = ROOT / "runs" / "phase1_kaggle"
OUT = ROOT / "runs" / "labeling_v3_validation"
SAMPLE_SEED = 2027


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT, keep_default_na=False)
    accepted = audit["accepted_answers"].map(lambda s: [a.strip() for a in s.split("|") if a.strip()])
    audit["v2"] = [int(is_correct(a, g, q)) if a else 0 for a, g, q in zip(audit["answer"], accepted, audit["question"])]
    audit["v3"] = [int(is_correct_v3(a, g, q)) for a, g, q in zip(audit["answer"], accepted, audit["question"])]
    audit["final_label"] = audit["final_label"].astype(int)
    audit["v2_error"] = audit["v2"] != audit["final_label"]
    audit["v3_error"] = audit["v3"] != audit["final_label"]

    per = audit.groupby("pairing").agg(rows=("id", "size"), v2_errors=("v2_error", "sum"), v3_errors=("v3_error", "sum"))
    per.loc["total"] = per.sum()
    per["v2_rate"] = (per["v2_errors"] / per["rows"]).round(3)
    per["v3_rate"] = (per["v3_errors"] / per["rows"]).round(3)
    per.to_csv(OUT / "audit_error_rates.csv")
    print(per.to_string())

    newly_wrong = audit[audit["v3_error"] & ~audit["v2_error"]]
    remaining = audit[audit["v3_error"]]
    cols = ["pairing", "id", "question", "accepted_answers", "answer", "v2", "v3", "final_label", "reason"]
    remaining[cols].to_csv(OUT / "audit_remaining_v3_errors.csv", index=False)
    print(f"\nv3 errors: {len(remaining)}; rows v3 breaks that v2 got right: {len(newly_wrong)}")
    for _, r in newly_wrong.iterrows():
        print("  NEWLY WRONG:", r["pairing"], repr(r["answer"][:70]), "| gold:", r["accepted_answers"][:60])

    # Label changes on all Phase 1 Kaggle records.
    rows = []
    for path in sorted(RECORDS.glob("*/*/records.jsonl")):
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            v3 = int(is_correct_v3(r["answer"], r["accepted_answers"], r["question"]))
            rows.append({"pairing": f"{r['model_tag']}-{r['dataset']}", "id": r["id"], "split": r["split"],
                         "question": r["question"], "accepted_answers": " | ".join(r["accepted_answers"]),
                         "answer": r["answer"], "v2": int(r["correctness_label"]), "v3": v3})
    allr = pd.DataFrame(rows)
    allr["changed"] = allr["v2"] != allr["v3"]
    flips = allr.groupby("pairing").agg(rows=("id", "size"), v2_accuracy=("v2", "mean"), v3_accuracy=("v3", "mean"),
                                         changed=("changed", "sum"),
                                         to_correct=("changed", lambda s: int((s & (allr.loc[s.index, "v3"] == 1)).sum())),
                                         to_incorrect=("changed", lambda s: int((s & (allr.loc[s.index, "v3"] == 0)).sum())))
    flips[["v2_accuracy", "v3_accuracy"]] = flips[["v2_accuracy", "v3_accuracy"]].round(3)
    flips.to_csv(OUT / "phase1_label_changes.csv")
    print("\nLabel changes on all Phase 1 Kaggle records:")
    print(flips.to_string())

    changed = allr[allr["changed"]]
    sample = changed.sample(n=min(60, len(changed)), random_state=SAMPLE_SEED)
    sample.to_csv(OUT / "phase1_changed_sample.csv", index=False)
    print(f"\nWrote a sample of {len(sample)} changed rows for review to {OUT / 'phase1_changed_sample.csv'}")


if __name__ == "__main__":
    main()
