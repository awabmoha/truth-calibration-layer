"""Draw the Phase 3 PopQA label-audit sample (protocol Section 6).

For one model, picks 100 PopQA answers at random with seed 2031 and writes them with
their rule v3 label, for manual review before any 3b metric is read.

Usage: python scripts/draw_phase3_audit.py qwen|phi
Output: runs/phase3_audit/label_audit_sample_<model>.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeling import is_correct_v3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEED = 2031
N = 100
FIELDS = ["source", "dataset", "id", "question", "accepted_answers", "answer", "v3_label",
          "manual_label", "flag_uncertain", "reason", "rule_error"]


def main() -> None:
    model = sys.argv[1]
    path = ROOT / "runs" / "phase3_popqa" / model / "popqa" / "records.jsonl"
    recs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    recs.sort(key=lambda r: r["id"])  # fixed order before sampling
    picked = random.Random(SEED).sample(recs, N)
    out = ROOT / "runs" / "phase3_audit" / f"label_audit_sample_{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in picked:
            acc = r["accepted_answers"]
            w.writerow({"source": f"phase3_popqa_{model}", "dataset": "popqa", "id": r["id"],
                        "question": r["question"], "accepted_answers": " | ".join(acc) if isinstance(acc, list) else acc,
                        "answer": r["answer"], "v3_label": int(is_correct_v3(r["answer"], r["accepted_answers"], r["question"])),
                        "manual_label": "", "flag_uncertain": "", "reason": "", "rule_error": ""})
    print(f"wrote {N} rows to {out}")


if __name__ == "__main__":
    main()
