"""Draw the H4 self-consistency label audit (protocol Section 6).

For one model, picks 50 answers at random (seed 2032) from the held-out test splits of
its eligible pairs and writes the greedy answer, the 10 sampled answers, and the NLI
count of samples in the greedy answer's meaning cluster (10 * self_consistency). The
reviewer counts by hand how many samples mean the same as the greedy answer; the audit
reports how often the two counts disagree, and by how much.

Usage: python scripts/draw_phase3_h4_audit.py qwen|phi
Output: runs/phase3_h4_audit/consistency_audit_sample_<model>.csv
"""

from __future__ import annotations

import ast
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2031 + 1  # 2032
N = 50
HELD_OUT = {"qwen": ["triviaqa", "squad", "popqa"], "phi": ["triviaqa", "nq_open", "popqa"]}
FIELDS = ["model", "dataset", "id", "question", "answer"] + [f"sample_{i}" for i in range(1, 11)] + \
         ["nli_same_count", "manual_same_count", "flag_uncertain", "reason"]


def records(model: str, dataset: str) -> list[dict]:
    sub = Path("phase3_popqa") / model / "popqa" if dataset == "popqa" else Path("phase1_kaggle") / model / dataset
    path = ROOT / "runs" / sub / "records.jsonl"
    return [r for r in (json.loads(line) for line in path.open(encoding="utf-8") if line.strip())
            if r["split"] == "test" and r.get("self_consistency") not in (None, "None")]


def main() -> None:
    model = sys.argv[1]
    pool = sorted((r for d in HELD_OUT[model] for r in records(model, d)), key=lambda r: r["id"])
    picked = random.Random(SEED).sample(pool, N)
    out = ROOT / "runs" / "phase3_h4_audit" / f"consistency_audit_sample_{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in picked:
            samples = r["se_samples"]
            if isinstance(samples, str):
                samples = ast.literal_eval(samples) if samples.startswith("[") else [samples]
            row = {"model": model, "dataset": r["dataset"], "id": r["id"], "question": r["question"],
                   "answer": r["answer"], "nli_same_count": round(10 * float(r["self_consistency"])),
                   "manual_same_count": "", "flag_uncertain": "", "reason": ""}
            row.update({f"sample_{i + 1}": s for i, s in enumerate(samples[:10])})
            w.writerow(row)
    print(f"wrote {N} rows to {out}")


if __name__ == "__main__":
    main()
