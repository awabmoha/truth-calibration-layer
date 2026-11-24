"""Draw the H4 grounding label audit (grounding protocol, Section 6).

For one model, picks 50 answers at random (seed 2035) from the held-out test splits of
its eligible pairs: 25 from gold-passage rows and 25 from distractor rows. Abstentions
are left out (they have no grounding label). The same 50 answers serve both audits:
the reviewer judges whether the passage the model saw supports the answer (grounding)
and whether the answer matches the gold answers (correctness, rule v3).

Usage: python scripts/draw_phase3_grounding_audit.py qwen|phi
Output: runs/phase3_h4g_audit/grounding_audit_sample_<model>.csv
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 2035
N_PER_TYPE = 25
# Eligible pairs after the size rule (phi/relation_extraction has 125 incorrect test answers).
HELD_OUT = {"qwen": ["squad", "nq", "hotpotqa", "relation_extraction"], "phi": ["squad", "nq", "hotpotqa"]}
FIELDS = ["model", "dataset", "id", "distractor", "question", "passage", "gold_answers", "answer",
          "nli_entail_prob", "nli_grounded", "manual_grounded", "correctness_label", "manual_correct",
          "flag_uncertain", "reason"]


def rows(model: str, dataset: str) -> list[dict]:
    run = ROOT / "runs" / "phase3_grounding" / model / dataset
    g = {r["id"]: r for r in (json.loads(x) for x in (run / "grounding.jsonl").open(encoding="utf-8") if x.strip())}
    with (ROOT / "data" / "phase3_grounding" / f"{dataset}.csv").open(encoding="utf-8", newline="") as f:
        dis = {r["id"]: int(r["distractor"]) for r in csv.DictReader(f)}
    out = []
    for r in (json.loads(x) for x in (run / "records.jsonl").open(encoding="utf-8") if x.strip()):
        if r["split"] == "test" and not g[r["id"]]["abstain"]:
            out.append({"model": model, "dataset": dataset, "id": r["id"], "distractor": dis[r["id"]],
                        "question": r["question"], "passage": r["context"],
                        "gold_answers": r["accepted_answers"], "answer": r["answer"],
                        "nli_entail_prob": round(g[r["id"]]["entail_prob"], 3),
                        "nli_grounded": g[r["id"]]["grounded"], "manual_grounded": "",
                        "correctness_label": r["correctness_label"], "manual_correct": "",
                        "flag_uncertain": "", "reason": ""})
    return out


def main() -> None:
    model = sys.argv[1]
    pool = sorted((r for d in HELD_OUT[model] for r in rows(model, d)), key=lambda r: r["id"])
    rng = random.Random(SEED)
    picked = []
    for k in (0, 1):
        picked += rng.sample([r for r in pool if r["distractor"] == k], N_PER_TYPE)
    out = ROOT / "runs" / "phase3_h4g_audit" / f"grounding_audit_sample_{model}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(picked)
    print(f"wrote {len(picked)} rows to {out}")


if __name__ == "__main__":
    main()
