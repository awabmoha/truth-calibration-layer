"""Draw the three Phase 4a label audits (Phase 4a protocol, Section 6), before any metric.

Qwen2.5-7B (model tag qwen7b), outputs under runs/phase4a/. Same procedures as the
Phase 3 audits (draw_phase3_audit.py, draw_phase3_h4_audit.py,
draw_phase3_grounding_audit.py), with the Phase 4a seeds and the 7B eligible pairs:
  popqa        100 PopQA answers, seed 2040, rule v3 label
  consistency  50 held-out test answers (TriviaQA, NQ-Open, PopQA), seed 2041,
               with the 10 samples and the NLI same-meaning count
  grounding    50 non-abstained held-out test answers (SQuAD, NQ, HotpotQA,
               RelationExtraction), seed 2042, with the grounding label (chosen
               labeler) and the rule v3 label. Planned as 25 gold and 25 distractor
               rows; the 7B model answered only 4 distractor rows, so all of them
               are taken and gold rows fill the rest (amendment, protocol Section 11)
Phase 1 SQuAD fails the size rule (106 incorrect test answers) and is not held out.

Usage: python scripts/draw_phase4a_audits.py
Output: runs/phase4a_audit/{popqa,consistency,grounding}_audit_sample_qwen7b.csv
"""

from __future__ import annotations

import ast
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeling import is_correct_v3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "phase4a"
OUT = ROOT / "runs" / "phase4a_audit"
TAG = "qwen7b"
CONSISTENCY_HELD_OUT = ["triviaqa", "nq_open", "popqa"]
GROUNDING_HELD_OUT = ["squad", "nq", "hotpotqa", "relation_extraction"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write(name: str, fields: list[str], rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_audit_sample_{TAG}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def popqa_audit() -> None:
    recs = sorted(read_jsonl(RUNS / "phase3_popqa" / TAG / "popqa" / "records.jsonl"), key=lambda r: r["id"])
    picked = random.Random(2040).sample(recs, 100)
    fields = ["source", "dataset", "id", "question", "accepted_answers", "answer", "v3_label",
              "manual_label", "flag_uncertain", "reason", "rule_error"]
    rows = []
    for r in picked:
        acc = r["accepted_answers"]
        rows.append({"source": f"phase4a_popqa_{TAG}", "dataset": "popqa", "id": r["id"], "question": r["question"],
                     "accepted_answers": " | ".join(acc) if isinstance(acc, list) else acc, "answer": r["answer"],
                     "v3_label": int(is_correct_v3(r["answer"], r["accepted_answers"], r["question"])),
                     "manual_label": "", "flag_uncertain": "", "reason": "", "rule_error": ""})
    write("popqa", fields, rows)


def consistency_audit() -> None:
    pool = []
    for d in CONSISTENCY_HELD_OUT:
        sub = RUNS / ("phase3_popqa" if d == "popqa" else "phase1_kaggle") / TAG / d
        pool += [r for r in read_jsonl(sub / "records.jsonl")
                 if r["split"] == "test" and r.get("self_consistency") not in (None, "None")]
    pool.sort(key=lambda r: r["id"])
    picked = random.Random(2041).sample(pool, 50)
    fields = ["model", "dataset", "id", "question", "answer"] + [f"sample_{i}" for i in range(1, 11)] + \
             ["nli_same_count", "manual_same_count", "flag_uncertain", "reason"]
    rows = []
    for r in picked:
        samples = r["se_samples"]
        if isinstance(samples, str):
            samples = ast.literal_eval(samples) if samples.startswith("[") else [samples]
        row = {"model": TAG, "dataset": r["dataset"], "id": r["id"], "question": r["question"], "answer": r["answer"],
               "nli_same_count": round(10 * float(r["self_consistency"])), "manual_same_count": "",
               "flag_uncertain": "", "reason": ""}
        row.update({f"sample_{i + 1}": s for i, s in enumerate(samples[:10])})
        rows.append(row)
    write("consistency", fields, rows)


def grounding_audit() -> None:
    pool = []
    for d in GROUNDING_HELD_OUT:
        run = RUNS / "phase3_grounding" / TAG / d
        g = {r["id"]: r for r in read_jsonl(run / "grounding.jsonl")}
        with (ROOT / "data" / "phase3_grounding" / f"{d}.csv").open(encoding="utf-8", newline="") as f:
            dis = {r["id"]: int(r["distractor"]) for r in csv.DictReader(f)}
        for r in read_jsonl(run / "records.jsonl"):
            if r["split"] == "test" and not g[r["id"]]["abstain"]:
                pool.append({"model": TAG, "dataset": d, "id": r["id"], "distractor": dis[r["id"]],
                             "question": r["question"], "passage": r["context"],
                             "gold_answers": r["accepted_answers"], "answer": r["answer"],
                             "nli_entail_prob": round(g[r["id"]]["entail_prob"], 3),
                             "nli_grounded": g[r["id"]]["grounded"], "manual_grounded": "",
                             "correctness_label": r["correctness_label"], "manual_correct": "",
                             "flag_uncertain": "", "reason": ""})
    pool.sort(key=lambda r: r["id"])
    # Amendment (protocol Section 11): the 7B model abstained on almost every distractor row, so
    # the distractor half takes every non-abstained distractor answer (up to 25) and gold rows fill
    # the rest of the 50. Decided from abstention counts alone, before any metric.
    rng = random.Random(2042)
    gold = [r for r in pool if r["distractor"] == 0]
    dist = [r for r in pool if r["distractor"] == 1]
    n_dist = min(25, len(dist))
    picked = rng.sample(gold, 50 - n_dist) + rng.sample(dist, n_dist)
    write("grounding", list(picked[0].keys()), picked)


if __name__ == "__main__":
    popqa_audit()
    consistency_audit()
    grounding_audit()
