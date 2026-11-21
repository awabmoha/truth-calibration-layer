"""Draw the Phase 3 PopQA test sample (protocol Section 3).

Fetches the full PopQA test set (akariasai/PopQA) from the Hugging Face
datasets-server, shuffles it with seed 2030, and writes the first 4,500 questions
in that order as three batches of 1,500 (batch 1, 2, 3). Inference runs on batch 1;
batches 2 and 3 are added only if the size rule needs them, decided from labels alone.

Output: data/phase3/popqa.csv with the columns run_inference_v2.py reads
(id, title, context, question, accepted_answers, split) plus batch, s_pop, prop.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_phase1_kaggle_data import fetch_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "phase3" / "popqa.csv"
SEED = 2030
BATCH = 1500
N_BATCHES = 3
FIELDS = ["id", "title", "context", "question", "accepted_answers", "split", "batch", "s_pop", "prop"]


def to_row(r: dict, batch: int) -> dict:
    answers = r["possible_answers"]
    if isinstance(answers, str):
        answers = json.loads(answers)
    return {"id": f"popqa-{r['id']}", "title": "", "context": "", "question": r["question"].strip(),
            "accepted_answers": " | ".join(a.strip() for a in answers if a.strip()), "split": "test",
            "batch": batch, "s_pop": r.get("s_pop", ""), "prop": r.get("prop", "")}


def main() -> None:
    rows = [p["row"] for p in fetch_rows("akariasai/PopQA", "default", "test", limit=100_000)]
    rows.sort(key=lambda r: int(r["id"]))  # fixed order before shuffling
    random.Random(SEED).shuffle(rows)
    picked = [to_row(r, 1 + i // BATCH) for i, r in enumerate(rows[: BATCH * N_BATCHES])]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(picked)
    meta = {"source": "akariasai/PopQA (test)", "total_rows": len(rows), "seed": SEED,
            "batches": N_BATCHES, "batch_size": BATCH}
    (OUT.parent / "popqa_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(picked)} questions from {len(rows)} to {OUT}")


if __name__ == "__main__":
    main()
