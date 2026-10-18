"""Build the Phase 1 Kaggle question sets with fixed splits.

Implements Section 2 of docs/phase1/TCL-Phase1-Kaggle-Protocol.md:
- TriviaQA rc.nocontext: first 4,500 validation rows, random split 2,500 / 500 / 1,500.
- SQuAD v1.1: 4,500 validation questions, split by Wikipedia article so that no
  article appears in two splits, 2,500 / 500 / 1,500.
- NQ-Open: all 3,610 validation rows, random split 1,610 / 500 / 1,500.

Each output CSV has id, title, context, question, accepted_answers, split.
"""

from __future__ import annotations

import csv
import json
import random
import time
import urllib.error
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import prepare_squad_subset as squad
import prepare_triviaqa_subset as triviaqa

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "phase1_kaggle"
SEED = 42
FIELDS = ["id", "title", "context", "question", "accepted_answers", "split"]


def fetch_rows(dataset: str, config: str, split: str, limit: int, page_size: int = 100) -> list[dict]:
    """Page through the Hugging Face datasets-server, backing off on rate limits."""
    rows: list[dict] = []
    while len(rows) < limit:
        params = {"dataset": dataset, "config": config, "split": split, "offset": len(rows), "length": min(page_size, limit - len(rows))}
        for attempt in range(8):
            try:
                payload = squad.fetch_json("rows", params)
                break
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 7:
                    raise
                time.sleep(2 ** attempt)
        page = payload.get("rows", [])
        if not page:
            break
        rows.extend(page)
        time.sleep(0.5)
    return rows


def random_split(rows: list[dict], sizes: dict[str, int]) -> list[dict]:
    rng = random.Random(SEED)
    order = rows[:]
    rng.shuffle(order)
    assigned, start = [], 0
    for split in ("train", "validation", "test"):
        for row in order[start : start + sizes[split]]:
            assigned.append({**row, "split": split})
        start += sizes[split]
    if start != len(rows):
        raise ValueError(f"split sizes sum to {start}, but there are {len(rows)} rows")
    return assigned


def article_split(rows: list[dict], sizes: dict[str, int]) -> list[dict]:
    by_title: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_title[row["title"]].append(row)
    titles = sorted(by_title)
    rng = random.Random(SEED)
    rng.shuffle(titles)

    pools: dict[str, list[dict]] = {"test": [], "validation": [], "train": []}
    order = ["test", "validation", "train"]
    current = 0
    for title in titles:
        while current < len(order) - 1 and len(pools[order[current]]) >= sizes[order[current]]:
            current += 1
        pools[order[current]].extend(by_title[title])

    assigned = []
    for split in ("train", "validation", "test"):
        pool = pools[split]
        if len(pool) < sizes[split]:
            raise ValueError(f"SQuAD {split}: only {len(pool)} questions, need {sizes[split]}")
        for row in rng.sample(pool, sizes[split]):
            assigned.append({**row, "split": split})
    return assigned


def fetch_triviaqa() -> list[dict]:
    rows = fetch_rows("mandarjoshi/trivia_qa", "rc.nocontext", "validation", 4500, 100)
    out = []
    for item in rows:
        answers = triviaqa.extract_answers(item["row"]["answer"])
        if answers:
            out.append(
                {
                    "id": f"triviaqa-rc.nocontext-validation-{item['row_idx']}",
                    "title": "",
                    "context": "",
                    "question": item["row"]["question"].strip(),
                    "accepted_answers": " | ".join(answers),
                }
            )
    return out


def fetch_squad() -> list[dict]:
    rows = fetch_rows("rajpurkar/squad", "plain_text", "validation", 20000, 100)
    out = []
    for item in rows:
        row = item["row"]
        answers = squad.extract_answers(row.get("answers", {}))
        if answers:
            out.append(
                {
                    "id": f"squad-validation-{item['row_idx']}",
                    "title": row.get("title", "").strip(),
                    "context": row["context"].strip(),
                    "question": row["question"].strip(),
                    "accepted_answers": " | ".join(answers),
                }
            )
    return out


def fetch_nq_open() -> list[dict]:
    rows = fetch_rows("google-research-datasets/nq_open", "nq_open", "validation", 20000, 100)
    out = []
    for item in rows:
        answers = [str(a).strip().replace("|", "/") for a in item["row"]["answer"] if str(a).strip()]
        if answers:
            out.append(
                {
                    "id": f"nq-open-validation-{item['row_idx']}",
                    "title": "",
                    "context": "",
                    "question": item["row"]["question"].strip(),
                    "accepted_answers": " | ".join(answers),
                }
            )
    return out


def write(name: str, rows: list[dict], meta: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{name}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")}
    meta = {**meta, "n": len(rows), "split_counts": counts, "seed": SEED, "created_at": datetime.now(timezone.utc).isoformat()}
    if name == "squad":
        titles = {s: sorted({r["title"] for r in rows if r["split"] == s}) for s in counts}
        meta["articles_per_split"] = {s: len(t) for s, t in titles.items()}
        meta["articles_shared_between_splits"] = len(
            (set(titles["train"]) & set(titles["test"]))
            | (set(titles["train"]) & set(titles["validation"]))
            | (set(titles["validation"]) & set(titles["test"]))
        )
    (OUT_DIR / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, meta["split_counts"], meta.get("articles_per_split", ""))


def main() -> None:
    tqa = fetch_triviaqa()
    write("triviaqa", random_split(tqa[:4500], {"train": 2500, "validation": 500, "test": 1500}),
          {"source": "mandarjoshi/trivia_qa rc.nocontext validation", "split_unit": "question"})

    sq = fetch_squad()
    write("squad", article_split(sq, {"train": 2500, "validation": 500, "test": 1500}),
          {"source": "rajpurkar/squad plain_text validation", "split_unit": "article", "source_rows": len(sq)})

    nq = fetch_nq_open()
    n_train = len(nq) - 2000
    write("nq_open", random_split(nq, {"train": n_train, "validation": 500, "test": 1500}),
          {"source": "google-research-datasets/nq_open validation", "split_unit": "question"})


if __name__ == "__main__":
    main()
