"""Draw the Phase 3 grounding question sets from MRQA (grounding protocol, Section 3).

Pool datasets (SQuAD, NaturalQuestionsShort, HotpotQA): 2,500 train and 500 validation
questions from MRQA's training split, 1,500 test questions from its validation split.
New held-out dataset (RelationExtraction): 1,500 test questions from MRQA's test split.

Within each dataset and MRQA split, only the first question (by qid) of each passage is
kept, so no passage appears twice in a split, and splits drawn from different MRQA
splits never share a passage. Questions are ordered by qid and shuffled with seed 2033.
Then, per dataset and split, one third of the questions (seed 2034) get a distractor:
the passage of another question in the same dataset and split.

Downloads MRQA's parquet files once into --cache (outside the repository) and writes
data/phase3_grounding/<dataset>.csv with the columns run_inference_v2.py reads
(id, title, context, question, accepted_answers, split) plus distractor, gold_context.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "phase3_grounding"
PARQUET_API = "https://datasets-server.huggingface.co/parquet?dataset=mrqa-workshop/mrqa"
SAMPLE_SEED = 2033
DISTRACTOR_SEED = 2034
DISTRACTOR_SHARE = 1 / 3
POOL = {"squad": "SQuAD", "nq": "NaturalQuestionsShort", "hotpotqa": "HotpotQA"}
NEW = {"relation_extraction": "RelationExtraction"}
SIZES = {"train": 2500, "validation": 500, "test": 1500}
FIELDS = ["id", "title", "context", "question", "accepted_answers", "split", "distractor", "gold_context"]


def parquet_files(cache: Path, mrqa_split: str) -> list[Path]:
    files = [f for f in json.load(urllib.request.urlopen(PARQUET_API, timeout=60))["parquet_files"]
             if f["split"] == mrqa_split]
    out = []
    for f in sorted(files, key=lambda f: f["url"]):
        path = cache / mrqa_split / f["url"].rsplit("/", 1)[-1]
        if not path.exists() or path.stat().st_size != f["size"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            print("downloading", f["url"], f["size"] // 1_000_000, "MB", flush=True)
            urllib.request.urlretrieve(f["url"], path)
        out.append(path)
    return out


def read_subsets(files: list[Path], subsets: set[str]) -> dict[str, list[dict]]:
    """Rows of the wanted subsets, one question per passage (the lowest qid)."""
    best: dict[str, dict[str, dict]] = {s: {} for s in subsets}
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(columns=["subset", "qid", "question", "context", "answers"]):
            for r in batch.to_pylist():
                s = r["subset"]
                if s in subsets:
                    seen = best[s].get(r["context"])
                    if seen is None or r["qid"] < seen["qid"]:
                        best[s][r["context"]] = r
    return {s: sorted(v.values(), key=lambda r: r["qid"]) for s, v in best.items()}


def to_row(tag: str, r: dict, split: str) -> dict:
    answers = list(dict.fromkeys(a.strip() for a in r["answers"] if a and a.strip()))
    return {"id": f"{tag}-{r['qid']}", "title": "", "context": r["context"].strip(),
            "question": r["question"].strip(), "accepted_answers": " | ".join(answers), "split": split,
            "distractor": 0, "gold_context": r["context"].strip()}


def add_distractors(rows: list[dict], rng: random.Random) -> None:
    n = round(len(rows) * DISTRACTOR_SHARE)
    chosen = rng.sample(range(len(rows)), n)
    for i in chosen:
        j = rng.randrange(len(rows) - 1)
        j = j + 1 if j >= i else j          # any other question's passage
        rows[i]["context"] = rows[j]["gold_context"]
        rows[i]["distractor"] = 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True, help="folder for MRQA parquet files (outside the repo)")
    args = ap.parse_args()
    train = read_subsets(parquet_files(args.cache, "train"), set(POOL.values()))
    valid = read_subsets(parquet_files(args.cache, "validation"), set(POOL.values()))
    test = read_subsets(parquet_files(args.cache, "test"), set(NEW.values()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"source": "mrqa-workshop/mrqa (parquet)", "sample_seed": SAMPLE_SEED,
            "distractor_seed": DISTRACTOR_SEED, "distractor_share": DISTRACTOR_SHARE, "datasets": {}}
    plan = [(tag, sub, [("train", train), ("validation", train), ("test", valid)]) for tag, sub in POOL.items()]
    plan += [(tag, sub, [("test", test)]) for tag, sub in NEW.items()]
    for tag, sub, splits in plan:
        rows, used = [], {}
        for split, source in splits:
            pool = source[sub]
            key = id(source)
            if key not in used:                    # one shuffle per MRQA split, shared by train and validation
                order = list(range(len(pool)))
                random.Random(f"{SAMPLE_SEED}-{tag}-{key == id(train)}").shuffle(order)
                used[key] = {"order": order, "next": 0}
            u = used[key]
            take = [pool[i] for i in u["order"][u["next"]: u["next"] + SIZES[split]]]
            if len(take) < SIZES[split]:
                raise SystemExit(f"{sub} has only {len(take)} passages left for {split}")
            u["next"] += SIZES[split]
            split_rows = [to_row(tag, r, split) for r in take]
            add_distractors(split_rows, random.Random(f"{DISTRACTOR_SEED}-{tag}-{split}"))
            rows += split_rows
        with (OUT_DIR / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        meta["datasets"][tag] = {"mrqa_subset": sub, "rows": len(rows),
                                 "distractor_rows": sum(r["distractor"] for r in rows),
                                 "unique_passages_available": {s: len(src[sub]) for s, src in splits}}
        print(tag, len(rows), "rows,", meta["datasets"][tag]["distractor_rows"], "with distractors", flush=True)
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
