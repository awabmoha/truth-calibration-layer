from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://datasets-server.huggingface.co"


def fetch_json(endpoint: str, params: dict[str, str | int]):
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_answers(answer: dict) -> list[str]:
    values = []
    for key in ("value", "aliases", "normalized_value", "normalized_aliases"):
        item = answer.get(key)
        if item is None:
            continue
        if isinstance(item, list):
            values.extend(str(x) for x in item if str(x).strip())
        elif str(item).strip():
            values.append(str(item))

    deduped = []
    seen = set()
    for value in values:
        cleaned = value.strip().replace("|", "/")
        key = cleaned.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(cleaned)
    return deduped


def fetch_rows(dataset: str, config: str, split: str, limit: int, page_size: int):
    rows = []
    offset = 0
    while len(rows) < limit:
        length = min(page_size, limit - len(rows))
        payload = fetch_json(
            "rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
        )
        page_rows = payload.get("rows", [])
        if not page_rows:
            break
        rows.extend(page_rows)
        offset += len(page_rows)
    return rows


def make_splits(ids: list[str], seed: int, train_frac: float, val_frac: float):
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train:n_train + n_val])

    splits = {}
    for item_id in ids:
        if item_id in train_ids:
            splits[item_id] = "train"
        elif item_id in val_ids:
            splits[item_id] = "validation"
        else:
            splits[item_id] = "test"
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mandarjoshi/trivia_qa")
    parser.add_argument("--config", default="rc.nocontext")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.65)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "benchmarks" / "triviaqa"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch_rows(args.dataset, args.config, args.split, args.limit, args.page_size)
    examples = []
    for item in rows:
        source_row_idx = item["row_idx"]
        row = item["row"]
        answers = extract_answers(row["answer"])
        if not answers:
            continue
        item_id = f"triviaqa-{args.config}-{args.split}-{source_row_idx}"
        examples.append({
            "id": item_id,
            "question": row["question"],
            "answers": "|".join(answers),
            "source_dataset": args.dataset,
            "source_config": args.config,
            "source_split": args.split,
            "source_row_idx": source_row_idx,
            "question_id": row.get("question_id", ""),
            "answer_value": row["answer"].get("value", ""),
        })

    ids = [row["id"] for row in examples]
    splits = make_splits(ids, args.seed, args.train_frac, args.val_frac)
    for row in examples:
        row["split"] = splits[row["id"]]

    stem = f"triviaqa_{args.config.replace('.', '_')}_{args.split}_{len(examples)}"
    questions_path = out_dir / f"{stem}.csv"
    splits_path = out_dir / f"{stem}_splits.csv"
    review_path = out_dir / f"{stem}_manual_review.csv"
    meta_path = out_dir / f"{stem}_metadata.json"

    fieldnames = [
        "id",
        "question",
        "answers",
        "source_dataset",
        "source_config",
        "source_split",
        "source_row_idx",
        "question_id",
        "answer_value",
        "split",
    ]
    with questions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    with splits_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split"])
        writer.writeheader()
        writer.writerows({"id": row["id"], "split": row["split"]} for row in examples)

    with review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "split",
                "question",
                "accepted_answers",
                "model_answer",
                "auto_correctness_label",
                "manual_correctness_label",
                "review_notes",
            ],
        )
        writer.writeheader()
        writer.writerows({
            "id": row["id"],
            "split": row["split"],
            "question": row["question"],
            "accepted_answers": row["answers"],
            "model_answer": "",
            "auto_correctness_label": "",
            "manual_correctness_label": "",
            "review_notes": "",
        } for row in examples)

    split_counts = {}
    for row in examples:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "source_split": args.split,
        "limit_requested": args.limit,
        "examples_written": len(examples),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split_counts": split_counts,
        "questions_csv": str(questions_path),
        "splits_csv": str(splits_path),
        "manual_review_csv": str(review_path),
        "claim_boundary": "This prepares benchmark input data only; it does not run TCL-v0 or produce model results.",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
