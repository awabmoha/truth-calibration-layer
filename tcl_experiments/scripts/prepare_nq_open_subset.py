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
    parser.add_argument("--dataset", default="google-research-datasets/nq_open")
    parser.add_argument("--config", default="nq_open")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.65)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "benchmarks" / "nq_open"))
    args = parser.parse_args()

    rows = fetch_rows(args.dataset, args.config, args.split, args.limit, args.page_size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"nq_open_{args.split}_{len(rows)}"
    questions_path = out_dir / f"{base_name}.csv"
    splits_path = out_dir / f"{base_name}_splits.csv"
    manual_review_path = out_dir / f"{base_name}_manual_review.csv"
    metadata_path = out_dir / f"{base_name}_metadata.json"

    prepared = []
    for item in rows:
        row_idx = item["row_idx"]
        row = item["row"]
        item_id = f"nq-open-{args.split}-{row_idx}"
        answers = [str(answer).strip().replace("|", "/") for answer in row.get("answer", []) if str(answer).strip()]
        prepared.append({
            "id": item_id,
            "question": row["question"].strip(),
            "accepted_answers": " | ".join(answers),
        })

    splits = make_splits([row["id"] for row in prepared], args.seed, args.train_frac, args.val_frac)

    with questions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "accepted_answers"])
        writer.writeheader()
        writer.writerows(prepared)

    with splits_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "split"])
        writer.writeheader()
        for item_id in [row["id"] for row in prepared]:
            writer.writerow({"id": item_id, "split": splits[item_id]})

    with manual_review_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "question", "accepted_answers", "model_answer",
                "auto_correctness_label", "manual_correctness_label", "review_notes",
            ],
        )
        writer.writeheader()
        for row in prepared:
            writer.writerow({
                "id": row["id"],
                "question": row["question"],
                "accepted_answers": row["accepted_answers"],
                "model_answer": "",
                "auto_correctness_label": "",
                "manual_correctness_label": "",
                "review_notes": "",
            })

    split_counts = {}
    for split_name in splits.values():
        split_counts[split_name] = split_counts.get(split_name, 0) + 1

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": args.config,
        "source_split": args.split,
        "limit_requested": args.limit,
        "examples_written": len(prepared),
        "seed": args.seed,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "split_counts": split_counts,
        "questions_csv": str(questions_path),
        "splits_csv": str(splits_path),
        "manual_review_csv": str(manual_review_path),
        "claim_boundary": "This prepares benchmark input data only; it does not run TCL-v0 or produce model results.",
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
