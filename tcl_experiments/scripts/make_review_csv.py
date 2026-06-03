from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_records(Path(args.records))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "dataset",
                "split",
                "question",
                "accepted_answers",
                "model_answer",
                "auto_correctness_label",
                "manual_correctness_label",
                "raw_generation_confidence",
                "hidden_state_method",
                "run_id",
                "review_notes",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "id": record["id"],
                "dataset": record.get("dataset", ""),
                "split": record.get("split", ""),
                "question": record["question"],
                "accepted_answers": " | ".join(record.get("accepted_answers", [])),
                "model_answer": record.get("model_answer", ""),
                "auto_correctness_label": record.get("correctness_label", ""),
                "manual_correctness_label": "",
                "raw_generation_confidence": record.get("raw_generation_confidence", ""),
                "hidden_state_method": record.get("hidden_state_method", ""),
                "run_id": record.get("run_id", ""),
                "review_notes": "",
            })

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
