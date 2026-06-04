from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from labeling import CORRECTNESS_METHOD, is_correct


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--changes-csv", required=True)
    args = parser.parse_args()

    records_path = Path(args.records)
    out_path = Path(args.out)
    changes_path = Path(args.changes_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    changes_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(records_path)
    changed_rows = []
    relabeled = []

    for record in records:
        old_label = int(record["correctness_label"])
        accepted = record.get("accepted_answers") or []
        new_label = int(is_correct(record.get("model_answer", ""), accepted, record.get("question", "")))
        updated = dict(record)
        updated["auto_correctness_label"] = old_label
        updated["correctness_label"] = new_label
        updated["correctness_method"] = CORRECTNESS_METHOD
        updated["relabel_timestamp"] = datetime.now(timezone.utc).isoformat()
        relabeled.append(updated)

        if old_label != new_label:
            changed_rows.append({
                "id": record.get("id"),
                "split": record.get("split"),
                "question": record.get("question"),
                "accepted_answers": " | ".join(accepted),
                "model_answer": record.get("model_answer"),
                "old_correctness_label": old_label,
                "new_correctness_label": new_label,
                "old_correctness_method": record.get("correctness_method"),
                "new_correctness_method": CORRECTNESS_METHOD,
            })

    with out_path.open("w", encoding="utf-8") as f:
        for record in relabeled:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    fieldnames = [
        "id", "split", "question", "accepted_answers", "model_answer",
        "old_correctness_label", "new_correctness_label",
        "old_correctness_method", "new_correctness_method",
    ]
    with changes_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(changed_rows)

    summary = {
        "records_path": str(records_path),
        "out_path": str(out_path),
        "changes_csv": str(changes_path),
        "n_records": len(records),
        "old_positive_count": sum(int(r["correctness_label"]) for r in records),
        "new_positive_count": sum(int(r["correctness_label"]) for r in relabeled),
        "label_changes": len(changed_rows),
        "correctness_method": CORRECTNESS_METHOD,
        "claim_boundary": "Relabeling changes correctness labels only; it does not rerun model inference or validate full TCL.",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
