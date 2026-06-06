from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


CONFIDENCE_COLUMNS = [
    "raw_generation_confidence",
    "tcl_v0_probe_confidence",
    "tcl_v0_conservative_confidence",
    "tcl_v0_calibrated_confidence",
]


def load_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default: float | None = None):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def as_int(value, default: int | None = None):
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def add_candidate(candidates: dict[str, dict], row: dict, reason: str):
    candidate_id = row["id"]
    if candidate_id not in candidates:
        candidates[candidate_id] = dict(row)
        candidates[candidate_id]["review_reason"] = reason
    else:
        reasons = set(candidates[candidate_id]["review_reason"].split("; "))
        reasons.add(reason)
        candidates[candidate_id]["review_reason"] = "; ".join(sorted(reasons))


def enrich_row(row: dict, record_by_id: dict[str, dict]):
    record = record_by_id.get(row["id"], {})
    accepted_answers = record.get("accepted_answers", "")
    if isinstance(accepted_answers, list):
        accepted_answers = " | ".join(str(answer) for answer in accepted_answers)
    enriched = {
        "id": row.get("id", ""),
        "dataset": row.get("dataset") or record.get("dataset", ""),
        "split": row.get("split") or record.get("split", ""),
        "question": row.get("question") or record.get("question", ""),
        "context": row.get("context") or record.get("context", ""),
        "accepted_answers": accepted_answers,
        "model_answer": row.get("model_answer") or record.get("model_answer", ""),
        "auto_correctness_label": row.get("correctness_label", record.get("correctness_label", "")),
        "manual_correctness_label": "",
        "raw_generation_confidence": row.get("raw_generation_confidence", ""),
        "tcl_v0_probe_confidence": row.get("tcl_v0_probe_confidence", ""),
        "tcl_v0_conservative_confidence": row.get("tcl_v0_conservative_confidence", ""),
        "tcl_v0_calibrated_confidence": row.get("tcl_v0_calibrated_confidence", ""),
        "hidden_state_method": row.get("hidden_state_method", record.get("hidden_state_method", "")),
        "run_id": row.get("run_id", record.get("run_id", "")),
        "review_reason": "",
        "review_notes": "",
        "changes_aggregate_interpretation": "",
    }
    return enriched


def confidence_values(row: dict):
    return [
        value
        for value in (as_float(row.get(column)) for column in CONFIDENCE_COLUMNS)
        if value is not None
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--disagreement-threshold", type=float, default=0.4)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--low-confidence-sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_records(Path(args.records))
    record_by_id = {record["id"]: record for record in records}
    prediction_rows = [enrich_row(row, record_by_id) for row in load_csv_rows(Path(args.predictions))]
    candidates: dict[str, dict] = {}

    for row in prediction_rows:
        label = as_int(row.get("auto_correctness_label"))
        confidence_list = confidence_values(row)
        if label == 0 and confidence_list and max(confidence_list) >= args.high_confidence_threshold:
            add_candidate(candidates, row, "high_confidence_wrong")

        raw = as_float(row.get("raw_generation_confidence"))
        conservative = as_float(row.get("tcl_v0_conservative_confidence"))
        if raw is not None and conservative is not None and abs(raw - conservative) >= args.disagreement_threshold:
            add_candidate(candidates, row, "raw_vs_conservative_disagreement")

    low_confidence_correct = []
    low_confidence_incorrect = []
    for row in prediction_rows:
        conservative = as_float(row.get("tcl_v0_conservative_confidence"))
        label = as_int(row.get("auto_correctness_label"))
        if conservative is None or conservative > args.low_confidence_threshold:
            continue
        if label == 1:
            low_confidence_correct.append(row)
        elif label == 0:
            low_confidence_incorrect.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(low_confidence_correct)
    rng.shuffle(low_confidence_incorrect)
    for row in low_confidence_correct[: args.low_confidence_sample_size]:
        add_candidate(candidates, row, "low_confidence_correct_sample")
    for row in low_confidence_incorrect[: args.low_confidence_sample_size]:
        add_candidate(candidates, row, "low_confidence_incorrect_sample")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "dataset",
        "split",
        "question",
        "context",
        "accepted_answers",
        "model_answer",
        "auto_correctness_label",
        "manual_correctness_label",
        "raw_generation_confidence",
        "tcl_v0_probe_confidence",
        "tcl_v0_conservative_confidence",
        "tcl_v0_calibrated_confidence",
        "hidden_state_method",
        "run_id",
        "review_reason",
        "review_notes",
        "changes_aggregate_interpretation",
    ]
    rows = sorted(candidates.values(), key=lambda row: (row["review_reason"], row["id"]))
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        for reason in row["review_reason"].split("; "):
            counts[reason] = counts.get(reason, 0) + 1
    print(json.dumps({
        "out": str(out_path),
        "n_candidates": len(rows),
        "reason_counts": counts,
        "claim_boundary": "Manual review candidates support TCL-v0 label auditing only; they do not validate full TCL.",
    }, indent=2))


if __name__ == "__main__":
    main()
