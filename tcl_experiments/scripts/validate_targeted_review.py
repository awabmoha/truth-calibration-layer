from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TRUE_VALUES = {"1", "true", "yes", "y", "changed"}
FALSE_VALUES = {"0", "false", "no", "n", "unchanged", ""}


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_label(value: str):
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def normalize_bool(value: str):
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def build_markdown(report: dict):
    lines = [
        "# TCL-v0 Targeted Manual Review Status",
        "",
        f"Status: `{report['status']}`",
        f"Review CSV: `{report['review_csv']}`",
        f"Candidates: {report['n_candidates']}",
        f"Reviewed: {report['n_reviewed']}",
        f"Label changes: {report['n_label_changes']}",
        f"Aggregate interpretation changes: {report['n_aggregate_interpretation_changes']}",
        "",
        "## Reason Counts",
        "",
    ]
    for reason, count in sorted(report["reason_counts"].items()):
        lines.append(f"- {reason}: {count}")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This validates completion of TCL-v0 targeted label review only. It does not validate the full TCL theory.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--require-notes", action="store_true")
    args = parser.parse_args()

    review_path = Path(args.review_csv)
    rows = load_rows(review_path)
    errors = []
    required_fields = {
        "id",
        "auto_correctness_label",
        "manual_correctness_label",
        "review_reason",
        "review_notes",
        "changes_aggregate_interpretation",
    }
    if rows:
        missing = required_fields - set(rows[0].keys())
        if missing:
            errors.append(f"Review CSV is missing required columns: {', '.join(sorted(missing))}")
    else:
        errors.append("Review CSV has no candidates.")

    n_reviewed = 0
    n_label_changes = 0
    n_aggregate_changes = 0
    reason_counts = {}
    incomplete_ids = []
    invalid_change_flags = []
    missing_note_ids = []

    for row in rows:
        row_id = row.get("id", "")
        auto_label = normalize_label(row.get("auto_correctness_label", ""))
        manual_label = normalize_label(row.get("manual_correctness_label", ""))
        if manual_label is None:
            incomplete_ids.append(row_id)
            continue
        n_reviewed += 1
        if auto_label is not None and manual_label != auto_label:
            n_label_changes += 1

        change_flag = normalize_bool(row.get("changes_aggregate_interpretation", ""))
        if change_flag is None:
            invalid_change_flags.append(row_id)
        elif change_flag:
            n_aggregate_changes += 1

        if args.require_notes and not row.get("review_notes", "").strip():
            missing_note_ids.append(row_id)

        for reason in row.get("review_reason", "").split("; "):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    if incomplete_ids:
        errors.append(f"Missing manual_correctness_label for {len(incomplete_ids)} candidates.")
    if invalid_change_flags:
        errors.append(
            "Invalid changes_aggregate_interpretation value for "
            f"{len(invalid_change_flags)} candidates. Use yes/no or true/false."
        )
    if missing_note_ids:
        errors.append(f"Missing review_notes for {len(missing_note_ids)} reviewed candidates.")

    report = {
        "status": "complete" if not errors else "incomplete",
        "review_csv": str(review_path),
        "n_candidates": len(rows),
        "n_reviewed": n_reviewed,
        "n_label_changes": n_label_changes,
        "n_aggregate_interpretation_changes": n_aggregate_changes,
        "reason_counts": reason_counts,
        "errors": errors,
        "claim_boundary": "This validates TCL-v0 targeted manual review only. It does not validate the full TCL theory.",
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    out_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
