from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = [
    "planned_run_id",
    "benchmark",
    "limit",
    "model",
    "platform",
    "gpu",
    "status",
    "artifact_zip",
    "run_dir",
    "manual_review_status",
    "compute_limit_reason",
    "notes",
]

STATUS_VALUES = {
    "planned",
    "running",
    "artifact_downloaded",
    "imported",
    "verified",
    "manual_review_pending",
    "reviewed",
    "included_in_decision",
    "failed",
    "skipped_compute_limit",
}


def load_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def build_markdown(report: dict):
    lines = [
        "# TCL-v0 Extended Validation Run Tracker Status",
        "",
        f"Tracker: `{report['tracker']}`",
        f"Status: `{report['status']}`",
        f"Rows: `{report['row_count']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in sorted(report["counts"].items()):
        lines.append(f"- {key}: {value}")

    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This tracker records cloud execution attempts for TCL-v0 only. It is not evidence that full TCL is validated.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args()

    tracker = Path(args.tracker)
    errors: list[str] = []
    warnings: list[str] = []

    if not tracker.exists():
        raise SystemExit(f"Tracker CSV does not exist: {tracker}")

    fieldnames, rows = load_rows(tracker)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in fieldnames]
    if missing_fields:
        errors.append(f"Missing required columns: {', '.join(missing_fields)}")

    counts: dict[str, int] = {}
    for row_number, row in enumerate(rows, 2):
        status = (row.get("status") or "").strip()
        counts[status or "blank"] = counts.get(status or "blank", 0) + 1
        if status not in STATUS_VALUES:
            errors.append(f"Row {row_number} has unknown status: {status!r}")

        artifact_zip = (row.get("artifact_zip") or "").strip()
        run_dir = (row.get("run_dir") or "").strip()
        manual_review_status = (row.get("manual_review_status") or "").strip()
        compute_limit_reason = (row.get("compute_limit_reason") or "").strip()

        if status in {"artifact_downloaded", "imported", "verified", "manual_review_pending", "reviewed", "included_in_decision"}:
            if not artifact_zip and not run_dir:
                errors.append(f"Row {row_number} status {status!r} needs artifact_zip or run_dir.")
        if status in {"reviewed", "included_in_decision"} and manual_review_status != "complete":
            errors.append(f"Row {row_number} status {status!r} requires manual_review_status=complete.")
        if status == "skipped_compute_limit" and not compute_limit_reason:
            errors.append(f"Row {row_number} skipped for compute limit but has no compute_limit_reason.")
        if status == "failed" and not (compute_limit_reason or (row.get("notes") or "").strip()):
            warnings.append(f"Row {row_number} failed without notes or compute_limit_reason.")

    if not rows:
        warnings.append("Tracker has no run rows yet.")

    report = {
        "status": "pass" if not errors else "fail",
        "tracker": str(tracker),
        "row_count": len(rows),
        "counts": counts,
        "errors": errors,
        "warnings": warnings,
        "claim_boundary": "This validates the TCL-v0 execution tracker only. It does not validate the full TCL theory.",
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
