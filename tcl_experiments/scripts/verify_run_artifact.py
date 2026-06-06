from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LOWER_IS_BETTER = {
    "brier",
    "ece",
    "mce",
    "wrong_conf_ge_0_8",
    "wrong_conf_ge_0_9",
}
HIGHER_IS_BETTER = {
    "accuracy_at_0_5",
    "auc",
}

REQUIRED_RECORD_FIELDS = [
    "id",
    "question",
    "model_answer",
    "correctness_label",
    "raw_generation_confidence",
    "hidden_state_vector",
    "hidden_state_method",
    "run_id",
]

REQUIRED_PREDICTION_FIELDS = [
    "id",
    "question",
    "model_answer",
    "correctness_label",
    "raw_generation_confidence",
    "tcl_v0_probe_confidence",
    "tcl_v0_conservative_confidence",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
    return records


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def file_status(path: Path, label: str, errors: list[str], warnings: list[str], required: bool = True):
    if path.exists() and path.is_file():
        return True
    message = f"Missing {label}: {path}"
    if required:
        errors.append(message)
    else:
        warnings.append(message)
    return False


def check_record_schema(records: list[dict], errors: list[str], warnings: list[str], sample_limit: int = 10):
    if not records:
        errors.append("Records file is empty.")
        return

    for i, record in enumerate(records[:sample_limit], 1):
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            errors.append(f"Record sample {i} is missing fields: {', '.join(missing)}")
        vector = record.get("hidden_state_vector")
        if not isinstance(vector, list) or not vector:
            errors.append(f"Record sample {i} has no non-empty hidden_state_vector.")

    labels = {record.get("correctness_label") for record in records}
    if labels != {0, 1}:
        warnings.append(
            f"Records contain label set {sorted(str(label) for label in labels)}; "
            "probe metrics may be weak unless both 0 and 1 are present."
        )

    split_counts = {}
    for record in records:
        split = record.get("split", "")
        split_counts[split] = split_counts.get(split, 0) + 1
    for expected_split in ["train", "validation", "test"]:
        if split_counts.get(expected_split, 0) == 0:
            warnings.append(f"No records found for split '{expected_split}'.")


def metric_delta(raw_metrics: dict, candidate_metrics: dict, metric_name: str):
    raw_value = raw_metrics.get(metric_name)
    candidate_value = candidate_metrics.get(metric_name)
    if raw_value is None or candidate_value is None:
        return None

    delta = candidate_value - raw_value
    if metric_name in LOWER_IS_BETTER:
        winner = "candidate" if candidate_value < raw_value else "raw" if candidate_value > raw_value else "tie"
    elif metric_name in HIGHER_IS_BETTER:
        winner = "candidate" if candidate_value > raw_value else "raw" if candidate_value < raw_value else "tie"
    else:
        winner = "unknown"
    return {
        "raw": raw_value,
        "candidate": candidate_value,
        "delta_candidate_minus_raw": delta,
        "winner": winner,
    }


def compare_against_raw(metrics: dict, candidate_key: str):
    raw_metrics = metrics.get("raw_generation_confidence", {})
    candidate_metrics = metrics.get(candidate_key, {})
    comparison = {}
    for metric_name in sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER):
        delta = metric_delta(raw_metrics, candidate_metrics, metric_name)
        if delta is not None:
            comparison[metric_name] = delta
    wins = sum(1 for item in comparison.values() if item["winner"] == "candidate")
    losses = sum(1 for item in comparison.values() if item["winner"] == "raw")
    ties = sum(1 for item in comparison.values() if item["winner"] == "tie")
    return {
        "wins_vs_raw": wins,
        "losses_vs_raw": losses,
        "ties_vs_raw": ties,
        "metrics": comparison,
    }


def check_predictions(path: Path, errors: list[str], warnings: list[str]):
    rows = load_csv_rows(path)
    if not rows:
        errors.append(f"Prediction file is empty: {path}")
        return 0
    fieldnames = set(rows[0].keys())
    missing = [field for field in REQUIRED_PREDICTION_FIELDS if field not in fieldnames]
    if missing:
        errors.append(f"Prediction file is missing fields: {', '.join(missing)}")
    if "tcl_v0_calibrated_confidence" not in fieldnames:
        warnings.append("Prediction file has no tcl_v0_calibrated_confidence column.")
    return len(rows)


def build_markdown(report: dict):
    lines = [
        "# TCL-v0 Run Artifact Verification",
        "",
        f"Run directory: `{report['run_dir']}`",
        f"Status: `{report['status']}`",
        f"Records: `{report['record_count']}`",
        f"Test predictions: `{report['test_prediction_count']}`",
        "",
        "## Required Files",
        "",
    ]
    for item in report["files"]:
        lines.append(f"- `{item['path']}`: {item['status']}")

    lines.extend(["", "## Metric Comparison", ""])
    for method_name, method_report in report["methods"].items():
        lines.append(f"### {method_name}")
        for candidate_name, comparison in method_report["comparisons"].items():
            lines.append(
                f"- {candidate_name}: {comparison['wins_vs_raw']} wins, "
                f"{comparison['losses_vs_raw']} losses, {comparison['ties_vs_raw']} ties vs raw"
            )

    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])

    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "This verifies a TCL-v0 run artifact only. It does not validate the full TCL theory.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Run directory containing records, analysis, and review files.")
    parser.add_argument("--method", default="answer_mean")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--require-manual-review", action="store_true")
    parser.add_argument("--require-calibrated", action="store_true")
    parser.add_argument("--out-json")
    parser.add_argument("--out-md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    errors: list[str] = []
    warnings: list[str] = []
    files = []

    if not run_dir.exists():
        errors.append(f"Run directory does not exist: {run_dir}")

    records_path = run_dir / f"records_{args.method}.jsonl"
    config_path = run_dir / f"records_{args.method}.config.json"
    summary_path = run_dir / "analysis" / "summary.json"
    method_summary_path = run_dir / "analysis" / "method_summary.csv"
    method_dir = run_dir / "analysis" / args.method
    metrics_path = method_dir / "metrics.json"
    predictions_path = method_dir / "test_predictions.csv"
    raw_bins_path = method_dir / "raw_reliability_bins.csv"
    probe_bins_path = method_dir / "tcl_v0_reliability_bins.csv"
    conservative_bins_path = method_dir / "tcl_v0_conservative_reliability_bins.csv"
    calibrated_bins_path = method_dir / "tcl_v0_calibrated_reliability_bins.csv"
    manual_review_path = run_dir / "manual_review_all.csv"

    expected_files = [
        (records_path, "records", True),
        (config_path, "inference config", True),
        (summary_path, "analysis summary", True),
        (method_summary_path, "method summary", True),
        (metrics_path, "metrics", True),
        (predictions_path, "test predictions", True),
        (raw_bins_path, "raw reliability bins", True),
        (probe_bins_path, "probe reliability bins", True),
        (conservative_bins_path, "conservative reliability bins", True),
        (calibrated_bins_path, "calibrated reliability bins", args.require_calibrated),
        (manual_review_path, "manual review CSV", args.require_manual_review),
    ]
    for path, label, required in expected_files:
        exists = file_status(path, label, errors, warnings, required=required)
        files.append({"path": str(path), "status": "present" if exists else "missing", "required": required})

    records = load_jsonl(records_path) if records_path.exists() else []
    check_record_schema(records, errors, warnings)
    if len(records) < args.min_records:
        errors.append(f"Expected at least {args.min_records} records, found {len(records)}.")

    summary = load_json(summary_path) if summary_path.exists() else {}
    if summary.get("errors"):
        errors.append(f"Analysis summary contains probe errors: {summary['errors']}")

    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    if args.require_calibrated and "tcl_v0_calibrated_confidence" not in metrics:
        errors.append("Metrics are missing tcl_v0_calibrated_confidence.")

    test_prediction_count = check_predictions(predictions_path, errors, warnings) if predictions_path.exists() else 0

    methods = {}
    if metrics:
        methods[args.method] = {
            "n_records": metrics.get("n_records"),
            "n_train": metrics.get("n_train"),
            "n_validation": metrics.get("n_validation"),
            "n_test": metrics.get("n_test"),
            "split_mode": metrics.get("split_mode"),
            "calibration_status": metrics.get("calibration_status"),
            "comparisons": {
                "tcl_v0_probe_confidence": compare_against_raw(metrics, "tcl_v0_probe_confidence"),
                "tcl_v0_conservative_confidence": compare_against_raw(metrics, "tcl_v0_conservative_confidence"),
            },
        }
        if "tcl_v0_calibrated_confidence" in metrics:
            methods[args.method]["comparisons"]["tcl_v0_calibrated_confidence"] = compare_against_raw(
                metrics,
                "tcl_v0_calibrated_confidence",
            )

    report = {
        "status": "pass" if not errors else "fail",
        "run_dir": str(run_dir),
        "record_count": len(records),
        "test_prediction_count": test_prediction_count,
        "files": files,
        "methods": methods,
        "warnings": warnings,
        "errors": errors,
        "claim_boundary": "This verifies a TCL-v0 run artifact only. It does not validate the full TCL theory.",
    }

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    if args.out_md:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(build_markdown(report), encoding="utf-8")

    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()
