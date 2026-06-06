from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SIGNAL_COLUMNS = [
    "raw_generation_confidence",
    "tcl_v0_probe_confidence",
    "tcl_v0_conservative_confidence",
    "tcl_v0_calibrated_confidence",
]


def load_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_int(value):
    if value in (None, ""):
        return None
    return int(float(value))


def parse_float(value):
    if value in (None, ""):
        return None
    return float(value)


def reliability_bins(y_true: list[int], y_prob: list[float], n_bins: int):
    rows = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        if i == n_bins - 1:
            indices = [j for j, prob in enumerate(y_prob) if low <= prob <= high]
        else:
            indices = [j for j, prob in enumerate(y_prob) if low <= prob < high]
        count = len(indices)
        if count == 0:
            rows.append({"bin": i, "low": low, "high": high, "count": 0, "accuracy": None, "confidence": None, "gap": None})
            continue
        accuracy = sum(y_true[j] for j in indices) / count
        confidence = sum(y_prob[j] for j in indices) / count
        gap = abs(accuracy - confidence)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append({"bin": i, "low": low, "high": high, "count": count, "accuracy": accuracy, "confidence": confidence, "gap": gap})
    return ece, mce, rows


def roc_auc(y_true: list[int], y_prob: list[float]):
    positives = [score for label, score in zip(y_true, y_prob) if label == 1]
    negatives = [score for label, score in zip(y_true, y_prob) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def score_signal(y_true: list[int], y_prob: list[float], bins: int):
    ece, mce, bin_rows = reliability_bins(y_true, y_prob, bins)
    n = len(y_true)
    metrics = {
        "accuracy_at_0_5": sum(int(label == int(prob >= 0.5)) for label, prob in zip(y_true, y_prob)) / n,
        "ece": ece,
        "mce": mce,
        "brier": sum((prob - label) ** 2 for label, prob in zip(y_true, y_prob)) / n,
        "wrong_conf_ge_0_8": sum(1 for label, prob in zip(y_true, y_prob) if label == 0 and prob >= 0.8),
        "wrong_conf_ge_0_9": sum(1 for label, prob in zip(y_true, y_prob) if label == 0 and prob >= 0.9),
    }
    auc = roc_auc(y_true, y_prob)
    if auc is not None:
        metrics["auc"] = auc
    return metrics, bin_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out-predictions", required=True)
    parser.add_argument("--out-metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    prediction_rows = load_csv_rows(Path(args.predictions))
    review_rows = load_csv_rows(Path(args.review_csv))
    manual_by_id = {
        row["id"]: parse_int(row.get("manual_correctness_label"))
        for row in review_rows
        if row.get("manual_correctness_label", "") != ""
    }

    reviewed_rows = []
    label_changes = 0
    for row in prediction_rows:
        reviewed = dict(row)
        auto_label = parse_int(row.get("correctness_label"))
        manual_label = manual_by_id.get(row["id"], auto_label)
        reviewed["auto_correctness_label"] = auto_label
        reviewed["correctness_label"] = manual_label
        reviewed["review_label_source"] = "manual" if row["id"] in manual_by_id else "auto"
        if manual_label != auto_label:
            label_changes += 1
        reviewed_rows.append(reviewed)

    out_predictions = Path(args.out_predictions)
    fieldnames = list(reviewed_rows[0].keys()) if reviewed_rows else []
    write_csv(out_predictions, reviewed_rows, fieldnames)

    y_true = [parse_int(row["correctness_label"]) for row in reviewed_rows]
    signals = {}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for signal in SIGNAL_COLUMNS:
        values = [parse_float(row.get(signal)) for row in reviewed_rows]
        if any(value is None for value in values):
            continue
        metrics, bin_rows = score_signal(y_true, values, args.bins)
        signals[signal] = metrics
        write_csv(out_dir / f"reviewed_{signal}_reliability_bins.csv", bin_rows, [
            "bin", "low", "high", "count", "accuracy", "confidence", "gap",
        ])

    report = {
        "n_test": len(reviewed_rows),
        "auto_positive_count": sum(parse_int(row.get("auto_correctness_label")) for row in reviewed_rows),
        "reviewed_positive_count": sum(y_true),
        "manually_reviewed_count": len(manual_by_id),
        "label_changes": label_changes,
        "signals": signals,
        "claim_boundary": "These are reviewed-label test metrics for TCL-v0 diagnostics only; they do not validate full TCL.",
    }

    out_metrics = Path(args.out_metrics)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
