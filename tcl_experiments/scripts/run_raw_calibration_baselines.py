from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from recompute_reviewed_metrics import score_signal


SIGNAL_COLUMNS = [
    "raw_generation_confidence",
    "raw_temperature_scaled_confidence",
    "raw_platt_confidence",
    "raw_isotonic_confidence",
    "raw_platt_conservative_confidence",
    "raw_isotonic_conservative_confidence",
    "tcl_v0_probe_confidence",
    "tcl_v0_conservative_confidence",
    "tcl_v0_calibrated_confidence",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def clip_prob(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)


def logit(prob: np.ndarray) -> np.ndarray:
    prob = clip_prob(prob)
    return np.log(prob / (1.0 - prob))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y.astype(int), minlength=2)
    weights = np.ones(len(y), dtype=float)
    for label in (0, 1):
        if counts[label] > 0:
            weights[y == label] = len(y) / (2.0 * counts[label])
    return weights


def weighted_nll(y: np.ndarray, prob: np.ndarray, weights: np.ndarray) -> float:
    prob = clip_prob(prob)
    losses = -(y * np.log(prob) + (1 - y) * np.log(1.0 - prob))
    return float(np.average(losses, weights=weights))


def fit_temperature(raw: np.ndarray, y: np.ndarray, weighted: bool) -> dict:
    if len(set(y.tolist())) < 2:
        return {"status": "not_available_single_class", "temperature": None}
    weights = balanced_sample_weight(y) if weighted else np.ones(len(y), dtype=float)
    raw_logits = logit(raw)
    grid = np.logspace(-2, 2, 401)
    losses = [weighted_nll(y, sigmoid(raw_logits / t), weights) for t in grid]
    best_idx = int(np.argmin(losses))
    return {
        "status": "fitted",
        "temperature": float(grid[best_idx]),
        "weighted": bool(weighted),
        "objective": "balanced_nll" if weighted else "nll",
        "fit_nll": float(losses[best_idx]),
    }


def apply_temperature(raw: np.ndarray, temperature_info: dict) -> np.ndarray:
    if temperature_info.get("status") != "fitted":
        return raw
    return sigmoid(logit(raw) / float(temperature_info["temperature"]))


def fit_platt(raw: np.ndarray, y: np.ndarray, weighted: bool) -> dict | None:
    if len(set(y.tolist())) < 2:
        return None
    x = raw.astype(float)
    weights = balanced_sample_weight(y) if weighted else np.ones(len(y), dtype=float)
    coef = 0.0
    intercept = logit(np.asarray([max(min(float(np.average(y, weights=weights)), 1.0 - 1e-6), 1e-6)]))[0]
    lr = 0.2
    l2 = 1e-4
    weight_sum = float(weights.sum())
    for _ in range(5000):
        pred = sigmoid(coef * x + intercept)
        error = pred - y
        grad_coef = float((weights * error * x).sum() / weight_sum) + l2 * coef
        grad_intercept = float((weights * error).sum() / weight_sum)
        coef -= lr * grad_coef
        intercept -= lr * grad_intercept
    return {
        "coef": float(coef),
        "intercept": float(intercept),
        "l2": l2,
        "max_iter": 5000,
        "learning_rate": lr,
        "class_weight": "balanced" if weighted else None,
    }


def apply_platt(model: dict, raw: np.ndarray) -> np.ndarray:
    return sigmoid(float(model["coef"]) * raw + float(model["intercept"]))


def fit_isotonic(raw: np.ndarray, y: np.ndarray, weighted: bool) -> dict | None:
    if len(set(y.tolist())) < 2:
        return None
    weights = balanced_sample_weight(y) if weighted else np.ones(len(y), dtype=float)
    order = np.argsort(raw)
    x_sorted = raw[order]
    y_sorted = y[order].astype(float)
    w_sorted = weights[order]

    blocks = []
    for x_value, y_value, weight in zip(x_sorted, y_sorted, w_sorted):
        blocks.append({
            "x_min": float(x_value),
            "x_max": float(x_value),
            "weight": float(weight),
            "value": float(y_value),
        })
        while len(blocks) >= 2 and blocks[-2]["value"] > blocks[-1]["value"]:
            right = blocks.pop()
            left = blocks.pop()
            weight_sum = left["weight"] + right["weight"]
            value = (left["value"] * left["weight"] + right["value"] * right["weight"]) / weight_sum
            blocks.append({
                "x_min": left["x_min"],
                "x_max": right["x_max"],
                "weight": weight_sum,
                "value": float(value),
            })
    return {
        "thresholds": [block["x_max"] for block in blocks],
        "values": [min(max(block["value"], 0.0), 1.0) for block in blocks],
        "sample_weight": "balanced" if weighted else None,
    }


def apply_isotonic(model: dict, raw: np.ndarray) -> np.ndarray:
    thresholds = np.asarray(model["thresholds"], dtype=float)
    values = np.asarray(model["values"], dtype=float)
    indices = np.searchsorted(thresholds, raw, side="left")
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def choose_calibration_records(records: list[dict]) -> tuple[str, list[dict]]:
    validation = [r for r in records if r.get("split") == "validation"]
    if has_two_classes(validation):
        return "validation", validation
    train = [r for r in records if r.get("split") == "train"]
    if has_two_classes(train):
        return "train", train
    combined = [r for r in records if r.get("split") in {"train", "validation"}]
    return "train_validation_combined", combined


def has_two_classes(records: list[dict]) -> bool:
    labels = [parse_int(r.get("correctness_label", r.get("is_correct"))) for r in records]
    labels = [label for label in labels if label is not None]
    return len(set(labels)) >= 2


def records_to_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    raw = []
    labels = []
    for record in records:
        conf = parse_float(record.get("raw_generation_confidence"))
        label = parse_int(record.get("correctness_label", record.get("is_correct")))
        if conf is None or label is None:
            continue
        raw.append(conf)
        labels.append(label)
    return clip_prob(raw), np.asarray(labels, dtype=int)


def metrics_for_rows(rows: list[dict], signal: str, bins: int) -> tuple[dict, list[dict]] | None:
    y_true = [parse_int(row.get("correctness_label")) for row in rows]
    y_prob = [parse_float(row.get(signal)) for row in rows]
    if any(label is None for label in y_true) or any(prob is None for prob in y_prob):
        return None
    return score_signal([int(y) for y in y_true], [float(p) for p in y_prob], bins)


def comparison_vs_raw(raw_metrics: dict, candidate_metrics: dict) -> dict:
    lower_is_better = {"ece", "mce", "brier", "wrong_conf_ge_0_8", "wrong_conf_ge_0_9"}
    higher_is_better = {"accuracy_at_0_5", "auc"}
    result = {}
    for metric, raw_value in raw_metrics.items():
        if metric not in candidate_metrics:
            continue
        candidate_value = candidate_metrics[metric]
        if metric in lower_is_better:
            winner = "candidate" if candidate_value < raw_value else "raw" if raw_value < candidate_value else "tie"
            delta = raw_value - candidate_value
        elif metric in higher_is_better:
            winner = "candidate" if candidate_value > raw_value else "raw" if raw_value > candidate_value else "tie"
            delta = candidate_value - raw_value
        else:
            continue
        result[metric] = {
            "raw": raw_value,
            "candidate": candidate_value,
            "delta_positive_favors_candidate": delta,
            "winner": winner,
        }
    return result


def write_bins(out_dir: Path, bins_by_signal: dict[str, list[dict]]):
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["bin", "low", "high", "count", "accuracy", "confidence", "gap"]
    for signal, rows in bins_by_signal.items():
        path = out_dir / f"{signal}_reliability_bins.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", default="answer_mean")
    parser.add_argument("--records", default=None)
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--unweighted", action="store_true", help="Use unweighted raw-only calibrators instead of balanced calibration.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    records_path = Path(args.records) if args.records else run_dir / f"records_{args.method}.jsonl"
    predictions_path = Path(args.predictions) if args.predictions else run_dir / "reviewed_test_predictions.csv"
    if not predictions_path.exists():
        predictions_path = run_dir / "analysis" / args.method / "test_predictions.csv"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "baseline_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(records_path)
    calibration_split, calibration_records = choose_calibration_records(records)
    raw_cal, y_cal = records_to_xy(calibration_records)
    if len(raw_cal) == 0 or len(set(y_cal.tolist())) < 2:
        raise SystemExit(f"Calibration split '{calibration_split}' has insufficient class diversity.")

    weighted = not args.unweighted
    temperature_info = fit_temperature(raw_cal, y_cal, weighted=weighted)
    platt = fit_platt(raw_cal, y_cal, weighted=weighted)
    isotonic = fit_isotonic(raw_cal, y_cal, weighted=weighted)

    rows = load_csv(predictions_path)
    raw_test = clip_prob([parse_float(row["raw_generation_confidence"]) for row in rows])
    temp_prob = apply_temperature(raw_test, temperature_info)
    platt_prob = apply_platt(platt, raw_test) if platt is not None else raw_test
    isotonic_prob = apply_isotonic(isotonic, raw_test) if isotonic is not None else raw_test

    enriched_rows = []
    for row, p_temp, p_platt, p_iso in zip(rows, temp_prob, platt_prob, isotonic_prob):
        enriched = dict(row)
        raw = parse_float(enriched["raw_generation_confidence"])
        enriched["raw_temperature_scaled_confidence"] = float(p_temp)
        enriched["raw_platt_confidence"] = float(p_platt)
        enriched["raw_isotonic_confidence"] = float(p_iso)
        enriched["raw_platt_conservative_confidence"] = min(float(raw), float(p_platt))
        enriched["raw_isotonic_conservative_confidence"] = min(float(raw), float(p_iso))
        enriched_rows.append(enriched)

    fieldnames = list(enriched_rows[0].keys()) if enriched_rows else []
    predictions_out = out_dir / "baseline_test_predictions.csv"
    with predictions_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)

    metrics = {}
    bins_by_signal = {}
    for signal in SIGNAL_COLUMNS:
        scored = metrics_for_rows(enriched_rows, signal, args.bins)
        if scored is None:
            continue
        signal_metrics, bin_rows = scored
        metrics[signal] = signal_metrics
        bins_by_signal[signal] = bin_rows
    write_bins(out_dir / "reliability_bins", bins_by_signal)

    raw_metrics = metrics["raw_generation_confidence"]
    comparisons = {
        signal: comparison_vs_raw(raw_metrics, signal_metrics)
        for signal, signal_metrics in metrics.items()
        if signal != "raw_generation_confidence"
    }

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "records_path": str(records_path),
        "predictions_path": str(predictions_path),
        "out_dir": str(out_dir),
        "method": args.method,
        "bins": args.bins,
        "calibration_split": calibration_split,
        "calibration_size": int(len(y_cal)),
        "calibration_positive_count": int(y_cal.sum()),
        "calibration_positive_rate": float(y_cal.mean()),
        "weighted_calibrators": weighted,
        "temperature_scaling": temperature_info,
        "platt": {
            "status": "fitted" if platt is not None else "not_available",
            "input": "raw_generation_confidence",
            "class_weight": "balanced" if weighted else None,
            "implementation": "local_numpy_1d_logistic_regression",
            "coef": None if platt is None else platt["coef"],
            "intercept": None if platt is None else platt["intercept"],
            "l2": None if platt is None else platt["l2"],
            "max_iter": None if platt is None else platt["max_iter"],
        },
        "isotonic": {
            "status": "fitted" if isotonic is not None else "not_available",
            "input": "raw_generation_confidence",
            "sample_weight": "balanced" if weighted else None,
            "implementation": "local_numpy_pava",
            "n_blocks": None if isotonic is None else len(isotonic["values"]),
        },
        "signals": metrics,
        "comparisons_vs_raw": comparisons,
        "claim_boundary": "Raw-only baseline metrics for TCL-v0 diagnostics. These do not validate full TCL.",
    }
    with (out_dir / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_rows = []
    for signal, signal_metrics in metrics.items():
        row = {"signal": signal}
        row.update(signal_metrics)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "baseline_method_summary.csv", index=False)

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
