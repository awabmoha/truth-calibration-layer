from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from recompute_reviewed_metrics import score_signal
from run_raw_calibration_baselines import (
    apply_isotonic,
    apply_platt,
    apply_temperature,
    choose_calibration_records,
    clip_prob,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    parse_float,
    parse_int,
    records_to_xy,
)
from train_probe import record_value, split_indices


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return records


def group_key(record: dict) -> tuple[str, str]:
    method = record.get("hidden_state_method", "unknown")
    layer = str(record.get("hidden_state_layer", "unknown"))
    return method, layer


def score_array(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> dict:
    metrics, _ = score_signal(y_true.astype(int).tolist(), y_prob.astype(float).tolist(), bins)
    return metrics


def fit_logistic(X_train: np.ndarray, y_train: np.ndarray, class_weight):
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight)
    clf.fit(X_train, y_train)
    return clf


def evaluate_group(records: list[dict], method: str, layer: str, bins: int, split_mode: str, weighted: bool) -> dict:
    X_hidden = np.asarray([record_value(r, "hidden_state_vector", "hidden_state") for r in records], dtype=float)
    y = np.asarray([record_value(r, "correctness_label", "is_correct") for r in records], dtype=int)
    raw = clip_prob([r["raw_generation_confidence"] for r in records])

    if len(records) < 8:
        raise ValueError(f"Need at least 8 records for {method}/layer={layer}.")
    if len(set(y.tolist())) < 2:
        raise ValueError(f"Need at least 2 classes for {method}/layer={layer}.")

    train_idx, val_idx, test_idx = split_indices(records, y, test_size=0.35, split_mode=split_mode)
    if len(set(y[train_idx].tolist())) < 2:
        raise ValueError(f"Train split lacks two classes for {method}/layer={layer}.")

    class_weight = "balanced" if weighted else None
    y_test = y[test_idx]
    raw_test = raw[test_idx]

    calibration_split, calibration_records = choose_calibration_records(records)
    raw_cal, y_cal = records_to_xy(calibration_records)
    if len(set(y_cal.tolist())) < 2:
        raw_cal = raw[train_idx]
        y_cal = y[train_idx]
        calibration_split = "train_fallback"

    temp_info = fit_temperature(raw_cal, y_cal, weighted=weighted)
    platt = fit_platt(raw_cal, y_cal, weighted=weighted)
    isotonic = fit_isotonic(raw_cal, y_cal, weighted=weighted)
    raw_temp = apply_temperature(raw_test, temp_info)
    raw_platt = apply_platt(platt, raw_test) if platt is not None else raw_test
    raw_isotonic = apply_isotonic(isotonic, raw_test) if isotonic is not None else raw_test
    raw_platt_cons = np.minimum(raw_test, raw_platt)
    raw_isotonic_cons = np.minimum(raw_test, raw_isotonic)

    scaler_hidden = StandardScaler()
    Xh_train = scaler_hidden.fit_transform(X_hidden[train_idx])
    Xh_test = scaler_hidden.transform(X_hidden[test_idx])
    hidden_clf = fit_logistic(Xh_train, y[train_idx], class_weight=class_weight)
    hidden_prob = hidden_clf.predict_proba(Xh_test)[:, 1]
    hidden_conservative = np.minimum(raw_test, hidden_prob)

    X_raw_hidden = np.column_stack([X_hidden, raw])
    scaler_raw_hidden = StandardScaler()
    Xrh_train = scaler_raw_hidden.fit_transform(X_raw_hidden[train_idx])
    Xrh_test = scaler_raw_hidden.transform(X_raw_hidden[test_idx])
    raw_hidden_clf = fit_logistic(Xrh_train, y[train_idx], class_weight=class_weight)
    raw_hidden_prob = raw_hidden_clf.predict_proba(Xrh_test)[:, 1]
    raw_hidden_conservative = np.minimum(raw_test, raw_hidden_prob)

    probe_train_score = hidden_clf.decision_function(Xh_train)
    probe_test_score = hidden_clf.decision_function(Xh_test)
    X_score_raw_train = np.column_stack([probe_train_score, raw[train_idx]])
    X_score_raw_test = np.column_stack([probe_test_score, raw_test])
    scaler_score_raw = StandardScaler()
    Xsr_train = scaler_score_raw.fit_transform(X_score_raw_train)
    Xsr_test = scaler_score_raw.transform(X_score_raw_test)
    score_raw_clf = fit_logistic(Xsr_train, y[train_idx], class_weight=class_weight)
    score_raw_prob = score_raw_clf.predict_proba(Xsr_test)[:, 1]

    signals = {
        "raw_generation_confidence": raw_test,
        "raw_temperature_scaled_confidence": raw_temp,
        "raw_platt_confidence": raw_platt,
        "raw_isotonic_confidence": raw_isotonic,
        "raw_platt_conservative_confidence": raw_platt_cons,
        "raw_isotonic_conservative_confidence": raw_isotonic_cons,
        "hidden_only_probe_confidence": hidden_prob,
        "hidden_conservative_confidence": hidden_conservative,
        "raw_plus_hidden_confidence": raw_hidden_prob,
        "raw_plus_hidden_conservative_confidence": raw_hidden_conservative,
        "probe_score_plus_raw_confidence": score_raw_prob,
    }
    metrics = {name: score_array(y_test, values, bins) for name, values in signals.items()}

    rows = []
    for local_i, record_i in enumerate(test_idx):
        record = records[int(record_i)]
        row = {
            "id": record.get("id"),
            "dataset": record.get("dataset"),
            "question": record.get("question"),
            "model_name": record.get("model_name", record.get("model")),
            "correctness_label": int(y_test[local_i]),
            "hidden_state_method": method,
            "hidden_state_layer": layer,
        }
        for name, values in signals.items():
            row[name] = float(values[local_i])
        rows.append(row)

    return {
        "hidden_state_method": method,
        "hidden_state_layer": layer,
        "n_records": int(len(records)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "positive_rate_all": float(y.mean()),
        "positive_rate_test": float(y_test.mean()),
        "test_has_two_classes": bool(len(set(y_test.tolist())) >= 2),
        "split_mode": split_mode,
        "weighted": weighted,
        "calibration_split": calibration_split,
        "feature_count": int(X_hidden.shape[1]),
        "p_greater_than_n_train": bool(X_hidden.shape[1] > len(train_idx)),
        "signals": metrics,
        "test_predictions": rows,
        "probe_models": {
            "hidden_only": {
                "type": "sklearn.linear_model.LogisticRegression",
                "C": float(hidden_clf.C),
                "class_weight": hidden_clf.class_weight,
                "solver": hidden_clf.solver,
                "max_iter": int(hidden_clf.max_iter),
            },
            "raw_plus_hidden": {
                "type": "sklearn.linear_model.LogisticRegression",
                "C": float(raw_hidden_clf.C),
                "class_weight": raw_hidden_clf.class_weight,
                "solver": raw_hidden_clf.solver,
                "max_iter": int(raw_hidden_clf.max_iter),
            },
            "probe_score_plus_raw": {
                "type": "sklearn.linear_model.LogisticRegression",
                "C": float(score_raw_clf.C),
                "class_weight": score_raw_clf.class_weight,
                "solver": score_raw_clf.solver,
                "max_iter": int(score_raw_clf.max_iter),
            },
        },
    }


def flatten_summary(report: dict) -> list[dict]:
    rows = []
    for result in report["results"]:
        for signal, metrics in result["signals"].items():
            row = {
                "hidden_state_method": result["hidden_state_method"],
                "hidden_state_layer": result["hidden_state_layer"],
                "signal": signal,
                "n_train": result["n_train"],
                "n_validation": result["n_validation"],
                "n_test": result["n_test"],
                "positive_rate_test": result["positive_rate_test"],
                "test_has_two_classes": result["test_has_two_classes"],
                "weighted": result["weighted"],
                "feature_count": result["feature_count"],
                "p_greater_than_n_train": result["p_greater_than_n_train"],
            }
            row.update(metrics)
            rows.append(row)
    return rows


def write_predictions(out_dir: Path, results: list[dict]) -> None:
    pred_dir = out_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        safe_method = str(result["hidden_state_method"]).replace("/", "_")
        safe_layer = str(result["hidden_state_layer"]).replace("/", "_")
        rows = result["test_predictions"]
        if not rows:
            continue
        path = pred_dir / f"{safe_method}_layer-{safe_layer}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True, help="One or more records JSONL files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split-mode", choices=["record", "random"], default="record")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--unweighted", action="store_true")
    args = parser.parse_args()

    records = load_records([Path(path) for path in args.records])
    grouped: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        grouped.setdefault(group_key(record), []).append(record)

    results = []
    errors = []
    for (method, layer), group_records in sorted(grouped.items()):
        try:
            result = evaluate_group(
                group_records,
                method=method,
                layer=layer,
                bins=args.bins,
                split_mode=args.split_mode,
                weighted=not args.unweighted,
            )
            results.append(result)
        except Exception as exc:
            errors.append({
                "hidden_state_method": method,
                "hidden_state_layer": layer,
                "error": str(exc),
            })

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": args.records,
        "out_dir": args.out_dir,
        "split_mode": args.split_mode,
        "bins": args.bins,
        "weighted": not args.unweighted,
        "results": [
            {key: value for key, value in result.items() if key != "test_predictions"}
            for result in results
        ],
        "errors": errors,
        "claim_boundary": "Ablation diagnostics for TCL-v0 only. These do not validate full TCL.",
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "ablation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    pd.DataFrame(flatten_summary(report)).to_csv(out_dir / "ablation_summary.csv", index=False)
    write_predictions(out_dir, results)

    print(json.dumps(report, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
