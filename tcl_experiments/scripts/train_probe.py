from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from metrics import brier_score, expected_calibration_error


ROOT = Path(__file__).resolve().parents[1]


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def record_value(record, new_key: str, old_key: str | None = None):
    if new_key in record:
        return record[new_key]
    if old_key and old_key in record:
        return record[old_key]
    raise KeyError(f"Record is missing required key: {new_key}")


def split_indices(records, y, test_size: float, split_mode: str):
    if split_mode == "record":
        train_idx = np.asarray([i for i, r in enumerate(records) if r.get("split") == "train"], dtype=int)
        test_idx = np.asarray([i for i, r in enumerate(records) if r.get("split") == "test"], dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Record split mode requires records with both 'train' and 'test' split values.")
        val_idx = np.asarray([i for i, r in enumerate(records) if r.get("split") == "validation"], dtype=int)
        return train_idx, val_idx, test_idx

    stratify = y if min(np.bincount(y)) >= 2 else None
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    return train_idx, np.asarray([], dtype=int), test_idx


def score_metrics(y_test, y_prob, bins: int):
    ece, mce, bin_rows = expected_calibration_error(y_test, y_prob, bins)
    metrics = {
        "accuracy_at_0_5": float(accuracy_score(y_test, y_prob >= 0.5)),
        "ece": ece,
        "mce": mce,
        "brier": brier_score(y_test, y_prob),
        "wrong_conf_ge_0_8": int(((y_test == 0) & (y_prob >= 0.8)).sum()),
        "wrong_conf_ge_0_9": int(((y_test == 0) & (y_prob >= 0.9)).sum()),
    }
    if len(set(y_test.tolist())) > 1:
        metrics["auc"] = float(roc_auc_score(y_test, y_prob))
    return metrics, bin_rows


def fit_score_calibrator(val_scores, y_val):
    if len(val_scores) == 0 or len(set(y_val.tolist())) < 2:
        return None
    calibrator = LogisticRegression(max_iter=1000, class_weight="balanced")
    calibrator.fit(val_scores.reshape(-1, 1), y_val)
    return calibrator


def evaluate_group(records, method_name: str, out_dir: Path, test_size: float, bins: int, split_mode: str, calibrate: bool):
    if len(records) < 8:
        raise ValueError(f"Need at least 8 records for method '{method_name}'. Found {len(records)}.")

    X = np.asarray([record_value(r, "hidden_state_vector", "hidden_state") for r in records], dtype=float)
    y = np.asarray([record_value(r, "correctness_label", "is_correct") for r in records], dtype=int)
    raw_conf = np.asarray([r["raw_generation_confidence"] for r in records], dtype=float)

    if len(set(y.tolist())) < 2:
        raise ValueError(
            f"All examples have the same correctness label for method '{method_name}'. "
            "Use a stronger/weaker model or more questions so the probe has both correct and incorrect examples."
        )

    train_idx, val_idx, test_idx = split_indices(records, y, test_size, split_mode)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx]) if len(val_idx) else None
    X_test = scaler.transform(X[test_idx])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y[train_idx])
    probe_prob = clf.predict_proba(X_test)[:, 1]
    probe_scores = clf.decision_function(X_test)
    raw_prob = np.clip(raw_conf[test_idx], 0.0, 1.0)
    conservative_prob = np.minimum(raw_prob, probe_prob)
    y_test = y[test_idx]

    raw_metrics, raw_bins = score_metrics(y_test, raw_prob, bins)
    probe_metrics, probe_bins = score_metrics(y_test, probe_prob, bins)
    conservative_metrics, conservative_bins = score_metrics(y_test, conservative_prob, bins)

    calibrated_prob = None
    calibrated_bins = None
    calibrated_metrics = None
    calibration_status = "not_requested"
    if calibrate:
        calibration_status = "not_available"
        if X_val is not None:
            y_val = y[val_idx]
            val_scores = clf.decision_function(X_val)
            calibrator = fit_score_calibrator(val_scores, y_val)
            if calibrator is not None:
                calibrated_prob = calibrator.predict_proba(probe_scores.reshape(-1, 1))[:, 1]
                calibrated_metrics, calibrated_bins = score_metrics(y_test, calibrated_prob, bins)
                calibration_status = "fitted_on_validation_scores"

    metrics = {
        "hidden_state_method": method_name,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "n_records": int(len(records)),
        "n_train": int(len(train_idx)),
        "n_validation": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "split_mode": split_mode,
        "calibration_status": calibration_status,
        "positive_rate_all": float(y.mean()),
        "raw_generation_confidence": raw_metrics,
        "tcl_v0_probe_confidence": probe_metrics,
        "tcl_v0_conservative_confidence": conservative_metrics,
    }
    if calibrated_metrics is not None:
        metrics["tcl_v0_calibrated_confidence"] = calibrated_metrics

    method_dir = out_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    with (method_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(raw_bins).to_csv(method_dir / "raw_reliability_bins.csv", index=False)
    pd.DataFrame(probe_bins).to_csv(method_dir / "tcl_v0_reliability_bins.csv", index=False)
    pd.DataFrame(conservative_bins).to_csv(method_dir / "tcl_v0_conservative_reliability_bins.csv", index=False)
    if calibrated_bins is not None:
        pd.DataFrame(calibrated_bins).to_csv(method_dir / "tcl_v0_calibrated_reliability_bins.csv", index=False)

    comparison = []
    calibrated_iter = calibrated_prob if calibrated_prob is not None else [None] * len(test_idx)
    for i, p_raw, p_probe, p_conservative, p_calibrated in zip(
        test_idx,
        raw_prob,
        probe_prob,
        conservative_prob,
        calibrated_iter,
    ):
        r = records[int(i)]
        comparison.append({
            "id": r["id"],
            "dataset": r.get("dataset"),
            "question": r["question"],
            "context": r.get("context", ""),
            "model_name": r.get("model_name", r.get("model")),
            "model_answer": r["model_answer"],
            "correctness_label": int(record_value(r, "correctness_label", "is_correct")),
            "raw_generation_confidence": float(p_raw),
            "tcl_v0_probe_confidence": float(p_probe),
            "tcl_v0_conservative_confidence": float(p_conservative),
            "tcl_v0_calibrated_confidence": None if p_calibrated is None else float(p_calibrated),
            "hidden_state_method": method_name,
            "split": "test",
            "run_id": r.get("run_id"),
        })
    pd.DataFrame(comparison).to_csv(method_dir / "test_predictions.csv", index=False)

    split_rows = []
    for i in train_idx:
        r = records[int(i)]
        split_rows.append({"id": r["id"], "split": "train", "hidden_state_method": method_name})
    for i in val_idx:
        r = records[int(i)]
        split_rows.append({"id": r["id"], "split": "validation", "hidden_state_method": method_name})
    for i in test_idx:
        r = records[int(i)]
        split_rows.append({"id": r["id"], "split": "test", "hidden_state_method": method_name})
    pd.DataFrame(split_rows).to_csv(method_dir / "splits.csv", index=False)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(ROOT / "runs" / "records.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "runs" / "analysis"))
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--split-mode",
        choices=["random", "record"],
        default="random",
        help="Use a random split or train/test split labels recorded in the records.",
    )
    parser.add_argument(
        "--hidden-state-method",
        default="all",
        help="Analyze one hidden-state method, or 'all' to analyze each method found in the records.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit a 1D logistic calibrator on validation-split probe scores and evaluate on test.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(Path(args.records))
    if not records:
        raise SystemExit("No records found.")

    grouped = {}
    for record in records:
        method = record.get("hidden_state_method", "legacy_unknown")
        grouped.setdefault(method, []).append(record)

    if args.hidden_state_method != "all":
        grouped = {
            args.hidden_state_method: grouped.get(args.hidden_state_method, [])
        }

    all_metrics = []
    errors = []
    for method_name, method_records in grouped.items():
        try:
            all_metrics.append(
                evaluate_group(
                    method_records,
                    method_name=method_name,
                    out_dir=out_dir,
                    test_size=args.test_size,
                    bins=args.bins,
                    split_mode=args.split_mode,
                    calibrate=args.calibrate,
                )
            )
        except ValueError as exc:
            errors.append({"hidden_state_method": method_name, "error": str(exc)})

    summary = {
        "records_path": str(Path(args.records)),
        "out_dir": str(out_dir),
        "test_size": args.test_size,
        "split_mode": args.split_mode,
        "calibrate": args.calibrate,
        "bins": args.bins,
        "metrics": all_metrics,
        "errors": errors,
        "claim_boundary": "These metrics are TCL-v0 probe diagnostics only. They do not validate the full TCL framework.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if all_metrics:
        pd.DataFrame([
            {
                "hidden_state_method": m["hidden_state_method"],
                "n_records": m["n_records"],
                "n_train": m["n_train"],
                "n_validation": m["n_validation"],
                "n_test": m["n_test"],
                "positive_rate_all": m["positive_rate_all"],
                "split_mode": m["split_mode"],
                "calibration_status": m["calibration_status"],
                "raw_ece": m["raw_generation_confidence"]["ece"],
                "probe_ece": m["tcl_v0_probe_confidence"]["ece"],
                "conservative_ece": m["tcl_v0_conservative_confidence"]["ece"],
                "calibrated_ece": m.get("tcl_v0_calibrated_confidence", {}).get("ece"),
                "raw_brier": m["raw_generation_confidence"]["brier"],
                "probe_brier": m["tcl_v0_probe_confidence"]["brier"],
                "conservative_brier": m["tcl_v0_conservative_confidence"]["brier"],
                "calibrated_brier": m.get("tcl_v0_calibrated_confidence", {}).get("brier"),
                "raw_mce": m["raw_generation_confidence"]["mce"],
                "probe_mce": m["tcl_v0_probe_confidence"]["mce"],
                "conservative_mce": m["tcl_v0_conservative_confidence"]["mce"],
                "calibrated_mce": m.get("tcl_v0_calibrated_confidence", {}).get("mce"),
                "raw_wrong_conf_ge_0_8": m["raw_generation_confidence"]["wrong_conf_ge_0_8"],
                "probe_wrong_conf_ge_0_8": m["tcl_v0_probe_confidence"]["wrong_conf_ge_0_8"],
                "conservative_wrong_conf_ge_0_8": m["tcl_v0_conservative_confidence"]["wrong_conf_ge_0_8"],
                "calibrated_wrong_conf_ge_0_8": m.get("tcl_v0_calibrated_confidence", {}).get("wrong_conf_ge_0_8"),
            }
            for m in all_metrics
        ]).to_csv(out_dir / "method_summary.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
