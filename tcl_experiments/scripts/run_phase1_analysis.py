"""Phase 1 local analysis: tests H1 and H2 on the six reviewed TCL-v0 runs.

The protocol is fixed in TCL-Phase1-Protocol.md at the repository root. This
script implements it exactly: unweighted fits, validation-only model selection,
reviewed test labels, paired bootstrap CIs, and predeclared decision rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"

RUNS = {
    ("qwen", "squad"): "freegpu-squad1000-Qwen_Qwen2.5-0.5B-Instruct-answermean",
    ("qwen", "triviaqa"): "freegpu-triviaqa1000-Qwen_Qwen2.5-0.5B-Instruct-answermean",
    ("phi", "squad"): "freegpu-squad1000-microsoft_Phi-3.5-mini-instruct-answermean",
    ("phi", "triviaqa"): "freegpu-triviaqa1000-microsoft_Phi-3.5-mini-instruct-answermean",
    ("gemma", "squad"): "freegpu-squad1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean",
    ("gemma", "triviaqa"): "freegpu-triviaqa1000-_kaggle_input_models_google_gemma_transformers_2b-it_3-answermean",
}
MODELS = ["qwen", "phi", "gemma"]
DATASETS = ["squad", "triviaqa"]

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
SEEDS = [0, 1, 2]
PRIMARY_SEED = 0
N_BOOT = 1000
BOOT_SEED = 2026
ECE_BINS = 15
EPS = 1e-6

H1_MIN_RUNS = 4
H2_RETENTION = 0.70
H2_MIN_DIRECTIONS = 4

METRICS = ["auroc", "aurc", "acc_at_50", "acc_at_80", "ece", "brier"]
LOWER_IS_BETTER = {"aurc", "ece", "brier"}


# ---------------------------------------------------------------- data


def load_run(folder: str) -> dict:
    run_dir = RUNS_DIR / folder
    reviewed = {}
    with (run_dir / "reviewed_test_predictions.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviewed[row["id"]] = int(float(row["correctness_label"]))

    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    with (run_dir / "records_answer_mean.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            split_rows[r["split"]].append(r)

    data = {}
    for split, rows in split_rows.items():
        if split == "test":
            missing = [r["id"] for r in rows if r["id"] not in reviewed]
            if missing:
                raise ValueError(f"{folder}: {len(missing)} test rows lack reviewed labels")
            y = np.asarray([reviewed[r["id"]] for r in rows], dtype=int)
        else:
            y = np.asarray([int(r["correctness_label"]) for r in rows], dtype=int)
        data[split] = {
            "ids": [r["id"] for r in rows],
            "X": np.asarray([r["hidden_state_vector"] for r in rows], dtype=np.float64),
            "raw": np.clip(np.asarray([r["raw_generation_confidence"] for r in rows], dtype=float), EPS, 1 - EPS),
            "y": y,
        }
    auto_test = {r["id"]: int(r["correctness_label"]) for r in split_rows["test"]}
    data["n_test_label_changes"] = int(sum(auto_test[i] != reviewed[i] for i in auto_test))
    return data


# ---------------------------------------------------------------- calibrators


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(raw: np.ndarray, y: np.ndarray):
    z = logit(raw)

    def nll(log_t: float) -> float:
        return log_loss(y, np.clip(sigmoid(z / np.exp(log_t)), EPS, 1 - EPS), labels=[0, 1])

    t = float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))
    return lambda r: sigmoid(logit(r) / t)


def fit_platt(score: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(score.reshape(-1, 1), y)
    return lambda s: lr.predict_proba(np.asarray(s).reshape(-1, 1))[:, 1]


def fit_platt_on_raw(raw: np.ndarray, y: np.ndarray):
    f = fit_platt(logit(raw), y)
    return lambda r: f(logit(r))


def fit_isotonic(raw: np.ndarray, y: np.ndarray):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw, y)
    return lambda r: np.asarray(iso.predict(r), dtype=float)


RAW_CALIBRATORS = {
    "temperature": fit_temperature,
    "platt": fit_platt_on_raw,
    "isotonic": fit_isotonic,
}


def select_raw_calibrator(raw_val: np.ndarray, y_val: np.ndarray, seed: int):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_brier = {}
    for name, fit in RAW_CALIBRATORS.items():
        scores = []
        for tr, te in skf.split(raw_val.reshape(-1, 1), y_val):
            f = fit(raw_val[tr], y_val[tr])
            scores.append(float(np.mean((f(raw_val[te]) - y_val[te]) ** 2)))
        cv_brier[name] = float(np.mean(scores))
    best = min(cv_brier, key=cv_brier.get)
    return best, RAW_CALIBRATORS[best](raw_val, y_val), cv_brier


# ---------------------------------------------------------------- probe


def fit_probe(train: dict, val: dict, seed: int):
    scaler = StandardScaler().fit(train["X"])
    Xtr = scaler.transform(train["X"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_loss = {}
    for c in C_GRID:
        losses = []
        for tr, te in skf.split(Xtr, train["y"]):
            lr = LogisticRegression(C=c, max_iter=5000)
            lr.fit(Xtr[tr], train["y"][tr])
            losses.append(log_loss(train["y"][te], lr.predict_proba(Xtr[te])[:, 1], labels=[0, 1]))
        cv_loss[c] = float(np.mean(losses))
    best_c = min(cv_loss, key=cv_loss.get)
    clf = LogisticRegression(C=best_c, max_iter=5000).fit(Xtr, train["y"])

    def score(X: np.ndarray) -> np.ndarray:
        return clf.decision_function(scaler.transform(X))

    platt = fit_platt(score(val["X"]), val["y"])
    return {
        "score": score,
        "uncalibrated": lambda X: sigmoid(score(X)),
        "calibrated": lambda X: platt(score(X)),
        "C": best_c,
        "cv_logloss": cv_loss,
    }


# ---------------------------------------------------------------- metrics


def aurc(y: np.ndarray, conf: np.ndarray) -> float:
    order = np.argsort(-conf, kind="stable")
    errors = 1 - y[order]
    return float(np.mean(np.cumsum(errors) / np.arange(1, len(y) + 1)))


def acc_at_coverage(y: np.ndarray, conf: np.ndarray, coverage: float) -> float:
    k = int(np.ceil(coverage * len(y)))
    order = np.argsort(-conf, kind="stable")
    return float(np.mean(y[order[:k]]))


def ece_equal_mass(y: np.ndarray, conf: np.ndarray, bins: int = ECE_BINS) -> float:
    order = np.argsort(conf, kind="stable")
    total = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            total += len(chunk) * abs(float(np.mean(y[chunk])) - float(np.mean(conf[chunk])))
    return total / len(y)


def score_all(y: np.ndarray, conf: np.ndarray) -> dict:
    two_classes = len(np.unique(y)) == 2
    return {
        "auroc": float(roc_auc_score(y, conf)) if two_classes else float("nan"),
        "aurc": aurc(y, conf),
        "acc_at_50": acc_at_coverage(y, conf, 0.5),
        "acc_at_80": acc_at_coverage(y, conf, 0.8),
        "ece": ece_equal_mass(y, conf),
        "brier": float(np.mean((conf - y) ** 2)),
    }


def paired_bootstrap(y: np.ndarray, signals: dict, pairs: list[tuple[str, str]], seed: int = BOOT_SEED) -> dict:
    """Return point estimates, per-method CIs, and CIs of pairwise differences."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: {k: [] for k in METRICS} for m in signals}
    skipped = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        for m, conf in signals.items():
            s = score_all(yb, conf[idx])
            for k in METRICS:
                draws[m][k].append(s[k])

    def ci(values):
        arr = np.asarray(values)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    out = {"skipped_one_class": skipped, "methods": {}, "differences": {}}
    for m, conf in signals.items():
        point = score_all(y, conf)
        out["methods"][m] = {k: {"value": point[k], "ci": ci(draws[m][k])} for k in METRICS}
    for a, b in pairs:
        point_a, point_b = score_all(y, signals[a]), score_all(y, signals[b])
        out["differences"][f"{a}-{b}"] = {
            k: {
                "value": point_a[k] - point_b[k],
                "ci": ci(np.asarray(draws[a][k]) - np.asarray(draws[b][k])),
            }
            for k in METRICS
        }
    return out


# ---------------------------------------------------------------- analyses


def in_distribution(data: dict, seed: int) -> dict:
    train, val, test = data["train"], data["validation"], data["test"]
    cal_name, cal_fn, cal_cv = select_raw_calibrator(val["raw"], val["y"], seed)
    probe = fit_probe(train, val, seed)
    signals = {
        "raw": test["raw"],
        "calibrated_raw": np.clip(cal_fn(test["raw"]), 0, 1),
        "probe": probe["calibrated"](test["X"]),
        "probe_uncalibrated": probe["uncalibrated"](test["X"]),
    }
    signals["min_raw_probe"] = np.minimum(signals["raw"], signals["probe_uncalibrated"])
    signals["min_raw_calibrated_raw"] = np.minimum(signals["raw"], signals["calibrated_raw"])
    boot = paired_bootstrap(
        test["y"], signals, pairs=[("probe", "calibrated_raw"), ("probe", "raw"), ("min_raw_probe", "min_raw_calibrated_raw")]
    )
    return {
        "seed": seed,
        "raw_calibrator": cal_name,
        "raw_calibrator_cv_brier": cal_cv,
        "probe_C": probe["C"],
        "probe_cv_logloss": {str(k): v for k, v in probe["cv_logloss"].items()},
        "n": {s: int(len(data[s]["y"])) for s in ("train", "validation", "test")},
        "positive_rate_test": float(test["y"].mean()),
        "bootstrap": boot,
        "_signals": signals,
        "_probe": probe,
    }


def transfer(source: dict, target: dict, target_probe_auroc: float, seed: int) -> dict:
    probe = fit_probe(source["train"], source["validation"], seed)
    t = target["test"]
    signals = {"raw": t["raw"], "transfer_probe": probe["calibrated"](t["X"])}
    boot = paired_bootstrap(t["y"], signals, pairs=[("transfer_probe", "raw")])
    raw_auroc = boot["methods"]["raw"]["auroc"]["value"]
    transfer_auroc = boot["methods"]["transfer_probe"]["auroc"]["value"]
    in_gain = target_probe_auroc - raw_auroc
    transfer_gain = transfer_auroc - raw_auroc
    retention = transfer_gain / in_gain if in_gain > 0 else float("nan")
    return {
        "seed": seed,
        "source_probe_C": probe["C"],
        "raw_auroc": raw_auroc,
        "in_distribution_probe_auroc": target_probe_auroc,
        "transfer_probe_auroc": transfer_auroc,
        "in_distribution_gain": in_gain,
        "transfer_gain": transfer_gain,
        "transfer_gain_ci": boot["differences"]["transfer_probe-raw"]["auroc"]["ci"],
        "retention": retention,
        "passes": bool(in_gain > 0 and retention >= H2_RETENTION),
        "transfer_metrics": boot["methods"]["transfer_probe"],
    }


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=RUNS_DIR / "phase1_local")
    args = parser.parse_args()
    out = args.out_dir
    (out / "test_predictions").mkdir(parents=True, exist_ok=True)

    data = {key: load_run(folder) for key, folder in RUNS.items()}

    h1: dict = {}
    h1_rows = []
    for (model, dataset), d in data.items():
        run = f"{model}-{dataset}"
        h1[run] = {}
        for seed in SEEDS:
            res = in_distribution(d, seed)
            signals = res.pop("_signals")
            res.pop("_probe")
            h1[run][seed] = res
            diff = res["bootstrap"]["differences"]["probe-calibrated_raw"]
            for metric in METRICS:
                row = {"run": run, "seed": seed, "metric": metric}
                for m, v in res["bootstrap"]["methods"].items():
                    row[m] = v[metric]["value"]
                row["probe_minus_calibrated_raw"] = diff[metric]["value"]
                row["diff_ci_low"], row["diff_ci_high"] = diff[metric]["ci"]
                h1_rows.append(row)
            if seed == PRIMARY_SEED:
                preds = pd.DataFrame({"id": d["test"]["ids"], "reviewed_label": d["test"]["y"]})
                for m, v in signals.items():
                    preds[m] = v
                preds.to_csv(out / "test_predictions" / f"{run}.csv", index=False)
        print(f"H1 done: {run}")
    pd.DataFrame(h1_rows).to_csv(out / "h1_metrics.csv", index=False)

    h1_pass_by_seed = {}
    for seed in SEEDS:
        wins = [
            run
            for run in h1
            if h1[run][seed]["bootstrap"]["differences"]["probe-calibrated_raw"]["auroc"]["ci"][0] > 0
        ]
        h1_pass_by_seed[seed] = {"wins": wins, "n_wins": len(wins), "passes": len(wins) >= H1_MIN_RUNS}

    h2_rows = []
    h2: dict = {}
    for model in MODELS:
        for src, tgt in (("squad", "triviaqa"), ("triviaqa", "squad")):
            direction = f"{model}:{src}->{tgt}"
            h2[direction] = {}
            for seed in SEEDS:
                target_auroc = h1[f"{model}-{tgt}"][seed]["bootstrap"]["methods"]["probe"]["auroc"]["value"]
                res = transfer(data[(model, src)], data[(model, tgt)], target_auroc, seed)
                h2[direction][seed] = res
                h2_rows.append(
                    {
                        "direction": direction,
                        "seed": seed,
                        **{k: res[k] for k in ("raw_auroc", "in_distribution_probe_auroc", "transfer_probe_auroc", "in_distribution_gain", "transfer_gain", "retention", "passes")},
                        "transfer_gain_ci_low": res["transfer_gain_ci"][0],
                        "transfer_gain_ci_high": res["transfer_gain_ci"][1],
                        "transfer_ece": res["transfer_metrics"]["ece"]["value"],
                        "transfer_brier": res["transfer_metrics"]["brier"]["value"],
                    }
                )
            print(f"H2 done: {direction}")
    pd.DataFrame(h2_rows).to_csv(out / "h2_transfer.csv", index=False)

    h2_pass_by_seed = {}
    for seed in SEEDS:
        passing = [d for d in h2 if h2[d][seed]["passes"]]
        h2_pass_by_seed[seed] = {"passing": passing, "n_passing": len(passing), "passes": len(passing) >= H2_MIN_DIRECTIONS}

    decisions = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "TCL-Phase1-Protocol.md",
        "primary_seed": PRIMARY_SEED,
        "H1": {
            "rule": f"probe - calibrated_raw AUROC CI lower bound > 0 in >= {H1_MIN_RUNS} of 6 runs",
            "primary": h1_pass_by_seed[PRIMARY_SEED],
            "by_seed": h1_pass_by_seed,
        },
        "H2": {
            "rule": f"retention >= {H2_RETENTION} in >= {H2_MIN_DIRECTIONS} of 6 directions",
            "primary": h2_pass_by_seed[PRIMARY_SEED],
            "by_seed": h2_pass_by_seed,
        },
        "test_label_changes_from_review": {f"{m}-{ds}": data[(m, ds)]["n_test_label_changes"] for (m, ds) in data},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    (out / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (out / "h1_details.json").write_text(json.dumps(h1, indent=2), encoding="utf-8")
    (out / "h2_details.json").write_text(json.dumps(h2, indent=2), encoding="utf-8")

    print(json.dumps({"H1": decisions["H1"]["primary"], "H2": decisions["H2"]["primary"]}, indent=2))


if __name__ == "__main__":
    main()
