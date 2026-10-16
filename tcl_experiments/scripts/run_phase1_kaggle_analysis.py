"""Analysis of the Phase 1 Kaggle runs (TCL-Phase1-Kaggle-Protocol.md, Section 5).

Expects the unzipped Kaggle outputs under runs/phase1_kaggle/<model>/<dataset>/,
each with records.jsonl and hidden_chunk_*.npz from run_inference_v2.py.

Reuses the fitting, metric, and bootstrap code of run_phase1_analysis.py so the
replication follows TCL-Phase1-Protocol.md exactly.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import run_phase1_analysis as p1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "runs" / "phase1_kaggle"
MODELS = ["qwen", "phi", "gemma"]
DATASETS = ["triviaqa", "squad", "nq_open"]
PRIMARY_FEATURE = "final_answer_mean"
SWEEP_MODEL = "qwen"
MIN_PER_CLASS = 150
AUDIT_ROWS = 100
AUDIT_SEED = 2026


def load_pairing(run_dir: Path) -> dict:
    records = [json.loads(l) for l in (run_dir / "records.jsonl").open(encoding="utf-8") if l.strip()]
    by_id = {r["id"]: i for i, r in enumerate(records)}
    features: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(str(run_dir / "hidden_chunk_*.npz"))):
        z = np.load(path)
        idx = [by_id[i] for i in z["ids"].tolist()]
        for key in z.files:
            if key == "ids":
                continue
            if key not in features:
                features[key] = np.zeros((len(records), z[key].shape[1]), dtype=np.float32)
            features[key][idx] = z[key].astype(np.float32)
    df = pd.DataFrame(records)
    return {"df": df, "features": features}


def split_view(pairing: dict, feature: str, refusals: bool = True) -> dict:
    df, X = pairing["df"], pairing["features"][feature]
    out = {}
    for split in ("train", "validation", "test"):
        mask = (df["split"] == split).to_numpy().copy()
        if not refusals:
            mask &= ~df["refusal"].to_numpy(bool)
        out[split] = {
            "ids": df.loc[mask, "id"].tolist(),
            "X": X[mask].astype(np.float64),
            "raw": np.clip(df.loc[mask, "raw_conf_span"].to_numpy(float), p1.EPS, 1 - p1.EPS),
            "raw_full": np.clip(df.loc[mask, "raw_conf_full"].to_numpy(float), p1.EPS, 1 - p1.EPS),
            "p_true": np.clip(df.loc[mask, "p_true"].to_numpy(float), p1.EPS, 1 - p1.EPS),
            "se_conf": -df.loc[mask, "semantic_entropy"].to_numpy(float) if split == "test" else None,
            "y": df.loc[mask, "correctness_label"].to_numpy(int),
        }
    return out


def class_counts(pairing: dict) -> dict:
    y = pairing["df"].loc[pairing["df"]["split"] == "test", "correctness_label"]
    return {"correct": int((y == 1).sum()), "incorrect": int((y == 0).sum())}


def rank_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Paired bootstrap of AUROC and AURC differences a - b (ranking metrics only)."""
    rng = np.random.default_rng(p1.BOOT_SEED)
    d_auc, d_aurc = [], []
    for _ in range(p1.N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        d_aurc.append(p1.aurc(y[idx], a[idx]) - p1.aurc(y[idx], b[idx]))
    return {
        "auroc": {"value": float(roc_auc_score(y, a) - roc_auc_score(y, b)), "ci": [float(np.percentile(d_auc, 2.5)), float(np.percentile(d_auc, 97.5))]},
        "aurc": {"value": float(p1.aurc(y, a) - p1.aurc(y, b)), "ci": [float(np.percentile(d_aurc, 2.5)), float(np.percentile(d_aurc, 97.5))]},
    }


def in_distribution(view: dict, seed: int) -> dict:
    train, val, test = view["train"], view["validation"], view["test"]
    cal_name, cal_fn, _ = p1.select_raw_calibrator(val["raw"], val["y"], seed)
    probe = p1.fit_probe(train, val, seed)
    pt_platt = p1.fit_platt(p1.logit(val["p_true"]), val["y"])
    signals = {
        "raw": test["raw"],
        "calibrated_raw": np.clip(cal_fn(test["raw"]), 0, 1),
        "probe": probe["calibrated"](test["X"]),
        "p_true": pt_platt(p1.logit(test["p_true"])),
        "raw_full": test["raw_full"],
    }
    boot = p1.paired_bootstrap(
        test["y"], signals, pairs=[("probe", "calibrated_raw"), ("probe", "p_true"), ("raw", "raw_full")]
    )
    out = {"seed": seed, "raw_calibrator": cal_name, "probe_C": probe["C"], "positive_rate_test": float(test["y"].mean()), "bootstrap": boot}
    se = test["se_conf"]
    if se is not None and np.isfinite(se).all():
        out["probe_vs_semantic_entropy"] = rank_bootstrap(test["y"], signals["probe"], se)
        out["semantic_entropy_auroc"] = float(roc_auc_score(test["y"], se))
    return out, signals, probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase1_kaggle_analysis")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    pairings, counts, dropped = {}, {}, []
    for m in MODELS:
        for ds in DATASETS:
            run_dir = args.in_dir / m / ds
            if not (run_dir / "records.jsonl").exists():
                dropped.append({"pairing": f"{m}-{ds}", "reason": "missing"})
                continue
            pr = load_pairing(run_dir)
            c = class_counts(pr)
            counts[f"{m}-{ds}"] = c
            if min(c.values()) < MIN_PER_CLASS:
                dropped.append({"pairing": f"{m}-{ds}", "reason": f"size rule: {c}"})
                continue
            pairings[(m, ds)] = pr
    print("retained:", [f"{m}-{d}" for m, d in pairings], "dropped:", dropped)

    # Label audit sample: drawn from labels only, before any metric.
    audit = []
    for (m, ds), pr in pairings.items():
        test = pr["df"][pr["df"]["split"] == "test"]
        sample = test.sample(n=min(AUDIT_ROWS, len(test)), random_state=AUDIT_SEED)
        for _, r in sample.iterrows():
            audit.append({"pairing": f"{m}-{ds}", "id": r["id"], "question": r["question"], "accepted_answers": " | ".join(r["accepted_answers"]),
                          "raw_output": r["raw_output"], "answer": r["answer"], "auto_label": r["correctness_label"], "refusal": r["refusal"], "manual_label": "", "notes": ""})
    pd.DataFrame(audit).to_csv(out / "label_audit_sample.csv", index=False)

    # H1-R and baseline comparisons.
    h1, rows, primary_probes = {}, [], {}
    for (m, ds), pr in pairings.items():
        key = f"{m}-{ds}"
        view = split_view(pr, PRIMARY_FEATURE)
        h1[key] = {}
        for seed in p1.SEEDS:
            res, signals, probe = in_distribution(view, seed)
            h1[key][seed] = res
            if seed == p1.PRIMARY_SEED:
                primary_probes[key] = probe
                preds = pd.DataFrame({"id": view["test"]["ids"], "label": view["test"]["y"], **signals})
                if view["test"]["se_conf"] is not None:
                    preds["neg_semantic_entropy"] = view["test"]["se_conf"]
                preds.to_csv(out / f"test_predictions_{key}.csv", index=False)
            d = res["bootstrap"]["differences"]["probe-calibrated_raw"]
            for metric in p1.METRICS:
                rows.append({"pairing": key, "seed": seed, "metric": metric,
                             **{s: v[metric]["value"] for s, v in res["bootstrap"]["methods"].items()},
                             "probe_minus_calibrated_raw": d[metric]["value"], "ci_low": d[metric]["ci"][0], "ci_high": d[metric]["ci"][1]})
        print("H1-R done:", key)
    pd.DataFrame(rows).to_csv(out / "h1r_metrics.csv", index=False)

    n_needed_h1 = math.ceil(len(pairings) * 2 / 3)
    h1_by_seed = {}
    for seed in p1.SEEDS:
        wins = [k for k in h1 if h1[k][seed]["bootstrap"]["differences"]["probe-calibrated_raw"]["auroc"]["ci"][0] > 0]
        h1_by_seed[seed] = {"wins": wins, "n_wins": len(wins), "n_needed": n_needed_h1, "passes": bool(pairings) and len(wins) >= n_needed_h1}

    # Secondary: refusals excluded.
    secondary = {}
    for (m, ds), pr in pairings.items():
        view = split_view(pr, PRIMARY_FEATURE, refusals=False)
        if min(np.bincount(view["test"]["y"], minlength=2)) < 2:
            continue
        res, _, _ = in_distribution(view, p1.PRIMARY_SEED)
        secondary[f"{m}-{ds}"] = res["bootstrap"]["differences"]["probe-calibrated_raw"]["auroc"]

    # H2-R: transfer between retained datasets of the same model.
    h2, h2_rows = {}, []
    for m in MODELS:
        dss = [ds for ds in DATASETS if (m, ds) in pairings]
        for src in dss:
            for tgt in dss:
                if src == tgt:
                    continue
                direction = f"{m}:{src}->{tgt}"
                h2[direction] = {}
                for seed in p1.SEEDS:
                    s_view = split_view(pairings[(m, src)], PRIMARY_FEATURE)
                    t_view = split_view(pairings[(m, tgt)], PRIMARY_FEATURE)
                    target_auroc = h1[f"{m}-{tgt}"][seed]["bootstrap"]["methods"]["probe"]["auroc"]["value"]
                    res = p1.transfer(s_view, t_view, target_auroc, seed)
                    h2[direction][seed] = res
                    h2_rows.append({"direction": direction, "seed": seed, **{k: res[k] for k in ("raw_auroc", "in_distribution_probe_auroc", "transfer_probe_auroc", "retention", "passes")},
                                    "transfer_gain_ci_low": res["transfer_gain_ci"][0], "transfer_gain_ci_high": res["transfer_gain_ci"][1]})
                print("H2-R done:", direction)
    pd.DataFrame(h2_rows).to_csv(out / "h2r_transfer.csv", index=False)
    n_needed_h2 = math.ceil(len(h2) * 2 / 3)
    h2_by_seed = {}
    for seed in p1.SEEDS:
        passing = [d for d in h2 if h2[d][seed]["passes"]]
        h2_by_seed[seed] = {"passing": passing, "n_passing": len(passing), "n_needed": n_needed_h2, "passes": bool(h2) and len(passing) >= n_needed_h2}

    # Layer sweep (descriptive): Qwen only; selection on validation AUROC.
    sweep_rows = []
    for ds in DATASETS:
        if (SWEEP_MODEL, ds) not in pairings:
            continue
        pr = pairings[(SWEEP_MODEL, ds)]
        for feature in sorted(pr["features"]):
            view = split_view(pr, feature)
            probe = p1.fit_probe(view["train"], view["validation"], p1.PRIMARY_SEED)
            sweep_rows.append({"dataset": ds, "feature": feature, "probe_C": probe["C"],
                               "validation_auroc": float(roc_auc_score(view["validation"]["y"], probe["score"](view["validation"]["X"]))),
                               "test_auroc": float(roc_auc_score(view["test"]["y"], probe["score"](view["test"]["X"])))})
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out / "layer_sweep_qwen.csv", index=False)
    best_feature = None
    if len(sweep):
        best_feature = sweep.groupby("feature")["validation_auroc"].mean().idxmax()

    decisions = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "TCL-Phase1-Kaggle-Protocol.md",
        "class_counts_test": counts,
        "dropped": dropped,
        "H1-R": {"rule": "probe - calibrated_raw AUROC CI lower bound > 0 in >= 2/3 of retained pairings", "primary": h1_by_seed[p1.PRIMARY_SEED], "by_seed": h1_by_seed},
        "H2-R": {"rule": "retention >= 0.70 in >= 2/3 of transfer directions", "primary": h2_by_seed[p1.PRIMARY_SEED], "by_seed": h2_by_seed},
        "secondary_refusals_excluded_auroc_diff": secondary,
        "layer_sweep_best_feature_by_mean_validation_auroc": best_feature,
    }
    (out / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (out / "h1r_details.json").write_text(json.dumps(h1, indent=2), encoding="utf-8")
    (out / "h2r_details.json").write_text(json.dumps(h2, indent=2), encoding="utf-8")
    print(json.dumps({"H1-R": decisions["H1-R"]["primary"], "H2-R": decisions["H2-R"]["primary"], "best_layer": best_feature}, indent=2))


if __name__ == "__main__":
    main()
