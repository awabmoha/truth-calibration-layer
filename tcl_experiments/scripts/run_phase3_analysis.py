"""Phase 3 (H4-T) analysis: apply the locked decision rule once.

Protocol: docs/phase3/TCL-Phase3-Protocol.md, Sections 6-7.

Reads the stage 3b scores written by run_phase3_heads.py score and computes, per
eligible (model, held-out dataset) pair:
  * d(MR, R)          the gated difference
  * d(M, S_d)         per single training dataset, and their mean (reported)
  * d(MR, M)          (reported)
  * d(MR, SE)         against semantic entropy (reported)
each as a seed-averaged AUROC difference with a 95% paired bootstrap CI (1,000
resamples of test items, seed 2026) and per-seed values; secondary metrics after a
Platt map fitted on the training mixture's validation splits; and every result again
with refusals excluded.

H4-T passes if d(MR, R) has a CI lower bound above zero on a majority of eligible
pairs (more than half), including at least one PopQA pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1_analysis as p1  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 1000
BOOT_SEED = 2026
NEW_DATASET = "popqa"


def seed_matrix(df: pd.DataFrame, arm: str, ids: np.ndarray) -> np.ndarray:
    """Scores of one arm on the test ids, as (n_seeds, n_items), aligned to ids."""
    a = df[(df["arm"] == arm) & (df["split"] == "test")]
    return np.vstack([a[a["seed"] == s].set_index("id").loc[ids, "score"].to_numpy(float)
                      for s in sorted(a["seed"].unique())])


def auc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """AUROC of each row of S (Mann-Whitney with average ranks for ties; equals roc_auc_score)."""
    ranks = rankdata(S, axis=1)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return (ranks[:, pos].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def mean_auc(y, S, idx=None) -> float:
    if idx is not None:
        y, S = y[idx], S[:, idx]
    return float(auc_rows(y, S).mean())


def paired_diff(y, A, B) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(mean_auc(y, A, idx) - mean_auc(y, B, idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    # Unseeded arms (R, SE) have one row; pair it with every seed of the other arm.
    n = max(len(A), len(B))
    A_, B_ = (np.repeat(X, n, 0) if len(X) == 1 else X for X in (A, B))
    per_seed = [float(roc_auc_score(y, a) - roc_auc_score(y, b)) for a, b in zip(A_, B_)]
    return {"value": mean_auc(y, A) - mean_auc(y, B), "ci": [float(lo), float(hi)], "per_seed": per_seed}


def secondary(df: pd.DataFrame, arm: str, test_ids: np.ndarray, y: np.ndarray) -> dict:
    """Seed-averaged AURC, coverage accuracy, ECE, Brier after a Platt map on mixture validation."""
    out = {k: [] for k in p1.METRICS}
    a = df[df["arm"] == arm]
    for s in sorted(a["seed"].unique()):
        va = a[(a["seed"] == s) & (a["split"] == "validation")]
        te = a[(a["seed"] == s) & (a["split"] == "test")].set_index("id").loc[test_ids]
        platt = p1.fit_platt(va["score"].to_numpy(float), va["label"].to_numpy(int))
        m = p1.score_all(y, platt(te["score"].to_numpy(float)))
        for k in p1.METRICS:
            out[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def analyze_pair(df: pd.DataFrame, refusals: bool) -> dict:
    test = df[(df["split"] == "test") & (df["arm"] == "R")].drop_duplicates("id")
    if not refusals:
        test = test[~test["refusal"]]
    ids, y = test["id"].to_numpy(), test["label"].to_numpy(int)
    arms = sorted(df["arm"].unique())
    singles = [a for a in arms if a.startswith("S_")]
    M = {a: seed_matrix(df, a, ids) for a in arms if a != "SE"}
    out = {"n": int(len(y)), "correct": int(y.sum()),
           "auroc": {a: mean_auc(y, M[a]) for a in M},
           "d(MR,R)": paired_diff(y, M["MR"], M["R"]),
           "d(MR,M)": paired_diff(y, M["MR"], M["M"])}
    out["d(M,S)"] = {s: paired_diff(y, M["M"], M[s]) for s in singles}
    out["d(M,S)_mean"] = float(np.mean([v["value"] for v in out["d(M,S)"].values()]))
    se = seed_matrix(df, "SE", ids)
    keep = ~np.isnan(se[0])
    if keep.sum() > 0 and len(np.unique(y[keep])) == 2:
        out["auroc"]["SE"] = mean_auc(y[keep], se[:, keep])
        out["d(MR,SE)"] = paired_diff(y[keep], M["MR"][:, keep], se[:, keep])
        out["SE_items"] = int(keep.sum())
    if refusals:
        out["secondary"] = {a: secondary(df, a, ids, y) for a in M}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "runs" / "phase3b")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase3b_analysis")
    args = ap.parse_args()
    manifest = json.loads((args.in_dir / "manifest.json").read_text(encoding="utf-8"))
    eligible = [(p["model"], p["held_out"]) for p in manifest["pairs"] if p["eligible"]]
    details, passed = {}, []
    for m, held in eligible:
        df = pd.read_csv(args.in_dir / f"scores_{m}_{held}.csv", keep_default_na=True)
        key = f"{m}/{held}"
        details[key] = {"all": analyze_pair(df, True), "refusals_excluded": analyze_pair(df, False)}
        lo = details[key]["all"]["d(MR,R)"]["ci"][0]
        if lo > 0:
            passed.append(key)
        d = details[key]["all"]["d(MR,R)"]
        print(f"{key:18s} d(MR,R) {d['value']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", flush=True)
    need = len(eligible) // 2 + 1
    popqa_ok = any(k.endswith("/" + NEW_DATASET) for k in passed)
    decision = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase3/TCL-Phase3-Protocol.md",
        "chosen_setting": manifest["chosen"],
        "eligible_pairs": [f"{m}/{h}" for m, h in eligible],
        "ineligible_pairs": [f"{p['model']}/{p['held_out']}" for p in manifest["pairs"] if not p["eligible"]],
        "pairs_with_d(MR,R)_ci_above_zero": passed,
        "needed": need,
        "includes_popqa_pair": popqa_ok,
        "H4-T": {"passes": len(passed) >= need and popqa_ok,
                 "rule": "d(MR,R) CI lower bound > 0 on a majority of eligible pairs, including at least one PopQA pair"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decisions.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.out_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
