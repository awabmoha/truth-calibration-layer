"""Phase 3 H4 (self-consistency) analysis: apply the locked decision rule once.

Protocol: docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md, Sections 4-5.

Per eligible (model, held-out dataset) pair, on the held-out test split:
  1. learnability     AUROC of C-hat against the binary consistency label (y_c >= 0.5)
  2. non-redundancy   Spearman correlation of C-hat with calibrated raw confidence
  3. added value      d_AURC = AURC(R) - AURC(RC), 95% paired bootstrap CI
                      (1,000 resamples of test items, seed 2026)
Metrics are computed per seed and averaged over seeds (5 for C-hat and RC); each
bootstrap resample averages over seeds too. A pair passes if mean AUROC >= 0.70,
mean Spearman < 0.7, and the d_AURC CI lower bound > 0. H4 passes if a majority of
eligible pairs pass, including at least one PopQA pair.

Also reported: correctness AUROC of each arm and d(RC, R); the oracle RC_oracle and SE;
secondary metrics; per-seed values; everything again with refusals excluded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phase1_analysis as p1  # noqa: E402
from run_phase3_analysis import auc_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 1000
BOOT_SEED = 2026
NEW_DATASET = "popqa"
CONS_THRESHOLD = 0.5
MIN_AUROC = 0.70
MAX_SPEARMAN = 0.7


def matrix(df: pd.DataFrame, arm: str, ids: np.ndarray) -> np.ndarray:
    a = df[df["arm"] == arm]
    return np.vstack([a[a["seed"] == s].set_index("id").loc[ids, "score"].to_numpy(float)
                      for s in sorted(a["seed"].unique())])


def aurc_rows(y: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.array([p1.aurc(y, s) for s in S])


def d_aurc(y: np.ndarray, R: np.ndarray, RC: np.ndarray) -> dict:
    """AURC(R) - AURC(RC), seed-averaged; positive means the combination is better."""
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        draws.append(aurc_rows(y[idx], R[:, idx]).mean() - aurc_rows(y[idx], RC[:, idx]).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    per_seed = [float(p1.aurc(y, R[0]) - p1.aurc(y, rc)) for rc in RC]
    return {"value": float(aurc_rows(y, R).mean() - aurc_rows(y, RC).mean()),
            "ci": [float(lo), float(hi)], "per_seed": per_seed}


def d_auroc(y: np.ndarray, A: np.ndarray, B: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            draws.append(auc_rows(y[idx], A[:, idx]).mean() - auc_rows(y[idx], B[:, idx]).mean())
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(auc_rows(y, A).mean() - auc_rows(y, B).mean()), "ci": [float(lo), float(hi)]}


def analyze_pair(df: pd.DataFrame, refusals: bool) -> dict:
    test = df[df["arm"] == "R"].drop_duplicates("id")
    if not refusals:
        test = test[~test["refusal"]]
    ids, y = test["id"].to_numpy(), test["label"].to_numpy(int)
    cons_bin = (test["cons"].to_numpy(float) >= CONS_THRESHOLD).astype(int)
    R, C, RC, RCo = (matrix(df, a, ids) for a in ("R", "CHAT", "RC", "RC_oracle"))
    learn = auc_rows(cons_bin, C)
    rho = [float(spearmanr(c, R[0]).statistic) for c in C]
    out = {
        "n": int(len(y)), "correct": int(y.sum()), "consistent": int(cons_bin.sum()),
        "learnability_auroc": {"value": float(learn.mean()), "per_seed": learn.tolist()},
        "spearman_with_raw": {"value": float(np.mean(rho)), "per_seed": rho},
        "d_aurc(R,RC)": d_aurc(y, R, RC),
        "correctness_auroc": {"R": float(auc_rows(y, R).mean()), "RC": float(auc_rows(y, RC).mean()),
                              "RC_oracle": float(auc_rows(y, RCo).mean())},
        "d_auroc(RC,R)": d_auroc(y, RC, R),
        "d_auroc(RC_oracle,RC)": d_auroc(y, RCo, RC),
    }
    se = matrix(df, "SE", ids)
    keep = ~np.isnan(se[0])
    if keep.sum() and len(np.unique(y[keep])) == 2:
        out["correctness_auroc"]["SE"] = float(auc_rows(y[keep], se[:, keep]).mean())
        out["SE_items"] = int(keep.sum())
    if refusals:
        out["secondary"] = {a: {k: float(np.mean([p1.score_all(y, s)[k] for s in M]))
                                for k in p1.METRICS}
                            for a, M in (("R", R), ("RC", RC), ("RC_oracle", RCo))}
    out["passes"] = bool(out["learnability_auroc"]["value"] >= MIN_AUROC
                         and out["spearman_with_raw"]["value"] < MAX_SPEARMAN
                         and out["d_aurc(R,RC)"]["ci"][0] > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "runs" / "phase3_h4")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase3_h4_analysis")
    args = ap.parse_args()
    manifest = json.loads((args.in_dir / "manifest.json").read_text(encoding="utf-8"))
    eligible = [(p["model"], p["held_out"]) for p in manifest["pairs"] if p["eligible"]]
    details, passed = {}, []
    for m, held in eligible:
        df = pd.read_csv(args.in_dir / f"scores_{m}_{held}.csv")
        key = f"{m}/{held}"
        details[key] = {"all": analyze_pair(df, True), "refusals_excluded": analyze_pair(df, False)}
        a = details[key]["all"]
        if a["passes"]:
            passed.append(key)
        d = a["d_aurc(R,RC)"]
        print(f"{key:16s} learn AUROC {a['learnability_auroc']['value']:.3f}  rho {a['spearman_with_raw']['value']:+.3f}  "
              f"d_AURC {d['value']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  {'PASS' if a['passes'] else 'fail'}",
              flush=True)
    need = len(eligible) // 2 + 1
    popqa_ok = any(k.endswith("/" + NEW_DATASET) for k in passed)
    decision = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md",
        "eligible_pairs": [f"{m}/{h}" for m, h in eligible],
        "ineligible_pairs": [f"{p['model']}/{p['held_out']}" for p in manifest["pairs"] if not p["eligible"]],
        "pairs_passing": passed, "needed": need, "includes_popqa_pair": popqa_ok,
        "H4_self_consistency": {
            "passes": len(passed) >= need and popqa_ok,
            "rule": "per pair: C-hat AUROC >= 0.70, Spearman with raw < 0.7, d_AURC CI lower bound > 0; "
                    "majority of eligible pairs, including at least one PopQA pair"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decisions.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.out_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
