"""Phase 3 H4 (grounding) analysis: apply the locked decision rule once.

Protocol: docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md, Section 5.

Per eligible (model, held-out dataset) pair, on the held-out test answers that are not
abstentions:
  1. learnability     AUROC of G-hat against the NLI grounding label, >= 0.70
  2. non-redundancy   Spearman correlation of G-hat with calibrated raw confidence, < 0.7
  3. added value      d_AURC = AURC(R) - AURC(RG), 95% paired bootstrap CI above zero
                      (1,000 resamples of test items, seed 2026)
Metrics are computed per seed and averaged. H4 (grounding) passes if a majority of
eligible pairs pass, including the new-dataset pair of at least one model.

Also reported: correctness AUROC of every arm (R, RG, RG_oracle, PTRUE); a "grounded and
correct" target; results on gold-passage and distractor rows separately; secondary
metrics; refusal-excluded results; per-seed values.
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
from run_phase3_h4_analysis import d_aurc, d_auroc, matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NEW = "relation_extraction"
MIN_AUROC = 0.70
MAX_SPEARMAN = 0.7


def safe_auc(y: np.ndarray, S: np.ndarray) -> float:
    return float(auc_rows(y, S).mean()) if len(np.unique(y)) == 2 else float("nan")


def analyze_pair(df: pd.DataFrame, refusals: bool) -> dict:
    test = df[df["arm"] == "R"].drop_duplicates("id")
    if not refusals:
        test = test[~test["refusal"]]
    ids, y, g = test["id"].to_numpy(), test["label"].to_numpy(int), test["grounded"].to_numpy(int)
    dis = test["distractor"].to_numpy(int) == 1
    R, G, RG, RGo, PT = (matrix(df, a, ids) for a in ("R", "GHAT", "RG", "RG_oracle", "PTRUE"))
    learn = auc_rows(g, G)
    rho = [float(spearmanr(s, R[0]).statistic) for s in G]
    out = {
        "n": int(len(y)), "correct": int(y.sum()), "grounded": int(g.sum()), "distractor_rows": int(dis.sum()),
        "learnability_auroc": {"value": float(learn.mean()), "per_seed": learn.tolist()},
        "spearman_with_raw": {"value": float(np.mean(rho)), "per_seed": rho},
        "d_aurc(R,RG)": d_aurc(y, R, RG),
        "correctness_auroc": {"R": safe_auc(y, R), "RG": safe_auc(y, RG), "RG_oracle": safe_auc(y, RGo)},
        "d_auroc(RG,R)": d_auroc(y, RG, R),
        "grounded_and_correct_auroc": {a: safe_auc(y * g, M) for a, M in (("R", R), ("RG", RG), ("GHAT", G))},
        "by_passage": {name: {"n": int(mask.sum()), "correct": int(y[mask].sum()), "grounded": int(g[mask].sum()),
                              "correctness_auroc": {a: safe_auc(y[mask], M[:, mask]) for a, M in (("R", R), ("RG", RG))},
                              "learnability_auroc": safe_auc(g[mask], G[:, mask])}
                       for name, mask in (("gold", ~dis), ("distractor", dis))},
    }
    pt_ok = ~np.isnan(PT[0])
    if pt_ok.sum() and len(np.unique(y[pt_ok])) == 2:
        out["correctness_auroc"]["PTRUE"] = safe_auc(y[pt_ok], PT[:, pt_ok])
    if refusals:
        out["secondary"] = {a: {k: float(np.mean([p1.score_all(y, s)[k] for s in M])) for k in p1.METRICS}
                            for a, M in (("R", R), ("RG", RG), ("RG_oracle", RGo))}
    out["passes"] = bool(out["learnability_auroc"]["value"] >= MIN_AUROC
                         and out["spearman_with_raw"]["value"] < MAX_SPEARMAN
                         and out["d_aurc(R,RG)"]["ci"][0] > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "runs" / "phase3_h4g")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase3_h4g_analysis")
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
        d = a["d_aurc(R,RG)"]
        print(f"{key:28s} learn AUROC {a['learnability_auroc']['value']:.3f}  rho {a['spearman_with_raw']['value']:+.3f}  "
              f"d_AURC {d['value']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  {'PASS' if a['passes'] else 'fail'}",
              flush=True)
    need = len(eligible) // 2 + 1
    new_ok = any(k.endswith("/" + NEW) for k in passed)
    decision = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md",
        "eligible_pairs": [f"{m}/{h}" for m, h in eligible],
        "ineligible_pairs": [f"{p['model']}/{p['held_out']}" for p in manifest["pairs"] if not p["eligible"]],
        "pairs_passing": passed, "needed": need, "includes_new_dataset_pair": new_ok,
        "H4_grounding": {
            "passes": len(passed) >= need and new_ok,
            "rule": "per pair: G-hat AUROC >= 0.70, Spearman with raw < 0.7, d_AURC CI lower bound > 0; "
                    "majority of eligible pairs, including the new-dataset pair of at least one model"},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "decisions.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (args.out_dir / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
