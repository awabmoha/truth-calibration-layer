"""Final reporting analyses for the closed TCL program (docs/final/TCL-Paper-Outline.md).

Reads saved analysis outputs and score files only. Trains nothing, decides nothing; every
recorded decision stands. Three parts:

  1. Minimum detectable effect (MDE) for every preregistered gated difference: from the
     recorded 95% bootstrap CI, SE ~= (hi - lo) / 3.92, and the effect a test of this
     precision detects with 80% power at the CI-above-zero rule is MDE = 2.80 * SE.
  2. The weak-confidence pattern, corrected for its built-in artifacts: for each head
     (H4-T M+R, self-consistency R+C-hat, grounding R+G-hat) across the eligible Phase 3 and
     Phase 4a pairs, the Spearman correlation between raw-confidence AUROC and the head's
     AUROC gain, computed (a) on the full test set (as in runs/phase4a_ungated), (b) with the
     gain normalized by headroom, gain / (1 - raw AUROC), (c) split-sample: raw AUROC on one
     random half of the test items and the gain on the other half, averaged over 200 splits,
     which removes the shared-noise artifact, and (d) with a 95% CI from resampling pairs,
     plus leave-one-model-out values.
  3. Baseline choice: for the H4-T held-out pairs, AUROC of the answer-span probability
     (TCL's baseline), the first-token probability, and the minimum token probability, next
     to the learned head M+R, to show how much the baseline choice moves the probe-vs-logit gap.

Usage: python scripts/final_analyses.py
Output: runs/final_analyses/{mde.csv, weak_confidence_corrected.json, baseline_choice.csv}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "final_analyses"
Z = 1.96 + 0.84  # one-sided 2.5% test (CI lower bound > 0) at 80% power
RNG_SEED = 2026


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 1. MDE
def mde_rows() -> list[dict]:
    rows = []

    def add(phase, test, pair, metric, d):
        lo, hi = d["ci"]
        se = (hi - lo) / 3.92
        rows.append({"phase": phase, "test": test, "pair": pair, "metric": metric, "value": d["value"],
                     "ci_lo": lo, "ci_hi": hi, "se": se, "mde80": Z * se, "ci_above_zero": lo > 0})

    h1 = load(RUNS / "phase1_kaggle_analysis" / "h1r_details.json")
    for pair, v in h1.items():
        add("1", "H1-R (in-distribution)", pair, "AUROC probe - calibrated raw",
            v["0"]["bootstrap"]["differences"]["probe-calibrated_raw"]["auroc"])
    for phase, d in (("2", "phase2_analysis"), ("2b", "phase2b_analysis")):
        det = load(RUNS / d / "details.json")
        for src, other in (("triviaqa", "squad"), ("squad", "triviaqa")):
            for k in ("d(C,A)", "d(C,B)"):
                add(phase, "H3-T" if k == "d(C,A)" else "H3-J", f"{src}->{other}", f"AUROC {k}", det[src]["transfer"][k])
    for phase, d in (("3", "phase3b_analysis"), ("4a", "phase4a_h4t_analysis")):
        for pair, v in load(RUNS / d / "details.json").items():
            add(phase, "H4-T", pair, "AUROC d(M+R, R)", v["all"]["d(MR,R)"])
    for phase, d, key, test in (("3", "phase3_h4_analysis", "d_aurc(R,RC)", "H4 self-consistency"),
                                ("4a", "phase4a_h4c_analysis", "d_aurc(R,RC)", "H4 self-consistency"),
                                ("3", "phase3_h4g_analysis", "d_aurc(R,RG)", "H4 grounding"),
                                ("4a", "phase4a_h4g_analysis", "d_aurc(R,RG)", "H4 grounding")):
        for pair, v in load(RUNS / d / "details.json").items():
            add(phase, test, pair, f"AURC gain {key}", v["all"][key])
    return rows


# ------------------------------------------------- 2. corrected weak-confidence pattern
HEADS = {  # test -> (score dirs, head arm)
    "H4-T (M+R)": (["phase3b", "phase4a_h4t"], "MR"),
    "self-consistency (R+C-hat)": (["phase3_h4", "phase4a_h4c"], "RC"),
    "grounding (R+G-hat)": (["phase3_h4g", "phase4a_h4g"], "RG"),
}
ANALYSIS = {"phase3b": "phase3b_analysis", "phase4a_h4t": "phase4a_h4t_analysis", "phase3_h4": "phase3_h4_analysis",
            "phase4a_h4c": "phase4a_h4c_analysis", "phase3_h4g": "phase3_h4g_analysis", "phase4a_h4g": "phase4a_h4g_analysis"}


def pair_items(score_dir: str, pair: str, head: str) -> pd.DataFrame:
    model, held = pair.split("/")
    df = pd.read_csv(RUNS / score_dir / f"scores_{model}_{held}.csv")
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    raw = df[df["arm"] == "R"].drop_duplicates("id").set_index("id")
    h = df[df["arm"] == head].pivot_table(index="id", columns="seed", values="score")
    out = pd.DataFrame({"label": raw["label"], "raw": raw["score"]}).join(h, how="inner")
    return out


def aurocs(items: pd.DataFrame) -> tuple[float, float]:
    y = items["label"].to_numpy()
    raw = roc_auc_score(y, items["raw"])
    seeds = [c for c in items.columns if c not in ("label", "raw")]
    head = float(np.mean([roc_auc_score(y, items[s]) for s in seeds]))
    return raw, head


def weak_confidence() -> dict:
    rng = np.random.default_rng(RNG_SEED)
    out = {}
    pooled = {"raw": [], "gain": [], "norm": [], "split": []}
    for test, (dirs, head) in HEADS.items():
        pairs = []
        for d in dirs:
            for pair in load(RUNS / ANALYSIS[d] / "details.json"):
                pairs.append((d, pair))
        raw_auc, gain, split_rho_pts = [], [], []
        items_by_pair = {}
        for d, pair in pairs:
            it = pair_items(d, pair, head)
            items_by_pair[pair] = it
            r, h = aurocs(it)
            raw_auc.append(r)
            gain.append(h - r)
        raw_auc, gain = np.array(raw_auc), np.array(gain)
        norm = gain / (1 - raw_auc)
        # split-sample: raw AUROC on half A, gain on half B
        split_rhos = []
        for _ in range(200):
            xa, yb = [], []
            for pair, it in items_by_pair.items():
                mask = rng.random(len(it)) < 0.5
                a, b = it[mask], it[~mask]
                if a["label"].nunique() < 2 or b["label"].nunique() < 2:
                    continue
                xa.append(aurocs(a)[0])
                rb, hb = aurocs(b)
                yb.append(hb - rb)
            split_rhos.append(spearmanr(xa, yb).statistic)
        # pair bootstrap CI for the full-sample Spearman
        n = len(raw_auc)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            if len(set(raw_auc[idx])) > 2:
                boots.append(spearmanr(raw_auc[idx], gain[idx]).statistic)
        models = sorted({p.split("/")[0] for _, p in pairs})
        lomo = {}
        for m in models:
            keep = np.array([p.split("/")[0] != m for _, p in pairs])
            if keep.sum() >= 4:
                lomo[m] = spearmanr(raw_auc[keep], gain[keep]).statistic
        out[test] = {
            "pairs": [p for _, p in pairs], "n": n,
            "raw_auroc": raw_auc.round(4).tolist(), "gain_auroc": gain.round(4).tolist(),
            "spearman_full": spearmanr(raw_auc, gain).statistic,
            "spearman_full_ci95_pair_bootstrap": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            "spearman_headroom_normalized": spearmanr(raw_auc, norm).statistic,
            "spearman_split_sample_mean": float(np.nanmean(split_rhos)),
            "spearman_split_sample_range_5_95": [float(np.nanpercentile(split_rhos, 5)), float(np.nanpercentile(split_rhos, 95))],
            "spearman_leave_one_model_out": lomo,
        }
        pooled["raw"] += raw_auc.tolist()
        pooled["gain"] += gain.tolist()
        pooled["norm"] += norm.tolist()
    out["pooled"] = {"n": len(pooled["raw"]), "spearman_full": spearmanr(pooled["raw"], pooled["gain"]).statistic,
                     "spearman_headroom_normalized": spearmanr(pooled["raw"], pooled["norm"]).statistic}
    return out


# ------------------------------------------------------------- 3. baseline choice
def records(model: str, dataset: str) -> pd.DataFrame:
    if model == "qwen7b":
        sub = RUNS / "phase4a" / ("phase3_popqa" if dataset == "popqa" else "phase1_kaggle") / model / dataset
    else:
        sub = RUNS / ("phase3_popqa" if dataset == "popqa" else "phase1_kaggle") / model / dataset
    rows = [json.loads(x) for x in (sub / "records.jsonl").open(encoding="utf-8") if x.strip()]
    return pd.DataFrame([{"id": r["id"], "first_token_prob": r.get("first_token_prob"), "min_token_prob": r.get("min_token_prob"),
                          "raw_conf_span": r.get("raw_conf_span")} for r in rows]).set_index("id")


def baseline_choice() -> list[dict]:
    rows = []
    for d in ("phase3b", "phase4a_h4t"):
        for pair in load(RUNS / ANALYSIS[d] / "details.json"):
            model, held = pair.split("/")
            it = pair_items(d, pair, "MR").join(records(model, held), how="inner")
            it = it.dropna(subset=["first_token_prob", "min_token_prob", "raw_conf_span"])
            y = it["label"]
            span = roc_auc_score(y, it["raw_conf_span"])
            first = roc_auc_score(y, it["first_token_prob"])
            mn = roc_auc_score(y, it["min_token_prob"])
            _, head = aurocs(it[["label", "raw"] + [c for c in it.columns if isinstance(c, (int, np.integer))]])
            rows.append({"phase": "3" if d == "phase3b" else "4a", "pair": pair, "n": len(it),
                         "auroc_span": span, "auroc_first_token": first, "auroc_min_token": mn, "auroc_head_MR": head,
                         "head_minus_span": head - span, "head_minus_first_token": head - first,
                         "head_minus_min_token": head - mn})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = mde_rows()
    write_csv(OUT / "mde.csv", m)
    print("== MDE (80% power) per gated difference")
    for r in m:
        print(f"{r['phase']:3s} {r['test']:24s} {r['pair']:28s} value {r['value']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  MDE {r['mde80']:.4f}")
    w = weak_confidence()
    (OUT / "weak_confidence_corrected.json").write_text(json.dumps(w, indent=2), encoding="utf-8")
    print("== weak-confidence pattern (Spearman of raw AUROC vs gain)")
    for t, v in w.items():
        if t == "pooled":
            print(f"pooled n={v['n']} full {v['spearman_full']:+.2f} headroom {v['spearman_headroom_normalized']:+.2f}")
            continue
        ci = v["spearman_full_ci95_pair_bootstrap"]
        print(f"{t:28s} n={v['n']} full {v['spearman_full']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] "
              f"headroom {v['spearman_headroom_normalized']:+.2f} split {v['spearman_split_sample_mean']:+.2f} "
              f"LOMO {json.dumps({k: round(x, 2) for k, x in v['spearman_leave_one_model_out'].items()})}")
    b = baseline_choice()
    write_csv(OUT / "baseline_choice.csv", b)
    print("== baseline choice (AUROC)")
    for r in b:
        print(f"{r['phase']:3s} {r['pair']:18s} span {r['auroc_span']:.3f} first {r['auroc_first_token']:.3f} "
              f"min {r['auroc_min_token']:.3f} head {r['auroc_head_MR']:.3f} | head-span {r['head_minus_span']:+.3f} "
              f"head-first {r['head_minus_first_token']:+.3f} head-min {r['head_minus_min_token']:+.3f}")


if __name__ == "__main__":
    main()
