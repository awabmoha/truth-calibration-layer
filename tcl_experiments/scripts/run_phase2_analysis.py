"""Phase 2 analysis (docs/phase2/TCL-Phase2-Protocol.md, Sections 6-9).

Expects the Kaggle outputs unpacked as
  runs/phase2/<train_dataset>/<arm>/seed<k>/{scores_fixed.csv, metrics.json, regenerated.csv}
where <arm> is A, B, D, or C-lam<lambda> (e.g. C-lam0.5).

Lambda for arm C is chosen per training dataset by validation AUROC with seed 0,
exactly as the protocol fixes it. Run once, after all runs are in.
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
from labeling import is_correct_v3

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("triviaqa", "squad")
SEEDS = (0, 1, 2)
LAMBDAS = (0.1, 0.5, 1.0)
N_BOOT = 1000
BOOT_SEED = 2026
AUDIT_SEED = 2028
ACC_DROP_MAX = 0.01   # 1 percentage point
PPL_RISE_MAX = 0.03   # 3%


def other(ds: str) -> str:
    return "squad" if ds == "triviaqa" else "triviaqa"


def load_run(path: Path) -> dict:
    return {
        "scores": pd.read_csv(path / "scores_fixed.csv"),
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "regen": pd.read_csv(path / "regenerated.csv", keep_default_na=False) if (path / "regenerated.csv").exists() else None,
    }


def choose_lambda(root: Path, ds: str) -> tuple[float, dict]:
    aucs = {}
    for lam in LAMBDAS:
        m = root / ds / f"C-lam{lam}" / "seed0" / "metrics.json"
        if m.exists():
            aucs[lam] = json.loads(m.read_text(encoding="utf-8"))["validation_auroc"]
    best = max(aucs, key=lambda k: (aucs[k] if aucs[k] is not None else -1))
    return best, aucs


def test_scores(run: dict, ds: str) -> pd.DataFrame:
    s = run["scores"]
    return s[(s["dataset"] == ds) & (s["split"] == "test")].sort_values("id").reset_index(drop=True)


def seed_matrix(runs: dict[int, dict], ds: str, refusals: bool = True) -> tuple[np.ndarray, np.ndarray]:
    frames = [test_scores(runs[s], ds) for s in SEEDS]
    base = frames[0]
    for f in frames[1:]:
        assert (f["id"].values == base["id"].values).all() and (f["label"].values == base["label"].values).all()
    keep = np.ones(len(base), dtype=bool) if refusals else ~base["refusal"].astype(bool).values
    return base["label"].values[keep].astype(int), np.stack([f["trust_logit"].values[keep] for f in frames])


def seed_mean_auc(y, scores, idx=None) -> float:
    if idx is not None:
        y, scores = y[idx], scores[:, idx]
    return float(np.mean([roc_auc_score(y, s) for s in scores]))


def paired_diff(y, a, b, rng_seed=BOOT_SEED) -> dict:
    rng = np.random.default_rng(rng_seed)
    point = seed_mean_auc(y, a) - seed_mean_auc(y, b)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        draws.append(seed_mean_auc(y, a, idx) - seed_mean_auc(y, b, idx))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    per_seed = [float(roc_auc_score(y, sa) - roc_auc_score(y, sb)) for sa, sb in zip(a, b)]
    return {"value": float(point), "ci": [float(lo), float(hi)], "per_seed": per_seed}


def secondary_metrics(runs: dict[int, dict], source: str, target: str) -> dict:
    """AURC, coverage accuracy, ECE, Brier (after a Platt map fitted on source validation), seed-averaged."""
    out = {k: [] for k in p1.METRICS}
    for s in SEEDS:
        sc = runs[s]["scores"]
        val = sc[(sc["dataset"] == source) & (sc["split"] == "validation")]
        test = test_scores(runs[s], target)
        platt = p1.fit_platt(val["trust_logit"].values, val["label"].values)
        prob = platt(test["trust_logit"].values)
        m = p1.score_all(test["label"].values.astype(int), prob)
        for k in p1.METRICS:
            out[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def generation(runs: dict[int, dict]) -> dict:
    gens = [runs[s]["metrics"]["generation"] for s in SEEDS if runs[s]["metrics"].get("generation")]
    if not gens:
        return {}
    return {
        "accuracy": {ds: float(np.mean([g["accuracy"][ds] for g in gens])) for ds in DATASETS},
        "wikitext_perplexity": float(np.mean([g["wikitext_perplexity"] for g in gens])),
        "refusals": {ds: float(np.mean([g["refusals"][ds] for g in gens])) for ds in DATASETS},
    }


def meets_bar(gen: dict, base: dict) -> dict:
    drops = {ds: base["accuracy"][ds] - gen["accuracy"][ds] for ds in DATASETS}
    rise = gen["wikitext_perplexity"] / base["wikitext_perplexity"] - 1
    return {"accuracy_drop": drops, "perplexity_rise": rise,
            "passes": bool(all(d <= ACC_DROP_MAX for d in drops.values()) and rise <= PPL_RISE_MAX)}


def reference_baselines(phase1_dir: Path, source: str, target: str) -> dict:
    """Phase 1 single-pass and sampling baselines on the target test answers, with v3 labels,
    plus a logistic probe on frozen 50%-depth features trained on the source (the Phase 1 method)."""
    out = {}
    def load(ds):
        recs = [json.loads(l) for l in (phase1_dir / ds / "records.jsonl").open(encoding="utf-8")]
        df = pd.DataFrame(recs)
        df["label_v3"] = [int(is_correct_v3(a, g, q)) for a, g, q in zip(df["answer"], df["accepted_answers"], df["question"])]
        return df
    tgt = load(target)
    t = tgt[tgt["split"] == "test"]
    out["raw_conf_span"] = float(roc_auc_score(t["label_v3"], t["raw_conf_span"]))
    out["p_true"] = float(roc_auc_score(t["label_v3"], t["p_true"]))
    se = t["semantic_entropy"].astype(float)
    if se.notna().all():
        out["semantic_entropy"] = float(roc_auc_score(t["label_v3"], -se))
    chunks = sorted(glob.glob(str(phase1_dir / source / "hidden_chunk_*.npz")))
    if chunks:
        src = load(source)
        feats = {}
        for ds, df in ((source, src), (target, tgt)):
            by_id = {i: k for k, i in enumerate(df["id"])}
            X = np.zeros((len(df), 0))
            for path in sorted(glob.glob(str(phase1_dir / ds / "hidden_chunk_*.npz"))):
                z = np.load(path)
                if X.shape[1] == 0:
                    X = np.zeros((len(df), z["q50_answer_mean"].shape[1]))
                X[[by_id[i] for i in z["ids"].tolist()]] = z["q50_answer_mean"].astype(np.float64)
            feats[ds] = X
        def split(df, X, name):
            m = (df["split"] == name).values
            return {"X": X[m], "y": df["label_v3"].values[m]}
        probe = p1.fit_probe(split(src, feats[source], "train"), split(src, feats[source], "validation"), 0)
        tt = split(tgt, feats[target], "test")
        out["frozen_logistic_probe_q50"] = float(roc_auc_score(tt["y"], probe["score"](tt["X"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=ROOT / "runs" / "phase2")
    ap.add_argument("--phase1-dir", type=Path, default=ROOT / "runs" / "phase1_kaggle" / "qwen")
    ap.add_argument("--audit-exclude", type=Path, default=ROOT / "runs" / "phase1_kaggle_analysis" / "label_audit_sample.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase2_analysis")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results, rows = {}, []
    for source in DATASETS:
        target = other(source)
        lam, lam_aucs = choose_lambda(args.in_dir, source)
        arm_dirs = {"A": "A", "B": "B", "C": f"C-lam{lam}", "D": "D"}
        runs = {arm: {s: load_run(args.in_dir / source / d / f"seed{s}") for s in SEEDS} for arm, d in arm_dirs.items()}

        res = {"chosen_lambda": lam, "lambda_validation_auroc": lam_aucs, "transfer": {}, "in_distribution": {}}
        for setting, ds in (("transfer", target), ("in_distribution", source)):
            mats = {arm: seed_matrix(runs[arm], ds) for arm in arm_dirs}
            y = mats["A"][0]
            block = {"auroc": {arm: seed_mean_auc(y, m[1]) for arm, m in mats.items()}}
            for a, b in (("C", "A"), ("C", "B"), ("C", "D"), ("B", "A"), ("D", "A")):
                block[f"d({a},{b})"] = paired_diff(y, mats[a][1], mats[b][1])
            nr = {arm: seed_matrix(runs[arm], ds, refusals=False) for arm in arm_dirs}
            block["refusals_excluded"] = {f"d({a},{b})": paired_diff(nr["A"][0], nr[a][1], nr[b][1])
                                          for a, b in (("C", "A"), ("C", "B"))}
            block["secondary"] = {arm: secondary_metrics(runs[arm], source, ds) for arm in arm_dirs}
            res[setting] = block
            for arm in arm_dirs:
                rows.append({"train": source, "test": ds, "setting": setting, "arm": arm, "auroc": block["auroc"][arm], **block["secondary"][arm]})

        base_gen = runs["A"][0]["metrics"]["generation"]
        res["generation"] = {"base": {"accuracy": base_gen["accuracy"], "wikitext_perplexity": base_gen["wikitext_perplexity"]}}
        for arm in ("B", "C", "D"):
            g = generation(runs[arm])
            res["generation"][arm] = {**g, "bar": meets_bar(g, base_gen)}
        if not args.skip_baselines:
            res["reference_baselines_transfer"] = reference_baselines(args.phase1_dir, source, target)
        results[source] = res
        print(f"done: train {source} (lambda {lam})")

    def ci_above_zero(src, key):
        return results[src]["transfer"][key]["ci"][0] > 0

    bar_c = all(results[s]["generation"]["C"]["bar"]["passes"] for s in DATASETS)
    bar_b = all(results[s]["generation"]["B"]["bar"]["passes"] for s in DATASETS)
    h3t = all(ci_above_zero(s, "d(C,A)") for s in DATASETS) and bar_c
    h3j = all(ci_above_zero(s, "d(C,B)") for s in DATASETS) and bar_c
    tie = not all(ci_above_zero(s, "d(C,B)") for s in DATASETS)
    if not tie:
        tie_reading = None
    elif bar_c and not bar_b:
        tie_reading = "generation-preserving trust training"
    elif bar_c and bar_b:
        tie_reading = "joint objective adds nothing measurable beyond trust-only fine-tuning"
    else:
        tie_reading = "joint arm fails the generation-quality bar"
    d_explains = all(results[s]["transfer"]["d(C,D)"]["ci"][0] <= 0 <= results[s]["transfer"]["d(C,D)"]["ci"][1] for s in DATASETS)

    decisions = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase2/TCL-Phase2-Protocol.md",
        "H3-T": {"passes": h3t, "rule": "d(C,A) CI lower bound > 0 in both transfer directions, and C meets the generation bar"},
        "H3-J": {"passes": h3j, "rule": "d(C,B) CI lower bound > 0 in both transfer directions, and C meets the generation bar"},
        "generation_bar": {"C": bar_c, "B": bar_b, "D": all(results[s]["generation"]["D"]["bar"]["passes"] for s in DATASETS)},
        "tie_reading": tie_reading,
        "transfer_gain_attributed_to_lm_fine_tuning": d_explains,
    }
    (args.out_dir / "decisions.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")
    (args.out_dir / "details.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_dir / "arm_metrics.csv", index=False)

    # Label audit samples (Section 9), drawn from labels only.
    excluded = set(pd.read_csv(args.audit_exclude, keep_default_na=False)["id"]) if args.audit_exclude.exists() else set()
    audit = []
    for ds in DATASETS:
        recs = pd.DataFrame([json.loads(l) for l in (args.phase1_dir / ds / "records.jsonl").open(encoding="utf-8")])
        t = recs[(recs["split"] == "test") & (~recs["id"].isin(excluded))]
        for _, r in t.sample(n=min(100, len(t)), random_state=AUDIT_SEED).iterrows():
            audit.append({"source": "base_model", "dataset": ds, "id": r["id"], "question": r["question"],
                          "accepted_answers": " | ".join(r["accepted_answers"]), "answer": r["answer"],
                          "v3_label": int(is_correct_v3(r["answer"], r["accepted_answers"], r["question"])), "manual_label": ""})
    for source in DATASETS:
        lam = results[source]["chosen_lambda"]
        regen = load_run(args.in_dir / source / f"C-lam{lam}" / "seed0")["regen"]
        qs = {}
        for ds in DATASETS:
            q = pd.read_csv(ROOT / "data" / "phase1_kaggle" / f"{ds}.csv", keep_default_na=False)
            qs.update({r["id"]: r for _, r in q.iterrows()})
        sample = regen[regen["dataset"] == source].sample(n=min(50, int((regen["dataset"] == source).sum())), random_state=AUDIT_SEED)
        for _, r in sample.iterrows():
            q = qs[r["id"]]
            audit.append({"source": f"arm_C_trained_on_{source}", "dataset": r["dataset"], "id": r["id"], "question": q["question"],
                          "accepted_answers": q["accepted_answers"], "answer": r["answer"], "v3_label": int(r["label"]), "manual_label": ""})
    pd.DataFrame(audit).to_csv(args.out_dir / "label_audit_sample.csv", index=False)

    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
