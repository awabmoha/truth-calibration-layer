"""Phase 3 H4 (self-consistency): train one-pass consistency heads and score the arms.

Protocol: docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md (locked 2026-11-22).

For each (model, held-out dataset) pair that meets the size rule (at least 150 answers
in each correctness class and each consistency class on the held-out test split):

  C-hat  consistency head: 50%-depth answer-mean hidden state -> predicted y_c, trained
         with soft-label BCE on the sampled self-consistency label of the OTHER pool
         datasets' test splits (the only splits with sampled answers), never the held-out
         dataset. Design fixed to the Phase 3a setting (runs/phase3a/selection.json).
  R      raw confidence, Platt-mapped on the training mixture's validation splits.
  RC     logistic regression on [logit(raw confidence), C-hat logit], fitted on the
         training mixture's validation splits (correctness labels, rule v3).
  RC_oracle  the same combination with the sampled label y_c instead of C-hat (reference).
  SE     semantic entropy, 10 samples (reference).

Implementation details fixed here (not in the protocol text):
  * Early stopping holds back a random 10% of the consistency head's training rows
    (seeded per seed); the rest are trained on. Equal total weight per training dataset.
  * The sampled label is smoothed before taking its logit: (10 * y_c + 0.5) / 11.
  * RC_oracle cannot be fitted on validation splits (they have no sampled answers), so
    its combiner is fitted on the training datasets' test splits. It is ungated.
  * Every arm outputs a probability of correctness; R and SE have one seed, C-hat, RC
    and RC_oracle five (0-4); RC_oracle is seed-independent but kept per seed for
    alignment.

Writes runs/phase3_h4/scores_<model>_<held>.csv and manifest.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase3_heads import (MIN_PER_CLASS, MODELS, NEW_DATASET, POOL, SEEDS,  # noqa: E402
                              Standardizer, fit_mlp, load_run, logit, run_dir)

ROOT = Path(__file__).resolve().parents[1]
CONS_THRESHOLD = 0.5
N_SAMPLES = 10


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def load(runs: Path, model: str, dataset: str) -> dict:
    """Phase 3 run (rule v3 labels, features) plus the sampled self-consistency label."""
    run = load_run(run_dir(runs, model, dataset))
    path = run_dir(runs, model, dataset) / "records.jsonl"
    cons = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            v = r.get("self_consistency")
            cons[r["id"]] = np.nan if v in (None, "None") else float(v)
    run["df"]["cons"] = run["df"]["id"].map(cons).astype(float)
    return run


def part(run: dict, split: str) -> dict:
    mask = (run["df"]["split"] == split).to_numpy()
    d = run["df"][mask]
    return {"ids": d["id"].to_numpy(), "y": d["label"].to_numpy(int), "raw": d["raw"].to_numpy(float),
            "refusal": d["refusal"].to_numpy(bool), "se": d["se"].to_numpy(object),
            "cons": d["cons"].to_numpy(float), "X": run["X"][mask]}


def cat(parts: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    total = sum(len(p["y"]) for p in parts)
    out["w"] = np.concatenate([np.full(len(p["y"]), total / (len(parts) * len(p["y"]))) for p in parts])
    return out


def eligible(test: dict) -> dict:
    pos = int(test["y"].sum())
    cons_hi = int(np.nansum(test["cons"] >= CONS_THRESHOLD))
    n_cons = int(np.sum(~np.isnan(test["cons"])))
    counts = {"correct": pos, "incorrect": len(test["y"]) - pos,
              "consistent": cons_hi, "inconsistent": n_cons - cons_hi}
    return {**counts, "eligible": min(counts.values()) >= MIN_PER_CLASS}


def smooth(c: np.ndarray) -> np.ndarray:
    return (N_SAMPLES * c + 0.5) / (N_SAMPLES + 1)


def train_consistency_head(train: dict, setting: dict, seed: int):
    keep = ~np.isnan(train["cons"])
    X, c, w = train["X"][keep], train["cons"][keep], train["w"][keep]
    rng = np.random.default_rng(seed)
    hold = rng.random(len(c)) < 0.10
    std = Standardizer(X[~hold])
    f = fit_mlp(std(X[~hold]), c[~hold], w[~hold], std(X[hold]), c[hold], w[hold], setting["reg"], seed)
    return lambda d: f(std(d["X"]))


def fit_combiner(cols_va: np.ndarray, y_va: np.ndarray, w_va: np.ndarray):
    clf = LogisticRegression(C=1e6, max_iter=2000)
    clf.fit(cols_va, y_va, sample_weight=w_va)
    return lambda cols: clf.predict_proba(cols)[:, 1]


def score(runs: Path, selection: Path, out_dir: Path) -> None:
    sel = json.loads(selection.read_text(encoding="utf-8"))
    setting = sel["chosen"]
    if setting["head"] != "mlp":
        raise SystemExit("protocol fixes the Phase 3a MLP setting; selection.json disagrees")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": "docs/phase3/TCL-Phase3-H4-Consistency-Protocol.md",
                "setting": setting, "cons_threshold": CONS_THRESHOLD, "seeds": SEEDS, "pairs": []}
    for m in MODELS:
        data = {d: load(runs, m, d) for d in POOL}
        if run_dir(runs, m, NEW_DATASET).exists():
            data[NEW_DATASET] = load(runs, m, NEW_DATASET)
        for held in POOL + [NEW_DATASET]:
            if held not in data:
                continue
            test = part(data[held], "test")
            info = eligible(test)
            manifest["pairs"].append({"model": m, "held_out": held, **info})
            if not info["eligible"]:
                continue
            train_sets = [d for d in POOL if d != held]
            tr_cons = cat([part(data[d], "test") for d in train_sets])   # sampled labels live on test splits
            va = cat([part(data[d], "validation") for d in train_sets])
            rows = []

            def add(arm, seed, prob):
                rows.extend({"id": i, "arm": arm, "seed": seed, "label": int(y), "cons": float(c),
                             "refusal": bool(r), "score": float(p)}
                            for i, y, c, r, p in zip(test["ids"], test["y"], test["cons"], test["refusal"], prob))

            r_map = fit_combiner(logit(va["raw"])[:, None], va["y"], va["w"])
            add("R", 0, r_map(logit(test["raw"])[:, None]))
            se = np.array([np.nan if v in (None, "None") else -float(v) for v in test["se"]])
            add("SE", 0, se)
            ok = ~np.isnan(tr_cons["cons"])
            oracle = fit_combiner(np.column_stack([logit(tr_cons["raw"][ok]), logit(smooth(tr_cons["cons"][ok]))]),
                                  tr_cons["y"][ok], tr_cons["w"][ok])
            oracle_test = oracle(np.column_stack([logit(test["raw"]), logit(smooth(np.nan_to_num(test["cons"], nan=0.5)))]))
            for seed in SEEDS:
                head = train_consistency_head(tr_cons, setting, seed)
                z_va, z_te = head(va), head(test)
                add("CHAT", seed, sigmoid(z_te))
                comb = fit_combiner(np.column_stack([logit(va["raw"]), z_va]), va["y"], va["w"])
                add("RC", seed, comb(np.column_stack([logit(test["raw"]), z_te])))
                add("RC_oracle", seed, oracle_test)
            pd.DataFrame(rows).to_csv(out_dir / f"scores_{m}_{held}.csv", index=False)
            print(f"scored {m} held-out {held}: {len(test['y'])} test answers", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--selection", type=Path, default=ROOT / "runs" / "phase3a" / "selection.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase3_h4")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model tags (default: the Phase 3 models)")
    args = ap.parse_args()
    MODELS[:] = args.models.split(",")  # in place: the H4 scripts share this list
    score(args.runs, args.selection, args.out_dir)


if __name__ == "__main__":
    main()
