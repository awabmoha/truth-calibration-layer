"""Phase 3 (H4-T): train and score trust heads on mixtures of datasets.

Protocol: docs/phase3/TCL-Phase3-Protocol.md (locked 2026-11-21).

Two modes:

  select  (stage 3a) Leave-one-dataset-out over the Phase 1 pool (TriviaQA, SQuAD,
          NQ-Open) for Qwen and Phi. Trains on the other datasets' TRAINING splits and
          scores the held-out dataset's VALIDATION split. Never reads a test split.
          Picks one head setting by the protocol's rule (Section 5.3) and writes
          runs/phase3a/selection.json.

  score   (stage 3b) With the selected setting, trains every arm and writes test-split
          scores for each eligible (model, held-out dataset) pair, including PopQA.
          Refuses to run without runs/phase3a/selection.json.

Arms (protocol Section 5.2):
  R    raw span confidence, Platt-mapped on the training mixture's validation splits
  S_d  hidden-state head trained on one training dataset d
  M    hidden-state head trained on the mixture
  MR   hidden states plus logit(raw confidence), trained on the mixture
  (SE  semantic entropy, reference only, taken from the records)

Implementation details fixed here (not in the protocol text):
  * Features: q50_answer_mean, standardized with training-mixture statistics.
  * Logistic head: scikit-learn LogisticRegression (lbfgs, C = 1 / regularization,
    max_iter 2000). It is deterministic, so its five seeds give identical scores.
  * MLP head: LayerNorm, Linear(d, 256), GELU, dropout 0.1, Linear(256, 1); AdamW,
    learning rate 1e-3, weight decay = regularization, batch 256, up to 60 epochs,
    keeping the epoch with the lowest weighted BCE on the training mixture's
    validation splits (never the held-out dataset).
  * Mixture weights: "equal" gives each training dataset the same total weight;
    "proportional" gives every example weight 1.
  * 3a uses seeds 0-4 and ranks settings by mean validation AUROC of MR, rounded to
    3 decimals; ties go to logistic over MLP, stronger regularization, equal weights.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labeling import CORRECTNESS_METHOD_V3, is_correct_v3, is_refusal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ["triviaqa", "squad", "nq_open"]
MODELS = ["qwen", "phi"]
NEW_DATASET = "popqa"
FEATURE = "q50_answer_mean"
SEEDS = [0, 1, 2, 3, 4]
REGS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
HEADS = ["logistic", "mlp"]
WEIGHTINGS = ["equal", "proportional"]
MIN_PER_CLASS = 150
EPS = 1e-6


# ---------------------------------------------------------------- data

def load_run(run_dir: Path) -> dict:
    """Records (relabeled with rule v3) and the 50%-depth answer-mean features."""
    records = [json.loads(l) for l in (run_dir / "records.jsonl").open(encoding="utf-8") if l.strip()]
    by_id = {r["id"]: i for i, r in enumerate(records)}
    X = None
    for path in sorted(glob.glob(str(run_dir / "hidden_chunk_*.npz"))):
        z = np.load(path)
        if X is None:
            X = np.full((len(records), z[FEATURE].shape[1]), np.nan, dtype=np.float32)
        X[[by_id[i] for i in z["ids"].tolist()]] = z[FEATURE].astype(np.float32)
    if X is None or np.isnan(X).any():
        raise ValueError(f"missing hidden states in {run_dir}")
    df = pd.DataFrame({
        "id": [r["id"] for r in records],
        "split": [r["split"] for r in records],
        "label": [int(is_correct_v3(r["answer"], r["accepted_answers"], r["question"])) for r in records],
        "refusal": [bool(is_refusal(r["answer"])) for r in records],
        "raw": np.clip([float(r["raw_conf_span"]) for r in records], EPS, 1 - EPS),
        "se": [r.get("semantic_entropy") for r in records],
    })
    return {"df": df, "X": X}


def run_dir(runs: Path, model: str, dataset: str) -> Path:
    if dataset == NEW_DATASET:
        return runs / "phase3_popqa" / model / dataset
    return runs / "phase1_kaggle" / model / dataset


def part(run: dict, split: str) -> dict:
    mask = (run["df"]["split"] == split).to_numpy()
    d = run["df"][mask]
    return {"ids": d["id"].to_numpy(), "y": d["label"].to_numpy(int), "raw": d["raw"].to_numpy(float),
            "refusal": d["refusal"].to_numpy(bool), "se": d["se"].to_numpy(object), "X": run["X"][mask]}


def eligible(test: dict) -> bool:
    pos = int(test["y"].sum())
    return pos >= MIN_PER_CLASS and len(test["y"]) - pos >= MIN_PER_CLASS


# ---------------------------------------------------------------- heads

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def mixture(parts: list[dict], weighting: str) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("y", "raw", "X")}
    if weighting == "equal":
        total = sum(len(p["y"]) for p in parts)
        out["w"] = np.concatenate([np.full(len(p["y"]), total / (len(parts) * len(p["y"]))) for p in parts])
    else:
        out["w"] = np.ones(len(out["y"]))
    return out


class Standardizer:
    def __init__(self, X: np.ndarray):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def design(d: dict, use_hidden: bool, use_raw: bool, std: Standardizer, raw_std: tuple[float, float]) -> np.ndarray:
    cols = []
    if use_hidden:
        cols.append(std(d["X"]))
    if use_raw:
        cols.append(((logit(d["raw"]) - raw_std[0]) / raw_std[1])[:, None])
    return np.hstack(cols).astype(np.float32)


def fit_logistic(Xtr, ytr, wtr, reg: float):
    clf = LogisticRegression(C=1.0 / reg, max_iter=2000)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    return lambda X: clf.decision_function(X)


def fit_mlp(Xtr, ytr, wtr, Xva, yva, wva, reg: float, seed: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Linear(Xtr.shape[1], 256), nn.GELU(),
                        nn.Dropout(0.1), nn.Linear(256, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=reg)
    Xt, yt, wt = (torch.tensor(a, dtype=torch.float32) for a in (Xtr, ytr, wtr))
    Xv, yv, wv = (torch.tensor(a, dtype=torch.float32) for a in (Xva, yva, wva))
    bce = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state = np.inf, None
    for _ in range(60):
        net.train()
        for idx in np.array_split(rng.permutation(len(yt)), max(1, len(yt) // 256)):
            opt.zero_grad()
            loss = (bce(net(Xt[idx]).squeeze(1), yt[idx]) * wt[idx]).sum() / wt[idx].sum()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float((bce(net(Xv).squeeze(1), yv) * wv).sum() / wv.sum())
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(torch.tensor(X, dtype=torch.float32)).squeeze(1).numpy()
    return predict


def train_head(train: dict, val: dict, setting: dict, use_raw: bool, seed: int):
    """Returns a scoring function on parts (dicts with X and raw)."""
    std = Standardizer(train["X"])
    lr = logit(train["raw"])
    raw_std = (float(lr.mean()), float(lr.std() + 1e-6))
    Xtr = design(train, True, use_raw, std, raw_std)
    if setting["head"] == "logistic":
        f = fit_logistic(Xtr, train["y"], train["w"], setting["reg"])
    else:
        Xva = design(val, True, use_raw, std, raw_std)
        f = fit_mlp(Xtr, train["y"], train["w"], Xva, val["y"], val["w"], setting["reg"], seed)
    return lambda d: f(design(d, True, use_raw, std, raw_std))


def fit_platt_raw(val: dict):
    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(logit(val["raw"])[:, None], val["y"], sample_weight=val["w"])
    return lambda d: clf.decision_function(logit(d["raw"])[:, None])


# ---------------------------------------------------------------- stage 3a

def settings() -> list[dict]:
    return [{"head": h, "reg": r, "weighting": w} for h in HEADS for r in REGS for w in WEIGHTINGS]


def simplicity_key(s: dict) -> tuple:
    return (HEADS.index(s["head"]), -s["reg"], WEIGHTINGS.index(s["weighting"]))


def select(runs: Path, out_dir: Path) -> None:
    data = {m: {d: load_run(run_dir(runs, m, d)) for d in POOL} for m in MODELS}
    results = []
    for s in settings():
        aucs = []
        for m in MODELS:
            for held in POOL:
                train_sets = [d for d in POOL if d != held]
                tr = mixture([part(data[m][d], "train") for d in train_sets], s["weighting"])
                va = mixture([part(data[m][d], "validation") for d in train_sets], s["weighting"])
                target = part(data[m][held], "validation")  # validation only: no test split in 3a
                seed_aucs = []
                for seed in SEEDS:
                    score = train_head(tr, va, s, use_raw=True, seed=seed)
                    seed_aucs.append(roc_auc_score(target["y"], score(target)))
                aucs.append({"model": m, "held_out": held, "auroc": float(np.mean(seed_aucs))})
        mean = float(np.mean([a["auroc"] for a in aucs]))
        results.append({**s, "mean_val_auroc_MR": mean, "rotations": aucs})
        print(f"{s['head']:8s} reg={s['reg']:<7g} {s['weighting']:12s} mean MR val AUROC {mean:.4f}", flush=True)
    best = sorted(results, key=lambda r: (-round(r["mean_val_auroc_MR"], 3), simplicity_key(r)))[0]
    chosen = {k: best[k] for k in ("head", "reg", "weighting")}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selection.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase3/TCL-Phase3-Protocol.md",
        "correctness_method": CORRECTNESS_METHOD_V3,
        "feature": FEATURE, "seeds": SEEDS,
        "chosen": chosen, "chosen_mean_val_auroc_MR": best["mean_val_auroc_MR"],
        "all_settings": results,
    }, indent=2), encoding="utf-8")
    print("chosen:", chosen, flush=True)


# ---------------------------------------------------------------- stage 3b

def score(runs: Path, selection: Path, out_dir: Path) -> None:
    if not selection.exists():
        raise SystemExit(f"{selection} not found: run the 3a selection first (protocol Section 8).")
    chosen = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
    data = {}
    for m in MODELS:
        data[m] = {d: load_run(run_dir(runs, m, d)) for d in POOL}
        if run_dir(runs, m, NEW_DATASET).exists():
            data[m][NEW_DATASET] = load_run(run_dir(runs, m, NEW_DATASET))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "chosen": chosen, "pairs": []}
    for m in MODELS:
        for held in POOL + [NEW_DATASET]:
            if held not in data[m]:
                continue
            test = part(data[m][held], "test")
            n_pos = int(test["y"].sum())
            ok = eligible(test)
            manifest["pairs"].append({"model": m, "held_out": held, "test_correct": n_pos,
                                      "test_incorrect": int(len(test["y"]) - n_pos), "eligible": ok})
            if not ok:
                continue
            train_sets = [d for d in POOL if d != held]
            parts_tr = {d: part(data[m][d], "train") for d in train_sets}
            parts_va = {d: part(data[m][d], "validation") for d in train_sets}
            tr = mixture(list(parts_tr.values()), chosen["weighting"])
            va = mixture(list(parts_va.values()), chosen["weighting"])
            rows = []

            def add(arm, seed, split, d, s):
                rows.extend({"id": i, "split": split, "arm": arm, "seed": seed, "label": int(y),
                             "refusal": bool(r), "score": float(v)}
                            for i, y, r, v in zip(d["ids"], d["y"], d["refusal"], s))

            platt = fit_platt_raw(va)
            add("R", 0, "test", test, platt(test))
            add("R", 0, "validation", va_ids(parts_va), platt(va_ids(parts_va)))
            se = np.array([np.nan if v is None else -float(v) for v in test["se"]])
            add("SE", 0, "test", test, se)
            for seed in SEEDS:
                for arm, use_raw in (("M", False), ("MR", True)):
                    f = train_head(tr, va, chosen, use_raw, seed)
                    add(arm, seed, "test", test, f(test))
                    add(arm, seed, "validation", va_ids(parts_va), f(va_ids(parts_va)))
                for d in train_sets:
                    tr_d = mixture([parts_tr[d]], "proportional")
                    va_d = mixture([parts_va[d]], "proportional")
                    f = train_head(tr_d, va_d, chosen, False, seed)
                    add(f"S_{d}", seed, "test", test, f(test))
                    add(f"S_{d}", seed, "validation", va_ids([parts_va[d]]), f(va_ids([parts_va[d]])))
            pd.DataFrame(rows).to_csv(out_dir / f"scores_{m}_{held}.csv", index=False)
            print(f"scored {m} held-out {held}: {len(test['y'])} test answers", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def va_ids(parts) -> dict:
    """Concatenate validation parts, keeping ids, labels and refusals (for calibration maps)."""
    parts = list(parts.values()) if isinstance(parts, dict) else parts
    return {k: np.concatenate([p[k] for p in parts]) for k in ("ids", "y", "raw", "refusal", "X")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["select", "score"])
    ap.add_argument("--runs", type=Path, default=ROOT / "runs", help="root holding phase1_kaggle/ and phase3_popqa/")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--selection", type=Path, default=ROOT / "runs" / "phase3a" / "selection.json")
    args = ap.parse_args()
    if args.mode == "select":
        select(args.runs, args.out_dir or ROOT / "runs" / "phase3a")
    else:
        score(args.runs, args.selection, args.out_dir or ROOT / "runs" / "phase3b")


if __name__ == "__main__":
    main()
