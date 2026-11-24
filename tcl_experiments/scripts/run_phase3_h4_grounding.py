"""Phase 3 H4 (grounding): train one-pass grounding heads and score the arms.

Protocol: docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md (locked 2026-11-22).

Data: runs/phase3_grounding/<model>/<dataset>/ (records.jsonl, hidden_chunk_*.npz,
grounding.jsonl from label_grounding.py) and data/phase3_grounding/<dataset>.csv for
the distractor flag. Pool: squad, nq, hotpotqa; new held-out dataset:
relation_extraction (test split only).

For each (model, held-out dataset) pair meeting the size rule (at least 150 answers in
each correctness class and each grounding class among the held-out test answers that
are not abstentions):

  G-hat      grounding head: 50%-depth answer-mean hidden state -> P(grounded), trained
             on the other pool datasets' TRAINING splits, early-stopped on their
             validation splits. Design fixed to the Phase 3a setting.
  R          raw confidence, Platt-mapped on the training datasets' validation splits.
  RG         logistic regression on [logit(raw confidence), G-hat logit], fitted on the
             same validation splits (correctness labels, rule v3).
  RG_oracle  the same combination with the NLI grounding label (reference).
  PTRUE      P(True) (reference).

Implementation details fixed here (not in the protocol text):
  * Abstentions (label_grounding.is_abstention) are left out of training, fitting, and
    evaluation for every arm, so all arms score the same answers; they are counted in
    the manifest.
  * The oracle's binary label enters the combiner as +-1.
  * Every arm outputs a probability of correctness; R and PTRUE have one seed, the
    others five (0-4).

Writes runs/phase3_h4g/scores_<model>_<held>.csv and manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_phase3_h4 import fit_combiner, sigmoid  # noqa: E402
from run_phase3_heads import MIN_PER_CLASS, MODELS, SEEDS, Standardizer, fit_mlp, load_run, logit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = ["squad", "nq", "hotpotqa"]
NEW = "relation_extraction"


def load(runs: Path, data_dir: Path, model: str, dataset: str) -> dict:
    run_dir = runs / "phase3_grounding" / model / dataset
    run = load_run(run_dir)
    g = {}
    for line in (run_dir / "grounding.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            g[r["id"]] = r
    with (data_dir / f"{dataset}.csv").open(encoding="utf-8", newline="") as f:
        distractor = {r["id"]: int(r["distractor"]) for r in csv.DictReader(f)}
    p_true = {}
    for line in (run_dir / "records.jsonl").open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            p_true[r["id"]] = float(r["p_true"]) if r.get("p_true") not in (None, "None") else np.nan
    df = run["df"]
    df["abstain"] = df["id"].map(lambda i: bool(g[i]["abstain"]))
    df["grounded"] = df["id"].map(lambda i: np.nan if g[i]["grounded"] is None else float(g[i]["grounded"]))
    df["distractor"] = df["id"].map(distractor).astype(int)
    df["p_true"] = df["id"].map(p_true).astype(float)
    return run


def part(run: dict, split: str) -> dict:
    df = run["df"]
    mask = ((df["split"] == split) & ~df["abstain"]).to_numpy()
    d = df[mask]
    return {"ids": d["id"].to_numpy(), "y": d["label"].to_numpy(int), "raw": d["raw"].to_numpy(float),
            "refusal": d["refusal"].to_numpy(bool), "g": d["grounded"].to_numpy(float),
            "distractor": d["distractor"].to_numpy(int), "p_true": d["p_true"].to_numpy(float),
            "X": run["X"][mask]}


def counts(run: dict, split: str) -> dict:
    df = run["df"][run["df"]["split"] == split]
    claims = df[~df["abstain"]]
    c = {"answers": int(len(df)), "abstentions": int(df["abstain"].sum()),
         "correct": int(claims["label"].sum()), "incorrect": int(len(claims) - claims["label"].sum()),
         "grounded": int(claims["grounded"].sum()), "not_grounded": int(len(claims) - claims["grounded"].sum())}
    c["eligible"] = min(c["correct"], c["incorrect"], c["grounded"], c["not_grounded"]) >= MIN_PER_CLASS
    return c


def cat(parts: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    total = sum(len(p["y"]) for p in parts)
    out["w"] = np.concatenate([np.full(len(p["y"]), total / (len(parts) * len(p["y"]))) for p in parts])
    return out


def train_grounding_head(tr: dict, va: dict, setting: dict, seed: int):
    std = Standardizer(tr["X"])
    f = fit_mlp(std(tr["X"]), tr["g"], tr["w"], std(va["X"]), va["g"], va["w"], setting["reg"], seed)
    return lambda d: f(std(d["X"]))


def score(runs: Path, data_dir: Path, selection: Path, out_dir: Path) -> None:
    setting = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
    if setting["head"] != "mlp":
        raise SystemExit("protocol fixes the Phase 3a MLP setting; selection.json disagrees")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": "docs/phase3/TCL-Phase3-H4-Grounding-Protocol.md",
                "setting": setting, "seeds": SEEDS, "pairs": []}
    for m in MODELS:
        data = {d: load(runs, data_dir, m, d) for d in POOL + [NEW]
                if (runs / "phase3_grounding" / m / d).exists()}
        for held in POOL + [NEW]:
            if held not in data:
                continue
            info = counts(data[held], "test")
            manifest["pairs"].append({"model": m, "held_out": held, **info})
            if not info["eligible"]:
                continue
            train_sets = [d for d in POOL if d != held]
            tr = cat([part(data[d], "train") for d in train_sets])
            va = cat([part(data[d], "validation") for d in train_sets])
            test = part(data[held], "test")
            rows = []

            def add(arm, seed, prob):
                rows.extend({"id": i, "arm": arm, "seed": seed, "label": int(y), "grounded": int(g),
                             "distractor": int(dd), "refusal": bool(r), "score": float(p)}
                            for i, y, g, dd, r, p in zip(test["ids"], test["y"], test["g"], test["distractor"],
                                                         test["refusal"], prob))

            r_map = fit_combiner(logit(va["raw"])[:, None], va["y"], va["w"])
            add("R", 0, r_map(logit(test["raw"])[:, None]))
            add("PTRUE", 0, test["p_true"])
            oracle = fit_combiner(np.column_stack([logit(va["raw"]), 2 * va["g"] - 1]), va["y"], va["w"])
            oracle_test = oracle(np.column_stack([logit(test["raw"]), 2 * test["g"] - 1]))
            for seed in SEEDS:
                head = train_grounding_head(tr, va, setting, seed)
                z_va, z_te = head(va), head(test)
                add("GHAT", seed, sigmoid(z_te))
                comb = fit_combiner(np.column_stack([logit(va["raw"]), z_va]), va["y"], va["w"])
                add("RG", seed, comb(np.column_stack([logit(test["raw"]), z_te])))
                add("RG_oracle", seed, oracle_test)
            pd.DataFrame(rows).to_csv(out_dir / f"scores_{m}_{held}.csv", index=False)
            print(f"scored {m} held-out {held}: {len(test['y'])} test answers (abstentions excluded)", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "phase3_grounding")
    ap.add_argument("--selection", type=Path, default=ROOT / "runs" / "phase3a" / "selection.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "phase3_h4g")
    ap.add_argument("--models", default=",".join(MODELS), help="comma-separated model tags (default: the Phase 3 models)")
    args = ap.parse_args()
    MODELS[:] = args.models.split(",")  # in place: the H4 scripts share this list
    score(args.runs, args.data_dir, args.selection, args.out_dir)


if __name__ == "__main__":
    main()
