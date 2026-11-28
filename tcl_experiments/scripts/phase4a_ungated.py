"""Phase 4a ungated reports (Phase 4a protocol, Section 7, "Also reported, not gated").

Reads the recorded analysis outputs only (details.json); trains nothing and decides nothing.
  1. Scale comparison: each primary metric at 0.5B (Qwen2.5-0.5B, tag qwen), 3.8B
     (Phi-3.5-mini, tag phi), and 7B (Qwen2.5-7B, tag qwen7b) on the held-out datasets
     all three share as eligible pairs: TriviaQA and PopQA for H4-T and self-consistency,
     NQ and HotpotQA for grounding (Phase 3 grounding with its recorded labels).
  2. Weak-confidence pattern: across every eligible pair from Phase 3 and Phase 4a, the
     Spearman and Pearson correlation between raw confidence's AUROC and each head's
     AUROC gain over it (H4-T: M+R minus R; self-consistency: R+C-hat minus R;
     grounding: R+G-hat minus R).

Usage: python scripts/phase4a_ungated.py
Output: runs/phase4a_ungated/scale_comparison.csv, weak_confidence.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "phase4a_ungated"
SIZE = {"qwen": "0.5B", "phi": "3.8B", "qwen7b": "7B"}
SOURCES = {  # test -> analysis dirs (Phase 3, Phase 4a)
    "H4-T": ["phase3b_analysis", "phase4a_h4t_analysis"],
    "self-consistency": ["phase3_h4_analysis", "phase4a_h4c_analysis"],
    "grounding": ["phase3_h4g_analysis", "phase4a_h4g_analysis"],
}
SHARED = {"H4-T": ["triviaqa", "popqa"], "self-consistency": ["triviaqa", "popqa"], "grounding": ["nq", "hotpotqa"]}


def pairs(test: str) -> dict[str, dict]:
    out = {}
    for d in SOURCES[test]:
        out.update({k: v["all"] for k, v in json.loads((RUNS / d / "details.json").read_text(encoding="utf-8")).items()})
    return out


def row(test: str, key: str, a: dict) -> dict:
    model, held = key.split("/")
    r = {"test": test, "model": model, "size": SIZE[model], "held_out": held, "n": a["n"]}
    if test == "H4-T":
        r.update(raw_auroc=a["auroc"]["R"], head_auroc=a["auroc"]["MR"], se_auroc=a["auroc"].get("SE"),
                 gain_auroc=a["d(MR,R)"]["value"], gain_ci_lo=a["d(MR,R)"]["ci"][0], gain_ci_hi=a["d(MR,R)"]["ci"][1])
    else:
        c = "RC" if test == "self-consistency" else "RG"
        ref = "SE" if test == "self-consistency" else "PTRUE"
        d = a[f"d_aurc(R,{c})"]
        r.update(raw_auroc=a["correctness_auroc"]["R"], head_auroc=a["correctness_auroc"][c],
                 se_auroc=a["correctness_auroc"].get(ref) if ref == "SE" else None,
                 gain_auroc=a["correctness_auroc"][c] - a["correctness_auroc"]["R"],
                 learnability=a["learnability_auroc"]["value"], spearman_with_raw=a["spearman_with_raw"]["value"],
                 d_aurc=d["value"], d_aurc_ci_lo=d["ci"][0], d_aurc_ci_hi=d["ci"][1], passes=a["passes"])
    return r


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, weak = [], {}
    for test in SOURCES:
        all_pairs = {k: row(test, k, a) for k, a in pairs(test).items()}
        rows += [r for r in all_pairs.values() if r["held_out"] in SHARED[test]]
        x = [r["raw_auroc"] for r in all_pairs.values()]
        y = [r["gain_auroc"] for r in all_pairs.values()]
        weak[test] = {"pairs": sorted(all_pairs), "n": len(x),
                      "spearman": spearmanr(x, y).statistic, "pearson": pearsonr(x, y).statistic,
                      "points": [{"pair": k, "raw_auroc": r["raw_auroc"], "gain_auroc": r["gain_auroc"]}
                                 for k, r in sorted(all_pairs.items())]}
    pooled_x = [p["raw_auroc"] for t in weak.values() for p in t["points"]]
    pooled_y = [p["gain_auroc"] for t in weak.values() for p in t["points"]]
    weak["pooled"] = {"n": len(pooled_x), "spearman": spearmanr(pooled_x, pooled_y).statistic,
                      "pearson": pearsonr(pooled_x, pooled_y).statistic}
    rows.sort(key=lambda r: (list(SOURCES).index(r["test"]), r["held_out"], list(SIZE).index(r["model"])))
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with (OUT / "scale_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / "weak_confidence.json").write_text(json.dumps(weak, indent=2), encoding="utf-8")
    for r in rows:
        extra = f"  learn {r['learnability']:.3f} rho {r['spearman_with_raw']:+.3f} d_AURC {r['d_aurc']:+.4f}" if "d_aurc" in r else ""
        print(f"{r['test']:16s} {r['held_out']:9s} {r['size']:5s} raw {r['raw_auroc']:.3f} head {r['head_auroc']:.3f} "
              f"gain {r['gain_auroc']:+.3f}{extra}")
    for t, v in weak.items():
        print(f"weak-confidence {t:16s} n={v['n']:2d} spearman {v['spearman']:+.2f} pearson {v['pearson']:+.2f}")


if __name__ == "__main__":
    main()
