"""Build the README figures for Phases 3 and 4a and the final analyses.

Writes:
  assets/phase3-4a-correctness-heads.svg  learned correctness head (M+R) vs three free
      token-probability baselines on every H4-T held-out pair (0.5B, 3.8B, 7B)
  assets/phase3-4a-trust-dimensions.svg   AURC gain over answer probability, with 95% CIs,
      for the self-consistency and grounding heads on every eligible pair

Data: tcl_experiments/runs/final_analyses/{baseline_choice.csv, mde.csv}, written by
tcl_experiments/scripts/final_analyses.py from the recorded Phase 3 and 4a results.

Run from the repository root: python assets/make_final_figures.py
"""

import csv
from pathlib import Path

from make_history_figures import scale, svg  # same palette and helpers

ROOT = Path(__file__).resolve().parents[1]
FA = ROOT / "tcl_experiments" / "runs" / "final_analyses"
MODEL = {"qwen": "Qwen 0.5B", "phi": "Phi 3.8B", "qwen7b": "Qwen 7B"}
DATA = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "popqa": "PopQA", "squad": "SQuAD", "nq": "NQ",
        "hotpotqa": "HotpotQA", "relation_extraction": "RelationExt."}
ORDER = ["qwen", "phi", "qwen7b"]
HOLLOW = 'style="fill:none;stroke:#8a8984;stroke-width:2"'


def label(pair: str) -> str:
    m, d = pair.split("/")
    return f"{MODEL[m]} / {DATA[d]}"


def read(name: str) -> list[dict]:
    with (FA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def correctness_heads() -> None:
    rows = sorted(read("baseline_choice.csv"), key=lambda r: (ORDER.index(r["pair"].split("/")[0]), r["pair"]))
    W, LEFT, RIGHT, TOP, ROW = 900, 200, 40, 128, 30
    H = TOP + ROW * len(rows) + 70
    x = scale(0.6, 0.95, LEFT, W - RIGHT)
    body = [
        '<text class="h1" x="24" y="34">Phases 3 and 4a: learned correctness heads vs free token probabilities</text>',
        '<text class="sub" x="24" y="54">Test AUROC on a dataset the head never saw. The head (blue) reads hidden states and also gets the</text>',
        '<text class="sub" x="24" y="70">answer-span probability as input. It beats the strongest free baseline on 1 of 9 pairs.</text>',
        '<circle class="c1" cx="30" cy="96" r="5"/><text class="leg" x="42" y="100">Learned head (M+R)</text>',
        '<circle class="c0" cx="190" cy="96" r="5"/><text class="leg" x="202" y="100">Answer-span probability (preregistered baseline)</text>',
        '<circle class="c2" cx="510" cy="96" r="5"/><text class="leg" x="522" y="100">Minimum token probability</text>',
        f'<circle cx="710" cy="96" r="4" {HOLLOW}/><text class="leg" x="722" y="100">First-token probability</text>',
    ]
    bottom = TOP + ROW * len(rows) - 12
    for t in (0.6, 0.7, 0.8, 0.9):
        body.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 14}" y2="{bottom}"/>')
        body.append(f'<text class="tick" x="{x(t):.1f}" y="{bottom + 16}" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        y = TOP + ROW * i
        vals = {k: float(r[k]) for k in ("auroc_span", "auroc_first_token", "auroc_min_token", "auroc_head_MR")}
        lo, hi = min(vals.values()), max(vals.values())
        body.append(f'<text class="row" x="{LEFT - 14}" y="{y + 5}" text-anchor="end">{label(r["pair"])}</text>')
        body.append(f'<line class="link" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y}" y2="{y}"/>')
        body.append(f'<circle cx="{x(vals["auroc_first_token"]):.1f}" cy="{y}" r="5" {HOLLOW}><title>first token {vals["auroc_first_token"]:.3f}</title></circle>')
        body.append(f'<circle class="c0" cx="{x(vals["auroc_span"]):.1f}" cy="{y}" r="6"><title>answer span {vals["auroc_span"]:.3f}</title></circle>')
        body.append(f'<circle class="c2" cx="{x(vals["auroc_min_token"]):.1f}" cy="{y}" r="6"><title>minimum token {vals["auroc_min_token"]:.3f}</title></circle>')
        body.append(f'<circle class="c1" cx="{x(vals["auroc_head_MR"]):.1f}" cy="{y}" r="6"><title>learned head {vals["auroc_head_MR"]:.3f}</title></circle>')
    body.append(f'<text class="note" x="24" y="{H - 30}">Head: mean of 5 seeds. The minimum-token and first-token comparisons are exploratory (not preregistered).</text>')
    body.append(f'<text class="note" x="24" y="{H - 14}">Source: docs/final/TCL-Final-Analyses.md, Section 3.</text>')
    (ROOT / "assets" / "phase3-4a-correctness-heads.svg").write_text(
        svg(W, H, "Learned correctness heads vs token-probability baselines",
            "Held-out test AUROC of the learned correctness head and three free token-probability baselines on nine "
            "model-dataset pairs. The head beats the best baseline only on Qwen 0.5B PopQA, by 0.012.", body),
        encoding="utf-8")


def trust_dimensions() -> None:
    rows = read("mde.csv")
    panels = [("H4 self-consistency", "Self-consistency (R + C-hat)"), ("H4 grounding", "Grounding (R + G-hat)")]
    data = {t: sorted([r for r in rows if r["test"] == t], key=lambda r: (ORDER.index(r["pair"].split("/")[0]), r["pair"]))
            for t, _ in panels}
    n = max(len(v) for v in data.values())
    W, TOP, ROW, PW = 980, 156, 30, 250
    H = TOP + ROW * n + 70
    body = [
        '<text class="h1" x="24" y="34">Phases 3 and 4a: extra trust dimensions add little to answer probability</text>',
        '<text class="sub" x="24" y="54">Gain in selective prediction (AURC of answer probability minus AURC of the combination), mean of 5 seeds,</text>',
        '<text class="sub" x="24" y="70">with its 95% paired bootstrap CI. A pair passed only if the CI is above zero (blue).</text>',
        '<circle class="c1" cx="30" cy="96" r="5"/><text class="leg" x="42" y="100">CI above zero</text>',
        '<circle class="c0" cx="160" cy="96" r="5"/><text class="leg" x="172" y="100">CI includes or is below zero</text>',
    ]
    for p, (test, title) in enumerate(panels):
        lx = 24 + p * 490
        x0 = lx + 180
        x = scale(-0.03, 0.035, x0, x0 + PW)
        rs = data[test]
        bottom = TOP + ROW * len(rs) - 12
        body.append(f'<text class="sub" x="{lx}" y="{TOP - 22}">{title}</text>')
        for t in (-0.02, 0.0, 0.02):
            cls = "axis" if t == 0 else "grid"
            body.append(f'<line class="{cls}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 14}" y2="{bottom}"/>')
            body.append(f'<text class="tick" x="{x(t):.1f}" y="{bottom + 16}" text-anchor="middle">{t:+.2f}</text>')
        for i, r in enumerate(rs):
            y = TOP + ROW * i
            v, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
            cls = "c1" if lo > 0 else "c0"
            body.append(f'<text class="row" x="{x0 - 12}" y="{y + 5}" text-anchor="end">{label(r["pair"])}</text>')
            body.append(f'<line class="link" x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{y}" y2="{y}"/>')
            body.append(f'<circle class="{cls}" cx="{x(v):.1f}" cy="{y}" r="6"><title>{label(r["pair"])}: {v:+.4f} [{lo:+.4f}, {hi:+.4f}]</title></circle>')
    body.append(f'<text class="note" x="24" y="{H - 30}">Each test needed a majority of its eligible pairs, including the new dataset (PopQA; RelationExtraction). Grounding at 7B</text>')
    body.append(f'<text class="note" x="24" y="{H - 14}">was not testable (RelationExtraction failed the size rule). Sources: docs/phase3/ and docs/phase4/ checkpoints.</text>')
    (ROOT / "assets" / "phase3-4a-trust-dimensions.svg").write_text(
        svg(W, H, "Self-consistency and grounding gains over answer probability",
            "AURC gain with 95% CIs for the self-consistency and grounding heads on every eligible pair at 0.5B, 3.8B "
            "and 7B. Self-consistency passes on 5 of 9 pairs, grounding on 2 of 9; neither passes its preregistered rule "
            "at any size.", body),
        encoding="utf-8")


if __name__ == "__main__":
    correctness_heads()
    trust_dimensions()
