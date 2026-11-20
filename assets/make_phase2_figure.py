"""Build the README figure for Phase 2 and 2b transfer results.

Writes assets/phase2-transfer.svg: transfer AUROC of each arm (Phase 2 and the 2b rerun)
against raw token confidence and semantic entropy, for both transfer directions.

Run from the repository root: python assets/make_phase2_figure.py
"""

import json
from pathlib import Path

from make_history_figures import scale, svg  # same palette and helpers

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "tcl_experiments" / "runs"
P2 = json.loads((RUNS / "phase2_analysis" / "details.json").read_text(encoding="utf-8"))
P2B = json.loads((RUNS / "phase2b_analysis" / "details.json").read_text(encoding="utf-8"))

ARMS = [("A", "A: frozen head"), ("B", "B: trust-only LoRA"), ("C", "C: joint LoRA (TCL-J)"), ("D", "D: LM-only LoRA")]
PANELS = [("triviaqa", "Trained on TriviaQA, tested on SQuAD"), ("squad", "Trained on SQuAD, tested on TriviaQA")]


def main() -> None:
    W, H, LEFT, PANEL_W, GAP, TOP, ROW = 940, 340, 190, 320, 80, 142, 34
    body = [
        '<text class="h1" x="24" y="34">Phase 2 and 2b: no trained trust head transfers</text>',
        '<text class="sub" x="24" y="54">Test AUROC on the other dataset\'s 1,500 fixed answers, mean of 3 seeds. Higher is better.</text>',
        '<text class="sub" x="24" y="70">Untrained signals (dashed) beat every arm, before and after the end-of-turn fix.</text>',
        '<circle class="c1" cx="30" cy="94" r="5"/><text class="leg" x="42" y="98">Phase 2</text>',
        '<circle class="c2" cx="120" cy="94" r="5"/><text class="leg" x="132" y="98">Phase 2b rerun (C and D only)</text>',
        '<line x1="340" x2="362" y1="94" y2="94" style="stroke:#8a8984;stroke-width:2;stroke-dasharray:6 4"/><text class="leg" x="368" y="98">Raw token confidence</text>',
        '<line class="dash" x1="520" x2="542" y1="94" y2="94" style="stroke:#8a8984;stroke-width:2;stroke-dasharray:2 3"/><text class="leg" x="548" y="98">Semantic entropy</text>',
    ]
    bottom = TOP + ROW * len(ARMS) - 10
    for i, (arm, label) in enumerate(ARMS):
        y = TOP + ROW * i
        body.append(f'<text class="row" x="24" y="{y + 4}">{label}</text>')
    for p, (ds, title) in enumerate(PANELS):
        x0 = LEFT + p * (PANEL_W + GAP)
        x = scale(0.5, 0.8, x0, x0 + PANEL_W)
        body.append(f'<text class="sub" x="{x0}" y="{TOP - 20}">{title}</text>')
        for t in (0.5, 0.6, 0.7, 0.8):
            body.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 12}" y2="{bottom}"/>')
            body.append(f'<text class="tick" x="{x(t):.1f}" y="{bottom + 16}" text-anchor="middle">{t:.1f}</text>')
        ref = P2[ds]["reference_baselines_transfer"]
        for key, dash in (("raw_conf_span", "6 4"), ("semantic_entropy", "2 3")):
            v = ref[key]
            body.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{TOP - 12}" y2="{bottom}" style="stroke:#8a8984;stroke-width:2;stroke-dasharray:{dash}"/>')
        for i, (arm, _) in enumerate(ARMS):
            y = TOP + ROW * i
            a = P2[ds]["transfer"]["auroc"][arm]
            if arm in ("C", "D"):
                b = P2B[ds]["transfer"]["auroc"][arm]
                body.append(f'<line class="link" x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y}" y2="{y}"/>')
                body.append(f'<circle class="c2" cx="{x(b):.1f}" cy="{y}" r="6"/>')
            body.append(f'<circle class="c1" cx="{x(a):.1f}" cy="{y}" r="6"/>')
    (ROOT / "assets" / "phase2-transfer.svg").write_text(
        svg(W, H, "Phase 2 and 2b transfer AUROC",
            "Transfer AUROC of the four arms in both directions, Phase 2 and the 2b rerun, against raw token "
            "confidence and semantic entropy. Every arm stays between 0.56 and 0.65, below raw confidence "
            "(0.71-0.76) and semantic entropy (0.77).", body),
        encoding="utf-8")


if __name__ == "__main__":
    main()
