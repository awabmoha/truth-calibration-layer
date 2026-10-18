"""Build assets/phase1-probe-vs-raw.svg from the committed Phase 1 Kaggle results.

Forest plot: frozen probe minus calibrated raw confidence, AUROC, with 95%
bootstrap CIs, for each retained pairing, with all answers and with refusals
excluded. Run from the repository root: python assets/make_phase1_figure.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tcl_experiments" / "runs" / "phase1_kaggle_analysis"
OUT = ROOT / "assets" / "phase1-probe-vs-raw.svg"

PAIRINGS = [
    ("qwen-triviaqa", "Qwen 0.5B, TriviaQA"),
    ("qwen-squad", "Qwen 0.5B, SQuAD"),
    ("phi-triviaqa", "Phi-3.5 mini, TriviaQA"),
    ("phi-nq_open", "Phi-3.5 mini, NQ-Open"),
    ("gemma-triviaqa", "Gemma 2B, TriviaQA"),
    ("gemma-squad", "Gemma 2B, SQuAD"),
]

W, H = 760, 410
LEFT, RIGHT, TOP, ROW = 200, 30, 122, 42
X_MIN, X_MAX = -0.12, 0.40


def x(v: float) -> float:
    return LEFT + (v - X_MIN) / (X_MAX - X_MIN) * (W - LEFT - RIGHT)


def main() -> None:
    h1 = json.loads((RESULTS / "h1r_details.json").read_text(encoding="utf-8"))
    dec = json.loads((RESULTS / "decisions.json").read_text(encoding="utf-8"))
    secondary = dec["secondary_refusals_excluded_auroc_diff"]

    parts = []
    # Grid and axis ticks.
    for t in [-0.1, 0.0, 0.1, 0.2, 0.3]:
        cls = "zero" if t == 0 else "grid"
        parts.append(f'<line class="{cls}" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 12}" y2="{TOP + ROW * len(PAIRINGS) - 10}"/>')
        parts.append(f'<text class="tick" x="{x(t):.1f}" y="{TOP + ROW * len(PAIRINGS) + 8}" text-anchor="middle">{t:+.1f}</text>'.replace("+0.0", "0"))

    for i, (key, label) in enumerate(PAIRINGS):
        cy = TOP + i * ROW + 6
        d = h1[key]["0"]["bootstrap"]["differences"]["probe-calibrated_raw"]["auroc"]
        s = secondary[key]
        significant = d["ci"][0] > 0
        parts.append(f'<text class="row" x="{LEFT - 14}" y="{cy + 5}" text-anchor="end">{label}</text>')
        # Primary: all answers (circle).
        y1 = cy - 6
        parts.append(f'<line class="ci s1" x1="{x(d["ci"][0]):.1f}" x2="{x(d["ci"][1]):.1f}" y1="{y1}" y2="{y1}"/>')
        parts.append(f'<circle class="m1" cx="{x(d["value"]):.1f}" cy="{y1}" r="5"><title>{label}, all answers: {d["value"]:+.3f} [{d["ci"][0]:+.3f}, {d["ci"][1]:+.3f}]</title></circle>')
        cls = "val sig" if significant else "val"
        if d["value"] < 0:
            parts.append(f'<text class="{cls}" x="{x(d["ci"][0]) - 8:.1f}" y="{y1 + 4}" text-anchor="end">{d["value"]:+.3f}</text>')
        else:
            parts.append(f'<text class="{cls}" x="{x(d["ci"][1]) + 8:.1f}" y="{y1 + 4}">{d["value"]:+.3f}</text>')
        # Secondary: refusals excluded (square).
        y2 = cy + 7
        parts.append(f'<line class="ci s2" x1="{x(s["ci"][0]):.1f}" x2="{x(s["ci"][1]):.1f}" y1="{y2}" y2="{y2}"/>')
        parts.append(f'<rect class="m2" x="{x(s["value"]) - 4.5:.1f}" y="{y2 - 4.5}" width="9" height="9" rx="1.5"><title>{label}, refusals excluded: {s["value"]:+.3f} [{s["ci"][0]:+.3f}, {s["ci"][1]:+.3f}]</title></rect>')

    n_wins = dec["H1-R"]["primary"]["n_wins"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
<title id="t">Phase 1: frozen hidden-state probe vs calibrated raw confidence</title>
<desc id="d">AUROC difference with 95% bootstrap confidence intervals for six model and dataset pairings. With all answers, the probe is significantly better in {n_wins} of 6 pairings and significantly worse on Qwen SQuAD. Excluding refusals, the large Gemma gains shrink to +0.122 and +0.047.</desc>
<style>
  .bg {{ fill: #fcfcfb; }}
  .h1 {{ font: 600 17px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0b0b0b; }}
  .sub, .tick, .leg, .note {{ font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #52514e; }}
  .row {{ font: 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0b0b0b; }}
  .val {{ font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0b0b0b; }}
  .sig {{ font-weight: 700; }}
  .grid {{ stroke: #e4e3df; stroke-width: 1; }}
  .zero {{ stroke: #8a8984; stroke-width: 1.5; }}
  .ci {{ stroke-width: 2; stroke-linecap: round; }}
  .s1 {{ stroke: #2a78d6; }} .m1 {{ fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; }}
  .s2 {{ stroke: #eb6834; }} .m2 {{ fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; }}
  @media (prefers-color-scheme: dark) {{
    .bg {{ fill: #1a1a19; }}
    .h1, .row, .val {{ fill: #f0efec; }}
    .sub, .tick, .leg, .note {{ fill: #c3c2b7; }}
    .grid {{ stroke: #33332f; }} .zero {{ stroke: #8a8984; }}
    .s1 {{ stroke: #3987e5; }} .m1 {{ fill: #3987e5; stroke: #1a1a19; }}
    .s2 {{ stroke: #d95926; }} .m2 {{ fill: #d95926; stroke: #1a1a19; }}
  }}
</style>
<rect class="bg" width="{W}" height="{H}" rx="8"/>
<text class="h1" x="24" y="34">Does the frozen probe rank answers better than calibrated raw confidence?</text>
<text class="sub" x="24" y="54">AUROC difference (probe minus calibrated raw) with 95% bootstrap CI, 1,500 test questions</text>
<text class="sub" x="24" y="70">per pairing. Right of 0 means the probe ranks correct vs incorrect answers better.</text>
<circle class="m1" cx="30" cy="94" r="5"/><text class="leg" x="42" y="98">All answers (primary test)</text>
<rect class="m2" x="225.5" y="89.5" width="9" height="9" rx="1.5"/><text class="leg" x="242" y="98">Refusals excluded (secondary)</text>
{chr(10).join(parts)}
<text class="note" x="24" y="{H - 16}">Bold value = CI excludes 0 in favor of the probe. Source: tcl_experiments/runs/phase1_kaggle_analysis/ (seed 0).</text>
</svg>
'''
    OUT.write_text(svg, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
