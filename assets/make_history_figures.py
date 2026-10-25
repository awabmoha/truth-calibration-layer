"""Build the README figures for TCL-v0 and the Phase 1 transfer and layer results.

Writes:
  assets/tcl-v0-high-confidence-errors.svg  (TCL-v0, June 2026: confidently wrong answers, raw vs conservative)
  assets/phase1-transfer.svg                (Phase 1: frozen probe in-distribution vs transferred, vs raw)
  assets/phase1-layer-sweep.svg             (Phase 1: Qwen test AUROC by layer depth and pooling)

Run from the repository root: python assets/make_history_figures.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "tcl_experiments" / "runs" / "phase1_kaggle_analysis"
ASSETS = ROOT / "assets"

# Same validated palette as make_phase1_figure.py: slot 1 blue, slot 2 orange, a neutral gray for baselines.
STYLE = """<style>
  .bg { fill: #fcfcfb; }
  .h1 { font: 600 17px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0b0b0b; }
  .sub, .tick, .leg, .note { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #52514e; }
  .row, .val { font: 13px system-ui, -apple-system, "Segoe UI", sans-serif; fill: #0b0b0b; }
  .val { font-size: 12px; }
  .grid { stroke: #e4e3df; stroke-width: 1; }
  .axis { stroke: #8a8984; stroke-width: 1.5; }
  .link { stroke: #c9c8c2; stroke-width: 2; stroke-linecap: round; }
  .c1 { fill: #2a78d6; stroke: #fcfcfb; stroke-width: 2; } .l1 { stroke: #2a78d6; stroke-width: 2; fill: none; }
  .c2 { fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; } .l2 { stroke: #eb6834; stroke-width: 2; fill: none; }
  .c0 { fill: #8a8984; stroke: #fcfcfb; stroke-width: 2; }
  .dash { stroke-dasharray: 6 4; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19; }
    .h1, .row, .val { fill: #f0efec; }
    .sub, .tick, .leg, .note { fill: #c3c2b7; }
    .grid { stroke: #33332f; } .link { stroke: #4a4a45; }
    .c1 { fill: #3987e5; stroke: #1a1a19; } .l1 { stroke: #3987e5; }
    .c2 { fill: #d95926; stroke: #1a1a19; } .l2 { stroke: #d95926; }
    .c0 { fill: #a3a29b; stroke: #1a1a19; }
  }
</style>"""


def svg(w: int, h: int, title: str, desc: str, body: list[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">\n'
            f'<title id="t">{title}</title>\n<desc id="d">{desc}</desc>\n{STYLE}\n'
            f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n' + "\n".join(body) + "\n</svg>\n")


def scale(lo: float, hi: float, a: float, b: float):
    return lambda v: a + (v - lo) / (hi - lo) * (b - a)


# ---------------------------------------------------------------- TCL-v0

# Source: docs/archive/tcl-v0/TCL-v0-evidence-report-v2.md (200 test answers per run, reviewed labels).
V0_RUNS = [
    ("Qwen 0.5B, SQuAD", 30, 10),
    ("Qwen 0.5B, TriviaQA", 9, 1),
    ("Phi-3.5 mini, SQuAD", 5, 4),
    ("Phi-3.5 mini, TriviaQA", 42, 9),
    ("Gemma 2B, SQuAD", 62, 10),
    ("Gemma 2B, TriviaQA", 76, 10),
]


def tcl_v0() -> None:
    W, H, LEFT, RIGHT, TOP, ROW = 760, 380, 200, 40, 118, 38
    x = scale(0, 80, LEFT, W - RIGHT)
    body = [
        '<text class="h1" x="24" y="34">TCL-v0 (June 2026): confidently wrong answers</text>',
        '<text class="sub" x="24" y="54">Wrong answers given with confidence of at least 0.8, out of 200 test questions per run.</text>',
        '<text class="sub" x="24" y="70">Conservative TCL-v0 = min(raw confidence, frozen final-layer probe).</text>',
        '<circle class="c0" cx="30" cy="92" r="5"/><text class="leg" x="42" y="96">Raw confidence</text>',
        '<circle class="c1" cx="160" cy="92" r="5"/><text class="leg" x="172" y="96">Conservative TCL-v0</text>',
    ]
    bottom = TOP + ROW * len(V0_RUNS) - 14
    for t in range(0, 81, 20):
        body.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 14}" y2="{bottom}"/>')
        body.append(f'<text class="tick" x="{x(t):.1f}" y="{bottom + 16}" text-anchor="middle">{t}</text>')
    for i, (label, raw, cons) in enumerate(V0_RUNS):
        cy = TOP + i * ROW
        body.append(f'<text class="row" x="{LEFT - 14}" y="{cy + 5}" text-anchor="end">{label}</text>')
        body.append(f'<line class="link" x1="{x(cons):.1f}" x2="{x(raw):.1f}" y1="{cy}" y2="{cy}"/>')
        body.append(f'<circle class="c0" cx="{x(raw):.1f}" cy="{cy}" r="6"><title>{label}, raw: {raw}</title></circle>')
        body.append(f'<circle class="c1" cx="{x(cons):.1f}" cy="{cy}" r="6"><title>{label}, conservative TCL-v0: {cons}</title></circle>')
        body.append(f'<text class="val" x="{x(raw) + 11:.1f}" y="{cy + 4}">{raw}</text>')
        if x(cons) - LEFT > 24:
            body.append(f'<text class="val" x="{x(cons) - 11:.1f}" y="{cy + 4}" text-anchor="end">{cons}</text>')
        else:  # too close to the row label: put the value above the dot
            body.append(f'<text class="val" x="{x(cons):.1f}" y="{cy - 10}" text-anchor="middle">{cons}</text>')
    body.append(f'<text class="note" x="24" y="{H - 30}">Phase 1 later found that recalibrating raw confidence alone also lowers this count, and that these</text>')
    body.append(f'<text class="note" x="24" y="{H - 14}">test sets were too small and the SQuAD split shared passages. Source: docs/archive/tcl-v0/TCL-v0-evidence-report-v2.md.</text>')
    (ASSETS / "tcl-v0-high-confidence-errors.svg").write_text(svg(
        W, H, "TCL-v0: confidently wrong answers, raw confidence vs conservative TCL-v0",
        "For six model and dataset runs of TCL-v0, the number of wrong answers with confidence of at least 0.8 fell from "
        + ", ".join(f"{r} to {c}" for _, r, c in V0_RUNS) + " under the conservative rule.", body), encoding="utf-8")


# ---------------------------------------------------------------- Phase 1 transfer

NAMES = {"qwen": "Qwen 0.5B", "phi": "Phi-3.5 mini", "gemma": "Gemma 2B"}
DS = {"triviaqa": "TriviaQA", "squad": "SQuAD", "nq_open": "NQ-Open"}


def transfer() -> None:
    with (RESULTS / "h2r_transfer.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["seed"] == "0"]
    W, LEFT, RIGHT, TOP, ROW = 760, 250, 40, 122, 40
    H = TOP + ROW * len(rows) + 56
    x = scale(0.5, 0.9, LEFT, W - RIGHT)
    body = [
        '<text class="h1" x="24" y="34">Phase 1: frozen probes do not transfer across datasets</text>',
        '<text class="sub" x="24" y="54">Test AUROC on the target dataset, 1,500 test questions. A probe trained on the target dataset</text>',
        '<text class="sub" x="24" y="70">(blue) vs the same probe trained on the other dataset and transferred (orange).</text>',
        '<circle class="c0" cx="30" cy="94" r="5"/><text class="leg" x="42" y="98">Raw confidence</text>',
        '<circle class="c1" cx="160" cy="94" r="5"/><text class="leg" x="172" y="98">Probe, trained on target</text>',
        '<circle class="c2" cx="340" cy="94" r="5"/><text class="leg" x="352" y="98">Probe, transferred</text>',
    ]
    bottom = TOP + ROW * len(rows) - 16
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        body.append(f'<line class="grid" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{TOP - 14}" y2="{bottom}"/>')
        body.append(f'<text class="tick" x="{x(t):.1f}" y="{bottom + 16}" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(rows):
        model, pair = r["direction"].split(":")
        src, tgt = pair.split("->")
        label = f"{NAMES[model]}: {DS[src]} → {DS[tgt]}"
        raw, ind, tr = float(r["raw_auroc"]), float(r["in_distribution_probe_auroc"]), float(r["transfer_probe_auroc"])
        cy = TOP + i * ROW
        body.append(f'<text class="row" x="{LEFT - 14}" y="{cy + 5}" text-anchor="end">{label}</text>')
        body.append(f'<line class="link" x1="{x(min(ind, tr)):.1f}" x2="{x(max(ind, tr)):.1f}" y1="{cy}" y2="{cy}"/>')
        body.append(f'<circle class="c0" cx="{x(raw):.1f}" cy="{cy}" r="5"><title>{label}, raw confidence: {raw:.3f}</title></circle>')
        body.append(f'<circle class="c1" cx="{x(ind):.1f}" cy="{cy}" r="6"><title>{label}, probe trained on target: {ind:.3f}</title></circle>')
        body.append(f'<circle class="c2" cx="{x(tr):.1f}" cy="{cy}" r="6"><title>{label}, probe transferred: {tr:.3f}</title></circle>')
    body.append(f'<text class="note" x="24" y="{H - 30}">Qwen and Phi probes transfer to the level of raw confidence or below. Gemma\'s transfer holds partly because</text>')
    body.append(f'<text class="note" x="24" y="{H - 14}">its probe detects refusals. Source: tcl_experiments/runs/phase1_kaggle_analysis/h2r_transfer.csv.</text>')
    (ASSETS / "phase1-transfer.svg").write_text(svg(
        W, H, "Phase 1: frozen probe AUROC in-distribution vs transferred",
        "For six transfer directions, test AUROC of raw confidence, a frozen probe trained on the target dataset, and a frozen probe "
        "trained on the other dataset. Qwen and Phi probes lose most of their advantage when transferred.", body), encoding="utf-8")


# ---------------------------------------------------------------- Phase 1 layer sweep

def layer_sweep() -> None:
    with (RESULTS / "layer_sweep_qwen.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    auc = {(r["dataset"], r["feature"]): float(r["test_auroc"]) for r in rows}
    depths = [("q25", "25% depth"), ("q50", "50% depth"), ("final", "Final layer")]
    W, H, LEFT, RIGHT, TOP, BOTTOM = 760, 440, 70, 170, 122, 364
    xs = {d: LEFT + i * (W - LEFT - RIGHT) / (len(depths) - 1) for i, (d, _) in enumerate(depths)}
    y = scale(0.6, 0.85, BOTTOM, TOP)
    body = [
        '<text class="h1" x="24" y="34">Phase 1: middle layers carry the most signal (Qwen 0.5B)</text>',
        '<text class="sub" x="24" y="54">Test AUROC of a frozen logistic probe by layer depth, pooling the answer tokens by mean (solid)</text>',
        '<text class="sub" x="24" y="70">or taking the last answer token (dashed). 1,500 test questions per dataset.</text>',
        '<line class="l1" x1="24" x2="44" y1="92" y2="92"/><text class="leg" x="52" y="96">TriviaQA</text>',
        '<line class="l2" x1="130" x2="150" y1="92" y2="92"/><text class="leg" x="158" y="96">SQuAD</text>',
        '<line class="axis" x1="230" x2="250" y1="92" y2="92"/><text class="leg" x="258" y="96">Mean pooling</text>',
        '<line class="axis dash" x1="360" x2="380" y1="92" y2="92"/><text class="leg" x="388" y="96">Last answer token</text>',
    ]
    for t in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        body.append(f'<line class="grid" x1="{LEFT}" x2="{W - RIGHT}" y1="{y(t):.1f}" y2="{y(t):.1f}"/>')
        body.append(f'<text class="tick" x="{LEFT - 10}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for d, name in depths:
        body.append(f'<text class="tick" x="{xs[d]:.1f}" y="{BOTTOM + 22}" text-anchor="middle">{name}</text>')
    series = [("triviaqa", "TriviaQA", "1"), ("squad", "SQuAD", "2")]
    ends = []
    for ds, ds_name, c in series:
        for pool, dash in (("mean", ""), ("last", " dash")):
            pts = [(xs[d], y(auc[(ds, f"{d}_answer_{pool}")])) for d, _ in depths]
            body.append(f'<polyline class="l{c}{dash}" points="{" ".join(f"{px:.1f},{py:.1f}" for px, py in pts)}"/>')
            for (d, name), (px, py) in zip(depths, pts):
                v = auc[(ds, f"{d}_answer_{pool}")]
                body.append(f'<circle class="c{c}" cx="{px:.1f}" cy="{py:.1f}" r="5"><title>{ds_name}, {pool} pooling, {name}: {v:.3f}</title></circle>')
            ends.append([pts[-1][1], f"{ds_name}, {pool}", auc[(ds, f'final_answer_{pool}')]])
    # Direct labels at the right end, nudged apart so they do not overlap.
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
    for ly, text, v in ends:
        body.append(f'<text class="val" x="{W - RIGHT + 12}" y="{ly + 4:.1f}">{text} {v:.3f}</text>')
    peak = {ds: auc[(ds, "q50_answer_mean")] for ds, _, _ in series}
    body.append(f'<text class="note" x="24" y="{H - 30}">50% depth is the best layer for both pooling methods on both datasets, so Phase 2\'s trust head reads from there.</text>')
    body.append(f'<text class="note" x="24" y="{H - 14}">Source: tcl_experiments/runs/phase1_kaggle_analysis/layer_sweep_qwen.csv.</text>')
    (ASSETS / "phase1-layer-sweep.svg").write_text(svg(
        W, H, "Phase 1: Qwen probe AUROC by layer depth and pooling",
        "Frozen-probe test AUROC is highest at 50% depth for both pooling methods on both datasets; with mean pooling it is "
        f"{peak['triviaqa']:.3f} on TriviaQA and {peak['squad']:.3f} on SQuAD.", body), encoding="utf-8")


if __name__ == "__main__":
    tcl_v0()
    transfer()
    layer_sweep()
    print("Wrote", ", ".join(p.name for p in sorted(ASSETS.glob("*.svg"))))
