"""Generate the paper's figures as pgfplots code from the recorded results.

Reads tcl_experiments/runs/final_analyses/{baseline_choice.csv, mde.csv} and writes
figures/fig_baselines.tex and figures/fig_dimensions.tex, which tcl-final.tex inputs.

Run from the repository root: python paper/tcl-final/make_figures.py
"""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
FA = HERE.parents[1] / "tcl_experiments" / "runs" / "final_analyses"
OUT = HERE / "figures"
MODEL = {"qwen": "Qwen 0.5B", "phi": "Phi 3.8B", "qwen7b": "Qwen 7B"}
DATA = {"triviaqa": "TriviaQA", "nq_open": "NQ-Open", "popqa": "PopQA", "squad": "SQuAD", "nq": "NQ",
        "hotpotqa": "HotpotQA", "relation_extraction": "RelationExt."}
ORDER = ["qwen", "phi", "qwen7b"]

COLORS = r"""\definecolor{tclblue}{HTML}{2A78D6}
\definecolor{tclorange}{HTML}{EB6834}
\definecolor{tclgray}{HTML}{8A8984}
"""


def read(name):
    with (FA / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


DORDER = ["triviaqa", "squad", "nq_open", "popqa", "nq", "hotpotqa", "relation_extraction"]


def key(r):
    m, d = r["pair"].split("/")
    return (ORDER.index(m), DORDER.index(d))


def label(pair):
    m, d = pair.split("/")
    return f"{MODEL[m]} / {DATA[d]}"


def axis_opts(n, xmin, xmax, xticks, labels, width, extra=""):
    ticklabels = ",".join("{" + l + "}" for l in labels)
    return (f"width={width}, height={0.52 * n + 1.0:.2f}cm, xmin={xmin}, xmax={xmax}, ymin=0.4, ymax={n + 0.6},"
            f" ytick={{1,...,{n}}}, yticklabels={{{ticklabels}}}, y dir=reverse, xtick={{{xticks}}},"
            " xmajorgrids, major grid style={gray!20}, tick label style={font=\\footnotesize},"
            " label style={font=\\footnotesize}, axis line style={gray!60}, tick style={gray!60}" + extra)


def fig_baselines():
    rows = sorted(read("baseline_choice.csv"), key=key)
    n = len(rows)
    series = {"auroc_head_MR": "only marks, mark=*, mark size=2.4pt, fill=tclblue, draw=white",
              "auroc_span": "only marks, mark=*, mark size=2.4pt, fill=tclgray, draw=white",
              "auroc_min_token": "only marks, mark=*, mark size=2.4pt, fill=tclorange, draw=white",
              "auroc_first_token": "only marks, mark=o, mark size=2.2pt, draw=tclgray, thick"}
    names = {"auroc_first_token": "First-token probability", "auroc_span": "Answer-span probability (preregistered)",
             "auroc_min_token": "Minimum-token probability", "auroc_head_MR": "Learned head (M+R)"}
    lines = [COLORS, r"\begin{tikzpicture}", r"\begin{axis}[" + axis_opts(
        n, 0.6, 0.95, "0.6,0.7,0.8,0.9", [label(r["pair"]) for r in rows], "0.72\\textwidth",
        ", xlabel={Held-out test AUROC (higher is better)}, legend style={font=\\footnotesize, at={(0.5,1.02)},"
        " anchor=south, legend columns=2, legend cell align=left, draw=none, /tikz/every even column/.append style={column sep=6pt}}") + "]"]
    for i, r in enumerate(rows, 1):
        vals = [float(r[k]) for k in series]
        lines.append(f"\\draw[gray!35, line width=1.2pt] (axis cs:{min(vals):.4f},{i}) -- (axis cs:{max(vals):.4f},{i});")
    for k, style in series.items():
        coords = " ".join(f"({float(r[k]):.4f},{i})" for i, r in enumerate(rows, 1))
        lines.append(f"\\addplot[{style}] coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{names[k]}}}")
    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    (OUT / "fig_baselines.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def panel(rows, title, xshift):
    n = len(rows)
    lines = [r"\begin{axis}[" + axis_opts(
        n, -0.03, 0.035, "-0.02,0,0.02", [label(r["pair"]) for r in rows], "0.34\\textwidth",
        f", title={{{title}}}, title style={{font=\\small}}, xlabel={{AURC gain (higher is better)}},"
        " xticklabels={{$-$0.02},{0},{+0.02}}, scaled x ticks=false,"
        f" at={{({xshift},0)}}") + "]",
        f"\\draw[gray!70, thick] (axis cs:0,0.4) -- (axis cs:0,{n + 0.6});"]
    for i, r in enumerate(rows, 1):
        v, lo, hi = float(r["value"]), float(r["ci_lo"]), float(r["ci_hi"])
        color = "tclblue" if lo > 0 else "tclgray"
        lines.append(f"\\draw[{color}, line width=1pt] (axis cs:{lo:.4f},{i}) -- (axis cs:{hi:.4f},{i});")
        lines.append(f"\\addplot[only marks, mark=*, mark size=2.3pt, fill={color}, draw=white] coordinates {{({v:.4f},{i})}};")
    lines.append(r"\end{axis}")
    return lines


def fig_dimensions():
    rows = read("mde.csv")
    cons = sorted([r for r in rows if r["test"] == "H4 self-consistency"], key=key)
    grnd = sorted([r for r in rows if r["test"] == "H4 grounding"], key=key)
    lines = [COLORS, r"\begin{tikzpicture}"]
    lines += panel(cons, "Self-consistency (R + C-hat)", "0cm")
    lines += panel(grnd, "Grounding (R + G-hat)", "0.53\\textwidth")
    lines += [r"\end{tikzpicture}"]
    (OUT / "fig_dimensions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_baselines()
    fig_dimensions()
    print("wrote", sorted(p.name for p in OUT.iterdir()))
