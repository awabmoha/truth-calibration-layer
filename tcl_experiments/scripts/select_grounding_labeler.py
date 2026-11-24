"""Choose the Phase 4a grounding labeler on the Phase 3 audit set (Phase 4a protocol, Section 4).

Development set: the 100 Phase 3 grounding audit answers with my labels
(runs/phase3_h4g_audit/grounding_audit_reviewed_{qwen,phi}.csv). Premise: the passage the
model saw; threshold 0.5. Candidates (label_grounding.LABELERS):
  1. phase3    deberta-large-mnli, hypothesis "question answer"
  2. template  deberta-large-mnli, hypothesis 'The answer to the question "q" is a.'
  3. wanli     DeBERTa-v3-large-mnli-fever-anli-ling-wanli, the template hypothesis
The candidate that agrees with my labels on the most answers is chosen; ties go to the
lower-numbered candidate. No 7B answer is used. The agreement is optimistic (the set was
used to choose); Phase 4a reports the error rate from a fresh audit of 7B answers.

Usage: python scripts/select_grounding_labeler.py
Output: runs/phase4a/grounding_labeler_selection.json
"""

from __future__ import annotations

import csv
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label_grounding import LABELERS, THRESHOLD, entail_probs, hypothesis, load_nli  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runs" / "phase3_h4g_audit"
ORDER = ["phase3", "template", "wanli"]


def main() -> None:
    rows = []
    for m in ("qwen", "phi"):
        with (AUDIT / f"grounding_audit_reviewed_{m}.csv").open(encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    truth = [int(r["manual_grounded"]) for r in rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name in ORDER:
        model, tok = load_nli(name, device)
        probs = entail_probs(model, tok, device, [r["passage"] for r in rows],
                             [hypothesis(r["question"], r["answer"], name) for r in rows], batch=8)
        pred = [int(p > THRESHOLD) for p in probs]
        by = {}
        for half, k in (("gold", "0"), ("distractor", "1")):
            idx = [i for i, r in enumerate(rows) if r["distractor"] == k]
            by[half] = {"agree": sum(pred[i] == truth[i] for i in idx), "n": len(idx),
                        "missed_support": sum(pred[i] == 0 and truth[i] == 1 for i in idx),
                        "false_support": sum(pred[i] == 1 and truth[i] == 0 for i in idx)}
        agree = sum(p == t for p, t in zip(pred, truth))
        results[name] = {**LABELERS[name], "agree": agree, "n": len(rows), "by_passage": by,
                         "probs": [round(p, 4) for p in probs]}
        print(f"{name:9s} agrees on {agree}/{len(rows)}  gold {by['gold']}  distractor {by['distractor']}", flush=True)
        del model, tok
        gc.collect()
    chosen = max(ORDER, key=lambda n: (results[n]["agree"], -ORDER.index(n)))
    out = ROOT / "runs" / "phase4a" / "grounding_labeler_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
                               "protocol": "docs/phase4/TCL-Phase4a-Protocol.md", "threshold": THRESHOLD,
                               "chosen": chosen, "candidates": results}, indent=2), encoding="utf-8")
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
