"""Grounding labels for Phase 3 H4 (grounding protocol, Section 3).

For every record of one run, the NLI model judges whether the passage the model saw
(premise) entails the answer as a response to the question (hypothesis: the question
followed by the answer). y_g = 1 if the entailment probability is above 0.5.
Abstentions get no label: the answer contains "unanswerable", matches the shared
refusal rule, or is empty.

Usage: python scripts/label_grounding.py --run-dir runs/phase3_grounding/<model>/<dataset>
Writes grounding.jsonl next to records.jsonl: id, abstain, entail_prob, grounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labeling import is_refusal, normalize

NLI_MODEL = "microsoft/deberta-large-mnli"
THRESHOLD = 0.5
METHOD = "nli_entailment_question_answer_v1"


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip()
    return not a or "unanswerable" in normalize(a) or is_refusal(a)


def hypothesis(question: str, answer: str) -> str:
    return f"{question.strip()} {answer.strip()}"


@torch.no_grad()
def entail_probs(model, tok, device, premises: list[str], hyps: list[str], batch: int) -> list[float]:
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    out = []
    for i in range(0, len(premises), batch):
        enc = tok(premises[i:i + batch], hyps[i:i + batch], return_tensors="pt", padding=True,
                  truncation="only_first", max_length=512).to(device)
        probs = torch.softmax(model(**enc).logits.float(), -1)[:, labels["entailment"]]
        out.extend(probs.cpu().tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    recs = [json.loads(line) for line in (args.run_dir / "records.jsonl").open(encoding="utf-8") if line.strip()]
    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    # float32, as in run_inference_v2.py: DeBERTa's attention fails in float16 on current transformers.
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, torch_dtype=torch.float32).to(device).eval()
    claims = [r for r in recs if not is_abstention(r["answer"])]
    probs = entail_probs(model, tok, device, [r["context"] for r in claims],
                         [hypothesis(r["question"], r["answer"]) for r in claims], args.batch)
    by_id = {r["id"]: p for r, p in zip(claims, probs)}
    with (args.run_dir / "grounding.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            p = by_id.get(r["id"])
            f.write(json.dumps({"id": r["id"], "abstain": p is None, "entail_prob": p,
                                "grounded": None if p is None else int(p > THRESHOLD),
                                "method": METHOD, "nli_model": NLI_MODEL}) + "\n")
    n_g = sum(int(p > THRESHOLD) for p in probs)
    print(f"{args.run_dir}: {len(recs)} records, {len(recs) - len(claims)} abstentions, "
          f"{n_g} grounded, {len(claims) - n_g} not grounded", flush=True)


if __name__ == "__main__":
    main()
