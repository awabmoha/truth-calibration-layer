from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from labeling import CORRECTNESS_METHOD, extract_primary_answer, is_correct


ROOT = Path(__file__).resolve().parents[1]
PROMPT_TEMPLATE_NAME = "chat_short_factual_answer_v1"
PROMPT_TEMPLATE = (
    "Answer with only the short factual answer. Do not explain.\n"
    "Question: {question}\n"
    "Answer:"
)
RAW_CONFIDENCE_METHOD = "geometric_mean_generated_token_probability"


def read_splits(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    split_path = Path(path)
    with split_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    splits = {}
    for row in rows:
        item_id = row.get("id") or row.get("question_id")
        split = row.get("split")
        if item_id and split:
            splits[item_id] = split
    return splits


def row_id(row: Mapping[str, str]) -> str:
    return row.get("id") or row.get("question_id") or ""


def read_questions(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    return rows


def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You answer factual questions with only the shortest correct answer. No explanation.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nAnswer with only the answer phrase.",
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return PROMPT_TEMPLATE.format(question=question)


def generation_confidence(scores, sequences, prompt_len: int) -> float:
    if not scores:
        return 0.0
    generated_ids = sequences[0, prompt_len:]
    probs = []
    for step_scores, token_id in zip(scores, generated_ids):
        prob = torch.softmax(step_scores[0], dim=-1)[token_id].item()
        probs.append(max(prob, 1e-12))
    if not probs:
        return 0.0
    return float(math.exp(sum(math.log(p) for p in probs) / len(probs)))


def extract_hidden_state(hidden_states, prompt_len: int, method: str, layer: int) -> np.ndarray:
    selected = hidden_states[layer][0]
    answer_hidden = selected[prompt_len:]

    if method == "answer_last":
        if answer_hidden.shape[0] == 0:
            vector = selected[-1]
        else:
            vector = answer_hidden[-1]
    elif method == "answer_mean":
        if answer_hidden.shape[0] == 0:
            vector = selected.mean(dim=0)
        else:
            vector = answer_hidden.mean(dim=0)
    elif method == "prompt_answer_mean":
        vector = selected.mean(dim=0)
    else:
        raise ValueError(f"Unknown hidden-state method: {method}")

    return vector.detach().float().cpu().numpy().astype(float)


def write_run_config(out_path: Path, args, run_id: str, started_at: str) -> None:
    config = {
        "run_id": run_id,
        "started_at": started_at,
        "model_name": args.model,
        "dataset": args.dataset,
        "questions": str(Path(args.questions)),
        "out": str(out_path),
        "limit": args.limit,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template_name": PROMPT_TEMPLATE_NAME,
        "prompt_template": PROMPT_TEMPLATE,
        "raw_confidence_method": RAW_CONFIDENCE_METHOD,
        "correctness_method": CORRECTNESS_METHOD,
        "hidden_state_layer": args.hidden_state_layer,
        "hidden_state_method": args.hidden_state_method,
        "note": "Smoke-test outputs are pipeline checks only unless the run is explicitly designated as an evidence run.",
    }
    config_path = out_path.with_suffix(".config.json")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset", default="seed_questions")
    parser.add_argument("--questions", default=str(ROOT / "data" / "seed_questions.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--splits", default=None, help="Optional CSV with id,split columns.")
    parser.add_argument(
        "--hidden-state-method",
        choices=["answer_mean", "answer_last", "prompt_answer_mean"],
        default="answer_mean",
    )
    parser.add_argument(
        "--hidden-state-layer",
        type=int,
        default=-1,
        help="Transformer layer index to extract. Default -1 means the final layer.",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"smoke-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc).isoformat()
    if args.out is None:
        out_path = ROOT / "runs" / run_id / f"records_{args.hidden_state_method}.jsonl"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_config(out_path, args, run_id, started_at)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    questions = read_questions(Path(args.questions), args.limit)
    splits = read_splits(args.splits)
    with out_path.open("w", encoding="utf-8") as f:
        for row in tqdm(questions, desc="questions"):
            prompt = build_prompt(tokenizer, row["question"])
            inputs = tokenizer(prompt, return_tensors="pt")
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                full_ids = generated.sequences
                answer_ids = full_ids[0, prompt_len:]
                raw_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                answer = extract_primary_answer(raw_answer)
                conf = generation_confidence(generated.scores, full_ids, prompt_len) if answer else 0.0

                hidden_out = model(
                    input_ids=full_ids,
                    attention_mask=torch.ones_like(full_ids),
                    output_hidden_states=True,
                    use_cache=False,
                )
                hidden_vector = extract_hidden_state(
                    hidden_out.hidden_states,
                    prompt_len=prompt_len,
                    method=args.hidden_state_method,
                    layer=args.hidden_state_layer,
                )

            answer_text = row.get("answers") or row.get("accepted_answers") or ""
            accepted = [answer.strip() for answer in answer_text.split("|") if answer.strip()]
            item_id = row_id(row)
            record = {
                "id": item_id,
                "dataset": args.dataset,
                "question": row["question"],
                "accepted_answers": accepted,
                "model_name": args.model,
                "prompt_template": PROMPT_TEMPLATE_NAME,
                "prompt": prompt,
                "raw_model_output": raw_answer,
                "model_answer": answer,
                "correctness_label": int(is_correct(answer, accepted, row["question"])),
                "correctness_method": CORRECTNESS_METHOD,
                "raw_generation_confidence": conf,
                "raw_confidence_method": RAW_CONFIDENCE_METHOD,
                "hidden_state_layer": args.hidden_state_layer,
                "hidden_state_method": args.hidden_state_method,
                "hidden_state_vector": hidden_vector.tolist(),
                "tcl_v0_confidence": None,
                "split": splits.get(item_id, "unassigned"),
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
