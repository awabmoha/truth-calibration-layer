"""Inference v2 for the Phase 1 Kaggle runs.

Implements Sections 3-4 of TCL-Phase1-Kaggle-Protocol.md:
- generation stops at EOS or end-of-turn; the answer span is the generated text
  before the first newline, and labels and raw confidence use that span only
- hidden states at 25%, 50%, and 100% depth, pooled over the answer span
  (answer_mean, answer_last), stored as float16 chunks
- zero-shot P(True) self-evaluation
- semantic entropy (10 samples, bidirectional NLI clustering) on chosen splits

Output is written in chunks so that an interrupted Kaggle session can resume
with --resume without redoing finished questions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from labeling import is_correct, normalize
from run_inference import runtime_info, select_device, select_dtype, user_message

CORRECTNESS_METHOD = "strict_answer_segment_match_v2_answer_span"
RAW_SPAN_METHOD = "geometric_mean_answer_span_token_probability"
RAW_FULL_METHOD = "geometric_mean_generated_token_probability"
SYSTEM_PROMPT = "You answer factual questions with only the shortest correct answer. No explanation."
STOP_TOKEN_STRINGS = ["<end_of_turn>", "<|im_end|>", "<|end|>", "<|eot_id|>", "<|endoftext|>"]
LAYER_FRACTIONS = {"q25": 0.25, "q50": 0.5, "final": 1.0}
POOLINGS = ["answer_mean", "answer_last"]

REFUSAL_PATTERN = re.compile(
    r"\b(does not (?:provide|mention|specify|contain|say|state)|do(?:es)? not (?:know|have)|don't know|"
    r"cannot (?:provide|answer|determine|find)|can't (?:provide|answer|determine)|unable to|i'm sorry|i am sorry|"
    r"not (?:mentioned|specified|provided|stated|given) in|no (?:information|answer) (?:is |was )?(?:provided|given|available))\b",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERN.search(text))


# ---------------------------------------------------------------- prompts


def chat_prompt(tokenizer, user_content: str, mode: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return f"{user_content}\nAnswer:"
    with_system = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
    user_only = [{"role": "user", "content": user_content}]
    if mode == "user_only":
        return tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(with_system, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        if mode == "system_user" or "System role not supported" not in str(exc):
            raise
        return tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)


def p_true_content(question: str, context: str, answer: str) -> str:
    parts = []
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Question: {question}")
    parts.append(f"Proposed answer: {answer if answer else '(no answer)'}")
    parts.append("Is the proposed answer correct? Reply with only True or False.")
    return "\n".join(parts)


# ---------------------------------------------------------------- token helpers


def stop_token_ids(tokenizer) -> list[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    vocab = tokenizer.get_vocab()
    for tok in STOP_TOKEN_STRINGS:
        if tok in vocab:
            ids.add(int(vocab[tok]))
    return sorted(ids)


def answer_span(tokenizer, gen_ids: list[int], stop_ids: set[int]) -> tuple[int, int]:
    """Return [start, end) token indices of the answer span within generated ids."""
    pieces = [tokenizer.decode([t], skip_special_tokens=False) for t in gen_ids]
    start = 0
    while start < len(gen_ids) and (gen_ids[start] in stop_ids or not pieces[start].strip()):
        start += 1
    end = start
    while end < len(gen_ids) and gen_ids[end] not in stop_ids and "\n" not in pieces[end]:
        end += 1
    return start, end


def token_probs(step_logits: list[torch.Tensor], gen_ids: list[int], row: int = 0) -> np.ndarray:
    probs = []
    for logits, tok in zip(step_logits, gen_ids):
        p = torch.softmax(logits[row].float(), dim=-1)[tok].item()
        probs.append(max(p, 1e-12))
    return np.asarray(probs, dtype=float)


def geo_mean(p: np.ndarray) -> float:
    return float(math.exp(np.mean(np.log(p)))) if len(p) else 0.0


def candidate_ids(tokenizer, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        enc = tokenizer.encode(w, add_special_tokens=False)
        if enc:
            ids.add(int(enc[0]))
    return sorted(ids)


# ---------------------------------------------------------------- semantic entropy


class NLIJudge:
    def __init__(self, name: str, device: torch.device, dtype):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.device = device
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    @torch.no_grad()
    def entails(self, premise: str, hypothesis: str) -> bool:
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        return int(self.model(**enc).logits.argmax(-1).item()) == self.entail

    def equivalent(self, question: str, a: str, b: str) -> bool:
        na, nb = normalize(a), normalize(b)
        if na == nb:
            return True
        if not na or not nb:
            return False
        pa, pb = f"{question} {a}", f"{question} {b}"
        return self.entails(pa, pb) and self.entails(pb, pa)


def cluster_answers(judge: NLIJudge, question: str, answers: list[str]) -> list[int]:
    reps: list[str] = []
    ids = []
    for a in answers:
        for k, rep in enumerate(reps):
            if judge.equivalent(question, a, rep):
                ids.append(k)
                break
        else:
            reps.append(a)
            ids.append(len(reps) - 1)
    return ids


# ---------------------------------------------------------------- main loop


def read_questions(path: Path, limit: int | None, splits: set[str] | None = None) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if splits:
        rows = [r for r in rows if r["split"] in splits]
    return rows[:limit] if limit else rows


def done_ids(records_path: Path) -> set[str]:
    if not records_path.exists():
        return set()
    with records_path.open(encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def flush(out_dir: Path, records: list[dict], hidden: dict[str, list[np.ndarray]], chunk_index: int) -> None:
    if not records:
        return
    arrays = {k: np.stack(v).astype(np.float16) for k, v in hidden.items()}
    np.savez_compressed(out_dir / f"hidden_chunk_{chunk_index:04d}.npz", ids=np.asarray([r["id"] for r in records]), **arrays)
    with (out_dir / "records.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True, help="Short name used in output paths, e.g. qwen")
    ap.add_argument("--dataset", required=True, help="Dataset tag, e.g. triviaqa")
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-splits", default="", help="Comma-separated splits to run (default: all); for quick checks")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--se-samples", type=int, default=10)
    ap.add_argument("--se-splits", default="test", help="Comma-separated splits that get semantic entropy; empty for none")
    ap.add_argument("--nli-model", default="microsoft/deberta-large-mnli")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    ap.add_argument("--chat-template-mode", choices=["auto", "system_user", "user_only"], default="auto")
    ap.add_argument("--chunk-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dtype = select_dtype(args.torch_dtype, device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    if records_path.exists() and not args.resume:
        raise SystemExit(f"{records_path} exists; pass --resume to continue it")
    finished = done_ids(records_path)
    chunk_index = len(list(out_dir.glob("hidden_chunk_*.npz")))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    stops = stop_token_ids(tokenizer)
    stop_set = set(stops)
    true_ids = candidate_ids(tokenizer, ["True", " True", "true", " true"])
    false_ids = candidate_ids(tokenizer, ["False", " False", "false", " false"])

    se_splits = {s for s in args.se_splits.split(",") if s}
    judge = None
    if se_splits and args.se_samples > 0:
        nli_device = torch.device("cuda:1") if device.type == "cuda" and torch.cuda.device_count() > 1 else device
        # DeBERTa NLI models fail under float16 on GPU, so the judge always runs in float32.
        judge = NLIJudge(args.nli_model, nli_device, torch.float32)

    n_layers = model.config.num_hidden_layers
    layer_index = {tag: (n_layers if frac == 1.0 else max(1, round(n_layers * frac))) for tag, frac in LAYER_FRACTIONS.items()}

    config = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "TCL-Phase1-Kaggle-Protocol.md",
        "model": args.model,
        "model_tag": args.model_tag,
        "dataset": args.dataset,
        "questions": str(args.questions),
        "max_new_tokens": args.max_new_tokens,
        "stop_token_ids": stops,
        "p_true_true_ids": true_ids,
        "p_true_false_ids": false_ids,
        "se_samples": args.se_samples,
        "se_splits": sorted(se_splits),
        "se_temperature": 1.0,
        "nli_model": args.nli_model if judge else None,
        "hidden_state_layers": layer_index,
        "num_hidden_layers": n_layers,
        "poolings": POOLINGS,
        "correctness_method": CORRECTNESS_METHOD,
        "raw_span_method": RAW_SPAN_METHOD,
        "raw_full_method": RAW_FULL_METHOD,
        "chat_template_mode": args.chat_template_mode,
        "seed": args.seed,
        "runtime": runtime_info(device, dtype),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    only = {x for x in args.only_splits.split(",") if x} or None
    rows = [r for r in read_questions(args.questions, args.limit, only) if r["id"] not in finished]
    buffer: list[dict] = []
    hidden: dict[str, list[np.ndarray]] = {f"{t}_{p}": [] for t in layer_index for p in POOLINGS}

    for row in tqdm(rows, desc=f"{args.model_tag}/{args.dataset}"):
        question, context = row["question"].strip(), row.get("context", "").strip()
        accepted = [a.strip() for a in row["accepted_answers"].split("|") if a.strip()]
        prompt = chat_prompt(tokenizer, user_message(question, context), args.chat_template_mode)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
                eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,
            )
            seq = gen.sequences
            gen_ids = seq[0, prompt_len:].tolist()
            probs = token_probs(list(gen.logits), gen_ids)
            start, end = answer_span(tokenizer, gen_ids, stop_set)
            span_ids = gen_ids[start:end]
            answer = tokenizer.decode(span_ids, skip_special_tokens=True).strip()
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            span_p = probs[start:end]

            hs = model(input_ids=seq, attention_mask=torch.ones_like(seq), output_hidden_states=True, use_cache=False).hidden_states
            positions = list(range(prompt_len + start, prompt_len + end)) or [prompt_len - 1]
            for tag, li in layer_index.items():
                layer = hs[li][0]
                hidden[f"{tag}_answer_mean"].append(layer[positions].mean(dim=0).float().cpu().numpy())
                hidden[f"{tag}_answer_last"].append(layer[positions[-1]].float().cpu().numpy())

            pt_prompt = chat_prompt(tokenizer, p_true_content(question, context, answer), args.chat_template_mode)
            pt_inputs = tokenizer(pt_prompt, return_tensors="pt", add_special_tokens=False).to(device)
            next_probs = torch.softmax(model(**pt_inputs).logits[0, -1].float(), dim=-1)
            pt_true = float(next_probs[true_ids].sum().item())
            pt_false = float(next_probs[false_ids].sum().item())

        record = {
            "id": row["id"],
            "dataset": args.dataset,
            "split": row["split"],
            "title": row.get("title", ""),
            "question": question,
            "context": context,
            "accepted_answers": accepted,
            "model": args.model,
            "model_tag": args.model_tag,
            "raw_output": raw_output,
            "answer": answer,
            "answer_span_tokens": [start, end],
            "n_generated_tokens": len(gen_ids),
            "hit_max_new_tokens": len(gen_ids) >= args.max_new_tokens and (not gen_ids or gen_ids[-1] not in stop_set),
            "empty_answer": not answer,
            "refusal": is_refusal(raw_output),
            "correctness_label": int(bool(answer) and is_correct(answer, accepted, question)),
            "correctness_method": CORRECTNESS_METHOD,
            "raw_conf_span": geo_mean(span_p),
            "raw_conf_full": geo_mean(probs),
            "first_token_prob": float(span_p[0]) if len(span_p) else 0.0,
            "min_token_prob": float(span_p.min()) if len(span_p) else 0.0,
            "span_logprob_sum": float(np.log(span_p).sum()) if len(span_p) else 0.0,
            "p_true": pt_true / (pt_true + pt_false) if (pt_true + pt_false) > 0 else 0.5,
            "p_true_mass": pt_true + pt_false,
            "semantic_entropy": None,
            "se_n_clusters": None,
            "self_consistency": None,
            "se_samples": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if judge is not None and row["split"] in se_splits:
            with torch.no_grad():
                samp = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    num_return_sequences=args.se_samples,
                    return_dict_in_generate=True,
                    eos_token_id=stops,
                    pad_token_id=tokenizer.pad_token_id,
                )
            samples = []
            for j in range(args.se_samples):
                ids_j = samp.sequences[j, prompt_len:].tolist()
                while ids_j and ids_j[-1] == tokenizer.pad_token_id and tokenizer.pad_token_id not in stop_set:
                    ids_j.pop()
                s, e = answer_span(tokenizer, ids_j, stop_set)
                samples.append(tokenizer.decode(ids_j[s:e], skip_special_tokens=True).strip())
            clusters = cluster_answers(judge, question, [answer] + samples)
            sample_clusters = clusters[1:]
            counts = np.bincount(sample_clusters)
            freq = counts[counts > 0] / len(sample_clusters)
            record["semantic_entropy"] = float(-(freq * np.log(freq)).sum())
            record["se_n_clusters"] = int((counts > 0).sum())
            record["self_consistency"] = float(np.mean([c == clusters[0] for c in sample_clusters]))
            record["se_samples"] = samples

        buffer.append(record)
        if len(buffer) >= args.chunk_size:
            flush(out_dir, buffer, hidden, chunk_index)
            chunk_index += 1
            buffer = []
            hidden = {k: [] for k in hidden}

    flush(out_dir, buffer, hidden, chunk_index)
    print(f"Wrote {records_path}")


if __name__ == "__main__":
    main()
