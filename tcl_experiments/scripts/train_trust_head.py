"""Phase 2: train and evaluate one trust-head arm (docs/phase2/TCL-Phase2-Protocol.md).

Arms
  A  frozen backbone, MLP trust head trained on L_trust
  B  LoRA backbone + head, trained on L_trust
  C  LoRA backbone + head, trained on L_LM + lambda * L_trust   (TCL-J)
  D  LoRA backbone trained on L_LM, then frozen; head trained on L_trust

One call trains one arm on one dataset with one seed, then writes:
  scores_fixed.csv     trust scores on the fixed base-model answers (validation of the
                       training dataset, and both test sets)
  regenerated.csv      the trained model's own test answers, labels, and trust scores
                       (arms B, C, D; and A when --regenerate is passed, as the base model)
  metrics.json         config, per-epoch validation losses, chosen epoch, validation AUROC,
                       generation accuracy, and held-out perplexity
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from labeling import CORRECTNESS_METHOD_V3, is_correct_v3, is_refusal
from run_inference import select_device, user_message
from run_inference_v2 import answer_span, chat_prompt, stop_token_ids

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("triviaqa", "squad")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------- data


def load_dataset_rows(ds: str, questions_dir: Path, records_dir: Path, tokenizer, mode: str) -> dict[str, list[dict]]:
    with (questions_dir / f"{ds}.csv").open(encoding="utf-8", newline="") as f:
        questions = {r["id"]: r for r in csv.DictReader(f)}
    answers = {}
    with (records_dir / ds / "records.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            answers[r["id"]] = r["answer"]
    by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    for qid, q in questions.items():
        if qid not in answers:
            continue
        accepted = [a.strip() for a in q["accepted_answers"].split("|") if a.strip()]
        prompt = chat_prompt(tokenizer, user_message(q["question"].strip(), q.get("context", "").strip()), mode)
        answer = answers[qid]
        by_split[q["split"]].append({
            "id": qid, "dataset": ds, "split": q["split"], "question": q["question"].strip(),
            "context": q.get("context", "").strip(), "accepted": accepted, "prompt": prompt,
            "answer": answer, "reference": accepted[0], "label": int(is_correct_v3(answer, accepted, q["question"])),
            "refusal": is_refusal(answer),
        })
    return by_split


def encode_trust(tokenizer, prompt: str, answer: str) -> tuple[list[int], int, int]:
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a = tokenizer(answer, add_special_tokens=False)["input_ids"] if answer else []
    return p + a, len(p), len(p) + len(a)


def encode_lm(tokenizer, prompt: str, reference: str) -> tuple[list[int], list[int]]:
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    r = tokenizer(reference, add_special_tokens=False)["input_ids"]
    return p + r, [-100] * len(p) + r


def pad(seqs: list[list[int]], value: int) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs], dtype=torch.long)


def trust_batch(tokenizer, rows: list[dict], device) -> dict:
    enc = [encode_trust(tokenizer, r["prompt"], r["answer"]) for r in rows]
    ids = pad([e[0] for e in enc], tokenizer.pad_token_id)
    attn = pad([[1] * len(e[0]) for e in enc], 0)
    pool = torch.zeros_like(ids, dtype=torch.float32)
    for i, (_, start, end) in enumerate(enc):
        if end > start:
            pool[i, start:end] = 1.0
        else:
            pool[i, start - 1] = 1.0  # empty answer: last prompt token
    return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "pool": pool.to(device),
            "labels": torch.tensor([float(r["label"]) for r in rows], device=device)}


def lm_batch(tokenizer, rows: list[dict], device) -> dict:
    enc = [encode_lm(tokenizer, r["prompt"], r["reference"]) for r in rows]
    return {"input_ids": pad([e[0] for e in enc], tokenizer.pad_token_id).to(device),
            "attention_mask": pad([[1] * len(e[0]) for e in enc], 0).to(device),
            "labels": pad([e[1] for e in enc], -100).to(device)}


# ---------------------------------------------------------------- model


class TrustHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pooled_features(model, batch: dict, layer: int) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True, use_cache=False)
    h = out.hidden_states[layer].float()
    w = batch["pool"].unsqueeze(-1)
    return (h * w).sum(1) / w.sum(1).clamp(min=1.0)


def lm_loss(model, batch: dict) -> torch.Tensor:
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], use_cache=False).loss


def add_lora(model, seed: int):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    targets = [t for t in LORA_TARGETS if any(n.endswith(t) for n, _ in model.named_modules())]
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets, task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


# ---------------------------------------------------------------- training


def batches(rows: list[dict], size: int, rng: random.Random, shuffle: bool):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), size):
        yield [rows[j] for j in order[i : i + size]]


@torch.no_grad()
def features_for(model, tokenizer, rows, layer, device, micro) -> torch.Tensor:
    model.eval()
    out = []
    for b in batches(rows, micro, random.Random(0), shuffle=False):
        with autocast(device):
            out.append(pooled_features(model, trust_batch(tokenizer, b, device), layer).float().cpu())
    return torch.cat(out)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def train_head_on_features(head, feats, labels, val_feats, val_labels, args, rng, device):
    """Arms A and D: the backbone is frozen, so features are computed once."""
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr_head, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(labels) / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    bce = nn.BCEWithLogitsLoss()
    history, best = [], (float("inf"), None, -1)
    for epoch in range(args.epochs):
        head.train()
        order = list(range(len(labels)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            idx = order[i : i + args.batch_size]
            loss = bce(head(feats[idx].to(device)), labels[idx].to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            val_loss = float(bce(head(val_feats.to(device)), val_labels.to(device)))
        history.append({"epoch": epoch + 1, "val_trust_loss": val_loss})
        if val_loss < best[0]:
            best = (val_loss, copy.deepcopy(head.state_dict()), epoch + 1)
    head.load_state_dict(best[1])
    return history, best[2]


def train_joint(model, head, tokenizer, train_rows, val_rows, args, rng, device, use_trust: bool, use_lm: bool, lam: float):
    """Arms B (trust only), C (LM + lambda * trust), and D's first stage (LM only)."""
    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr_lora}]
    if use_trust:
        params.append({"params": head.parameters(), "lr": args.lr_head})
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    accum = max(1, args.batch_size // args.micro_batch)
    micro_per_epoch = math.ceil(len(train_rows) / args.micro_batch)
    steps = args.epochs * math.ceil(micro_per_epoch / accum)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    bce = nn.BCEWithLogitsLoss()
    trainable = [p for g in params for p in g["params"]]
    history, best = [], (float("inf"), None, -1)

    def optimizer_step(counts):
        # fp16 loss scaling skips a step when gradients overflow; count those so a run that never updated is visible.
        scale_before = scaler.get_scale() if scaler.is_enabled() else None
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sched.step()
        skipped = scale_before is not None and scaler.get_scale() < scale_before
        counts["skipped" if skipped else "taken"] += 1

    for epoch in range(args.epochs):
        model.train()
        head.train()
        opt.zero_grad()
        counts = {"taken": 0, "skipped": 0}
        for step, b in enumerate(batches(train_rows, args.micro_batch, rng, shuffle=True)):
            with autocast(device):
                loss = torch.zeros((), device=device)
                if use_lm:
                    loss = loss + lm_loss(model, lm_batch(tokenizer, b, device))
                if use_trust:
                    tb = trust_batch(tokenizer, b, device)
                    loss = loss + lam * bce(head(pooled_features(model, tb, args.layer)), tb["labels"])
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == micro_per_epoch:
                optimizer_step(counts)
        # Checkpoint selection: lowest validation trust loss (LM loss for D's LM-only stage).
        model.eval()
        head.eval()
        with torch.no_grad():
            if use_trust:
                vf = features_for(model, tokenizer, val_rows, args.layer, device, args.micro_batch)
                vl = torch.tensor([float(r["label"]) for r in val_rows])
                val_loss = float(bce(head(vf.to(device)).float().cpu(), vl))
            else:
                losses = []
                for b in batches(val_rows, args.micro_batch, rng, shuffle=False):
                    with autocast(device):
                        losses.append(float(lm_loss(model, lm_batch(tokenizer, b, device))))
                val_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, ("val_trust_loss" if use_trust else "val_lm_loss"): val_loss,
                        "optimizer_steps": counts["taken"], "skipped_steps": counts["skipped"]})
        if val_loss < best[0]:
            state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            best = (val_loss, (state, copy.deepcopy(head.state_dict())), epoch + 1)
    state, head_state = best[1]
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.copy_(state[n])
    head.load_state_dict(head_state)
    return history, best[2]


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def score_rows(model, head, tokenizer, rows, layer, device, micro) -> np.ndarray:
    head.eval()
    feats = features_for(model, tokenizer, rows, layer, device, micro)
    return head(feats.to(device)).float().cpu().numpy()


@torch.no_grad()
def regenerate(model, head, tokenizer, rows, layer, device, max_new_tokens) -> list[dict]:
    model.eval()
    stops = stop_token_ids(tokenizer)
    stop_set = set(stops)
    out = []
    for r in rows:
        inputs = tokenizer(r["prompt"], return_tensors="pt", add_special_tokens=False).to(device)
        n = inputs["input_ids"].shape[1]
        with autocast(device):
            seq = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
        gen = seq[0, n:].tolist()
        s, e = answer_span(tokenizer, gen, stop_set)
        answer = tokenizer.decode(gen[s:e], skip_special_tokens=True).strip()
        ids = seq[0, : n + e] if e > s else seq[0, :n]
        pool = torch.zeros(1, ids.shape[0], device=device)
        if e > s:
            pool[0, n + s : n + e] = 1.0
        else:
            pool[0, n - 1] = 1.0
        batch = {"input_ids": ids.unsqueeze(0), "attention_mask": torch.ones_like(ids).unsqueeze(0), "pool": pool}
        with autocast(device):
            logit = float(head(pooled_features(model, batch, layer)).float())
        out.append({"id": r["id"], "dataset": r["dataset"], "answer": answer,
                    "label": int(is_correct_v3(answer, r["accepted"], r["question"])),
                    "refusal": is_refusal(answer), "trust_logit": logit})
    return out


@torch.no_grad()
def perplexity(model, tokenizer, paragraphs: list[str], device, max_tokens: int = 512) -> float:
    model.eval()
    nll, count = 0.0, 0
    for text in paragraphs:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][:, :max_tokens].to(device)
        if ids.shape[1] < 2:
            continue
        with autocast(device):
            loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        nll += float(loss) * (ids.shape[1] - 1)
        count += ids.shape[1] - 1
    return float(math.exp(nll / count))


def _versions() -> dict:
    import transformers

    try:
        import peft
        peft_version = peft.__version__
    except ImportError:
        peft_version = None
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft_version}


def safe_auc(labels, scores) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--train-dataset", choices=DATASETS, required=True)
    ap.add_argument("--lam", type=float, default=1.0, help="lambda for arm C")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--questions-dir", type=Path, default=ROOT / "data" / "phase1_kaggle")
    ap.add_argument("--records-dir", type=Path, default=ROOT / "runs" / "phase1_kaggle" / "qwen")
    ap.add_argument("--wikitext", type=Path, default=ROOT / "data" / "phase2" / "wikitext2_test_200.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--layer-frac", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--regenerate", action="store_true", help="also regenerate test answers (always on for B, C, D)")
    ap.add_argument("--chat-template-mode", default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--limit", type=int, default=None, help="rows per split, for quick checks only")
    ap.add_argument("--wikitext-limit", type=int, default=None, help="paragraphs, for quick checks only")
    args = ap.parse_args()

    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = select_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(device)
    n_layers = model.config.num_hidden_layers
    args.layer = max(1, round(n_layers * args.layer_frac))

    data = {ds: load_dataset_rows(ds, args.questions_dir, args.records_dir, tokenizer, args.chat_template_mode) for ds in DATASETS}
    if args.limit:
        for ds in data:
            for split in data[ds]:
                data[ds][split] = data[ds][split][: args.limit]
    train_rows = data[args.train_dataset]["train"]
    val_rows = data[args.train_dataset]["validation"]
    head = TrustHead(model.config.hidden_size).to(device)

    if args.arm in ("B", "C", "D"):
        model = add_lora(model, args.seed)
    for n, p in model.named_parameters():
        p.requires_grad = args.arm in ("B", "C", "D") and "lora_" in n

    if args.arm == "A":
        feats = features_for(model, tokenizer, train_rows, args.layer, device, args.micro_batch)
        vfeats = features_for(model, tokenizer, val_rows, args.layer, device, args.micro_batch)
        history, best_epoch = train_head_on_features(
            head, feats, torch.tensor([float(r["label"]) for r in train_rows]),
            vfeats, torch.tensor([float(r["label"]) for r in val_rows]), args, rng, device)
    elif args.arm == "B":
        history, best_epoch = train_joint(model, head, tokenizer, train_rows, val_rows, args, rng, device, True, False, 1.0)
    elif args.arm == "C":
        history, best_epoch = train_joint(model, head, tokenizer, train_rows, val_rows, args, rng, device, True, True, args.lam)
    else:  # D: LM-only LoRA, then freeze and train the head on the adapted backbone
        history, best_epoch = train_joint(model, head, tokenizer, train_rows, val_rows, args, rng, device, False, True, 0.0)
        for p in model.parameters():
            p.requires_grad = False
        feats = features_for(model, tokenizer, train_rows, args.layer, device, args.micro_batch)
        vfeats = features_for(model, tokenizer, val_rows, args.layer, device, args.micro_batch)
        head_history, head_epoch = train_head_on_features(
            head, feats, torch.tensor([float(r["label"]) for r in train_rows]),
            vfeats, torch.tensor([float(r["label"]) for r in val_rows]), args, rng, device)
        history = {"lm_stage": history, "head_stage": head_history}
        best_epoch = {"lm_stage": best_epoch, "head_stage": head_epoch}

    # Scores on the fixed base-model answers.
    eval_sets = [(args.train_dataset, "validation")] + [(ds, "test") for ds in DATASETS]
    fixed_rows, val_auc = [], None
    for ds, split in eval_sets:
        rows = data[ds][split]
        logits = score_rows(model, head, tokenizer, rows, args.layer, device, args.micro_batch)
        labels = [r["label"] for r in rows]
        if split == "validation":
            val_auc = safe_auc(labels, logits)
        for r, lg in zip(rows, logits):
            fixed_rows.append({"id": r["id"], "dataset": ds, "split": split, "label": r["label"],
                               "refusal": r["refusal"], "trust_logit": float(lg)})
    with (args.out_dir / "scores_fixed.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fixed_rows[0]))
        w.writeheader()
        w.writerows(fixed_rows)

    # Regenerated answers and generation quality.
    generation = None
    if args.arm in ("B", "C", "D") or args.regenerate:
        regen = []
        for ds in DATASETS:
            regen += regenerate(model, head, tokenizer, data[ds]["test"], args.layer, device, args.max_new_tokens)
        with (args.out_dir / "regenerated.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(regen[0]))
            w.writeheader()
            w.writerows(regen)
        paragraphs = json.loads(args.wikitext.read_text(encoding="utf-8"))["paragraphs"][: args.wikitext_limit]
        generation = {
            "accuracy": {ds: float(np.mean([r["label"] for r in regen if r["dataset"] == ds])) for ds in DATASETS},
            "refusals": {ds: int(sum(r["refusal"] for r in regen if r["dataset"] == ds)) for ds in DATASETS},
            "regenerated_trust_auroc": {ds: safe_auc([r["label"] for r in regen if r["dataset"] == ds],
                                                     [r["trust_logit"] for r in regen if r["dataset"] == ds]) for ds in DATASETS},
            "wikitext_perplexity": perplexity(model, tokenizer, paragraphs, device),
            "wikitext_paragraphs": len(paragraphs),
        }

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/phase2/TCL-Phase2-Protocol.md",
        "arm": args.arm, "train_dataset": args.train_dataset, "seed": args.seed,
        "lambda": args.lam if args.arm == "C" else None,
        "model": args.model, "layer": args.layer, "num_hidden_layers": n_layers,
        "correctness_method": CORRECTNESS_METHOD_V3,
        "hyperparameters": {k: getattr(args, k) for k in ("epochs", "batch_size", "micro_batch", "lr_lora", "lr_head", "layer_frac", "max_new_tokens")},
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05} if args.arm in ("B", "C", "D") else None,
        "n": {ds: {s: len(v) for s, v in data[ds].items()} for ds in DATASETS},
        "history": history, "best_epoch": best_epoch,
        "validation_auroc": val_auc,
        "generation": generation,
        "limit": args.limit,
        "device": str(device),
        "versions": _versions(),
        "seconds": round(time.time() - started, 1),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("arm", "train_dataset", "seed", "lambda", "best_epoch", "validation_auroc", "seconds")}))


if __name__ == "__main__":
    main()
