from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def normalize(text: str):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def token_f1(a: str, b: str):
    a_tokens = normalize(a).split()
    b_tokens = normalize(b).split()
    if not a_tokens or not b_tokens:
        return 0.0
    common = set(a_tokens) & set(b_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(b_tokens)
    return 2 * precision * recall / (precision + recall)


def accepted_aliases(text: str):
    return [part.strip() for part in (text or "").split("|") if part.strip()]


def easy_label(model_answer: str, accepted_answers: str):
    model_norm = normalize(model_answer)
    if not model_norm:
        return None, "blank model answer"

    best_f1 = 0.0
    best_alias = ""
    for alias in accepted_aliases(accepted_answers):
        alias_norm = normalize(alias)
        if not alias_norm:
            continue
        if model_norm == alias_norm:
            return 1, f"assistant-assisted exact alias match: {alias}"
        if alias_norm in model_norm or model_norm in alias_norm:
            short_side = min(len(alias_norm.split()), len(model_norm.split()))
            if short_side <= 4:
                return 1, f"assistant-assisted contained alias match: {alias}"
        f1 = token_f1(model_answer, alias)
        if f1 > best_f1:
            best_f1 = f1
            best_alias = alias

    if best_f1 >= 0.85:
        return 1, f"assistant-assisted high token overlap with alias: {best_alias}"
    if best_f1 == 0.0:
        return 0, "assistant-assisted no lexical overlap with accepted aliases"
    return None, f"needs semantic review; best alias overlap={best_f1:.3f} ({best_alias})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--uncertain-out", required=True)
    args = parser.parse_args()

    in_path = Path(args.review_csv)
    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    uncertain = []
    for row in rows:
        label, note = easy_label(row.get("model_answer", ""), row.get("accepted_answers", ""))
        if label is None:
            row["manual_correctness_label"] = ""
            row["review_notes"] = note
            row["changes_aggregate_interpretation"] = ""
            uncertain.append(row)
            continue
        auto_label = row.get("auto_correctness_label", "")
        row["manual_correctness_label"] = str(label)
        row["review_notes"] = note
        row["changes_aggregate_interpretation"] = "yes" if auto_label in {"0", "1"} and str(label) != auto_label else "no"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    uncertain_path = Path(args.uncertain_out)
    uncertain_path.parent.mkdir(parents=True, exist_ok=True)
    with uncertain_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(uncertain)

    print({
        "rows": len(rows),
        "auto_filled": len(rows) - len(uncertain),
        "uncertain": len(uncertain),
        "out": str(out_path),
        "uncertain_out": str(uncertain_path),
    })


if __name__ == "__main__":
    main()
