from __future__ import annotations

import re


CORRECTNESS_METHOD = "strict_answer_segment_match_v2"

ROMAN_NUMERAL = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
}
COMMON_SINGLE_TOKEN_FALSE_POSITIVES = {
    "a", "an", "the", "of", "in", "on", "to", "for", "and", "or", "is", "was",
    "were", "be", "been", "it", "its", "this", "that", "he", "she", "they",
}


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_answer_text(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^\s*[-*]\s*", "", answer)
    answer = re.sub(r"^\s*(answer|a)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\bYou are an AI assistant\b.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"\bAnswer the question\b.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"\bQuestion\s*:\s*.*$", "", answer, flags=re.IGNORECASE | re.DOTALL)
    return answer.strip()


def answer_segments(answer: str) -> list[str]:
    cleaned = clean_answer_text(answer)
    if not cleaned:
        return []

    segments = [cleaned]
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    if first_line:
        segments.append(first_line)

    split_parts = re.split(r"[.;\n]|\s+-\s+", cleaned, maxsplit=1)
    if split_parts and split_parts[0].strip():
        segments.append(split_parts[0].strip())

    if ":" in cleaned:
        after_colon = cleaned.split(":", 1)[1].strip()
        if after_colon:
            segments.append(after_colon)

    deduped = []
    seen = set()
    for segment in segments:
        norm = normalize(segment)
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(segment)
    return deduped


def extract_primary_answer(answer: str) -> str:
    segments = answer_segments(answer)
    return segments[0] if segments else ""


def close_long_answer_match(norm_segment: str, norm_gold: str) -> bool:
    return False


def gold_is_question_echo(norm_gold: str, norm_question: str) -> bool:
    if not norm_gold or not norm_question:
        return False
    return len(norm_gold.split()) >= 2 and re.search(rf"\b{re.escape(norm_gold)}\b", norm_question) is not None


def single_token_match(norm_segment: str, norm_gold: str, norm_question: str) -> bool:
    tokens = norm_segment.split()
    if not tokens:
        return False
    if norm_gold.isdigit():
        return re.search(rf"\b{re.escape(norm_gold)}\b", norm_segment) is not None
    if norm_question and "river" in norm_question:
        if tokens[0] == norm_gold:
            return True
        return re.search(rf"\b{re.escape(norm_gold)}\s+river\b", norm_segment) is not None
    if tokens[0] == norm_gold:
        if len(tokens) >= 2 and tokens[1] in ROMAN_NUMERAL:
            return False
        return True
    if norm_gold not in COMMON_SINGLE_TOKEN_FALSE_POSITIVES and re.search(rf"\b{re.escape(norm_gold)}\b", norm_segment):
        return True
    return False


def segment_matches_gold(segment: str, gold: str, question: str) -> bool:
    norm_segment = normalize(segment)
    norm_gold = normalize(gold)
    norm_question = normalize(question)
    if not norm_segment or not norm_gold:
        return False
    if gold_is_question_echo(norm_gold, norm_question):
        return False
    if norm_segment == norm_gold:
        return True

    gold_tokens = norm_gold.split()
    if len(gold_tokens) == 1:
        return single_token_match(norm_segment, norm_gold, norm_question)
    if gold_tokens[0].isdigit():
        segment_tokens = norm_segment.split()
        if bool(segment_tokens) and segment_tokens[0] == gold_tokens[0]:
            return True
        return re.search(rf"\b{re.escape(norm_gold)}\b", norm_segment) is not None

    if norm_segment.startswith(norm_gold):
        return True
    if re.search(rf"\b{re.escape(norm_gold)}\b", norm_segment):
        return True
    return close_long_answer_match(norm_segment, norm_gold)


def is_correct(answer: str, accepted: list[str], question: str = "") -> bool:
    segments = answer_segments(answer)
    if not segments:
        return False
    for segment in segments:
        for gold in accepted:
            if segment_matches_gold(segment, gold, question):
                return True
    return False
