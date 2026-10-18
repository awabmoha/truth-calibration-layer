from __future__ import annotations

import difflib
import re
import unicodedata


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


# ----------------------------------------------------------------------------
# Refusal detection (shared by inference v2 and the analyses)

REFUSAL_PATTERN = re.compile(
    r"\b(does not (?:provide|mention|specify|contain|say|state)|do(?:es)? not (?:know|have)|don't know|"
    r"cannot (?:provide|answer|determine|find)|can't (?:provide|answer|determine)|unable to|i'm sorry|i am sorry|"
    r"not (?:mentioned|specified|provided|stated|given) in|no (?:information|answer) (?:is |was )?(?:provided|given|available))\b",
    re.IGNORECASE,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERN.search(text))


# ----------------------------------------------------------------------------
# Labeling rule v3 (Phase 2 onward). Rule v2 above stays unchanged so that
# Phase 1 results remain reproducible. v3 fixes the form-level errors found in
# the Phase 1 label audit: plurals, UK/US spelling, articles and "and" in
# lists, units, UK/US abbreviations, one-letter transliteration variants,
# digits matching inside decimals, and refusals that repeat gold words.
# Synonyms and paraphrases are out of scope for a string rule.

CORRECTNESS_METHOD_V3 = "answer_match_v3"

_DROP_TOKENS = {"a", "an", "the", "and"}
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}
_UNITS = {
    "ml": "milliliter", "millilitre": "milliliter", "milliliter": "milliliter",
    "km": "kilometer", "kilometre": "kilometer", "cm": "centimeter", "centimetre": "centimeter",
    "mm": "millimeter", "millimetre": "millimeter", "metre": "meter", "litre": "liter",
    "kg": "kilogram", "lb": "pound", "lbs": "pound", "percent": "%",
    "rpm": "revolution per minute", "kph": "kilometer per hour", "mph": "mile per hour",
}
_DROP_UNIT_WORDS = {"degree", "deg"}  # "100 degrees Celsius" vs "100 °C"
_TITLES = {"dr", "mr", "mrs", "ms", "sir", "dame", "st", "prof"}  # honorifics only, never content words
_UNITS_AFTER_NUMBER = {  # ambiguous abbreviations, expanded only right after a number
    "m": "meter", "l": "liter", "g": "gram", "ft": "foot", "in": "inch", "mi": "mile", "s": "second",
    "c": "celsius", "f": "fahrenheit",
}
_UNIT_WORDS = {w for v in list(_UNITS.values()) + list(_UNITS_AFTER_NUMBER.values()) for w in v.split()} | {
    "foot", "inch", "mile", "yard", "ton", "tonne", "hour", "minute", "second", "day", "week", "month", "year",
    "dollar", "euro", "pound", "acre", "hectare", "horsepower", "volt", "watt", "kelvin", "%", "building", "people",
}
_UK_US = {
    "colour": "color", "honour": "honor", "favour": "favor", "labour": "labor", "neighbour": "neighbor",
    "behaviour": "behavior", "harbour": "harbor", "flavour": "flavor", "humour": "humor",
    "armour": "armor", "rumour": "rumor", "vapour": "vapor", "centre": "center", "theatre": "theater",
    "fibre": "fiber", "defence": "defense", "offence": "offense", "licence": "license",
    "programme": "program", "grey": "gray", "aluminium": "aluminum", "catalogue": "catalog",
    "analyse": "analyze", "jewellery": "jewelry", "tyre": "tire", "plough": "plow",
}
_SPELLING_SUFFIXES = (("isation", "ization"), ("ising", "izing"), ("ised", "ized"), ("yse", "yze"))


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _singular(tok: str) -> str:
    if len(tok) <= 3 or not tok.isalpha():
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith(("sses", "xes", "zes", "ches", "shes")):
        return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "sis", "tis", "nis")):
        return tok[:-1]
    return tok


def _spelling(tok: str) -> str:
    tok = _UK_US.get(tok, tok)
    for uk, us in _SPELLING_SUFFIXES:
        if tok.endswith(uk) and len(tok) > len(uk) + 2:
            return tok[: -len(uk)] + us
    return tok


def normalize_tokens_v3(text: str) -> list[str]:
    text = _strip_accents(text)
    text = re.sub(r"\bU\.?\s?S\.?\s?A\.?(?=\W|$)|\bU\.\s?S\.(?=\W|$)|\bUS\b", " usa ", text)
    text = re.sub(r"\bU\.?\s?K\.?(?=\W|$)", " uk ", text)
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\bunited states(?: of america)?\b", " usa ", text)
    text = re.sub(r"\bunited kingdom\b", " uk ", text)
    text = re.sub(r"\bper cent\b", " % ", text)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000
    out: list[str] = []
    for tok in re.findall(r"\d+(?:\.\d+)?|[a-z0-9]+|%", text):
        if tok in _DROP_TOKENS:
            continue
        tok = _NUMBER_WORDS.get(tok, tok)
        if re.fullmatch(r"\d+\.0+", tok):
            tok = tok.split(".")[0]  # "6.0" -> "6"
        if out and _is_number(out[-1]) and tok in _UNITS_AFTER_NUMBER:
            out.append(_UNITS_AFTER_NUMBER[tok])  # "1.5 m" -> "1.5 meter", only after a number
            continue
        tok = _spelling(_singular(tok))
        if tok in _DROP_UNIT_WORDS:
            continue
        out.extend(_UNITS.get(tok, tok).split())
    # "12 to 15" and "12-15" both become "12 15".
    out = [t for i, t in enumerate(out)
           if not (t == "to" and 0 < i < len(out) - 1 and _is_number(out[i - 1]) and _is_number(out[i + 1]))]
    # Drop a title only when a name follows it: "Dr Pangloss" -> "pangloss", while
    # "the President" and "President of the United States" keep "president".
    return [t for i, t in enumerate(out)
            if not (t in _TITLES and i + 1 < len(out) and out[i + 1] not in _TITLES and out[i + 1] != "of")]


def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", tok))


def _count(hay: list[str], needle: list[str]) -> int:
    n = len(needle)
    return sum(hay[i : i + n] == needle for i in range(len(hay) - n + 1))


def answer_segments_v3(answer: str) -> list[str]:
    cleaned = clean_answer_text(answer)
    if not cleaned:
        return []
    segments = [cleaned]
    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    if first_line:
        segments.append(first_line)
    first_sentence = re.split(r"(?<!\d)[.;!?](?:\s|$)|\n", cleaned, maxsplit=1)[0].strip()
    if first_sentence:
        segments.append(first_sentence)
    if ":" in cleaned:
        after = cleaned.split(":", 1)[1].strip()
        if after:
            segments.append(after)
    seen, out = set(), []
    for segment in segments:
        key = " ".join(normalize_tokens_v3(segment))
        if key and key not in seen:
            seen.add(key)
            out.append(segment)
    return out


def _contains(hay: list[str], needle: list[str], fuzzy: bool) -> bool:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        window = hay[i : i + n]
        if window == needle:
            return True
        if fuzzy and n >= 2:
            diffs = [(a, b) for a, b in zip(window, needle) if a != b]
            if len(diffs) == 1:
                a, b = diffs[0]
                if min(len(a), len(b)) >= 5 and abs(len(a) - len(b)) <= 1 and a.isalpha() and b.isalpha() \
                        and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8:
                    return True
    return False


def segment_matches_gold_v3(segment: str, gold: str, question: str) -> bool:
    seg = normalize_tokens_v3(segment)
    gold_toks = normalize_tokens_v3(gold)
    if not seg or not gold_toks:
        return False
    q_toks = normalize_tokens_v3(question)
    # A gold answer that also appears in the question counts only if the answer
    # uses it more often than the question does, i.e. supplies it rather than
    # repeating it ("Mercury is the working fluid in a mercury vapor turbine").
    supplies_gold = _count(seg, gold_toks) > _count(q_toks, gold_toks)
    if len(gold_toks) >= 2 and _contains(q_toks, gold_toks, fuzzy=True) and not supplies_gold:
        return False  # the gold answer only echoes the question
    if seg == gold_toks:
        return True
    if _is_number(gold_toks[0]) and seg[0] == gold_toks[0] and len(gold_toks) >= 2:
        # "300 km" vs "300 km long", "2,000" vs "2,000 buildings": same number, real unit
        if gold_toks[1] in _UNIT_WORDS and (len(seg) == 1 or seg[1] == gold_toks[1]):
            return True
    if len(gold_toks) == 1:
        g = gold_toks[0]
        is_choice_question = " or " in f" {question.lower()} "
        if g in q_toks and not _is_number(g) and not is_choice_question and not supplies_gold:
            return False  # a one-word gold answer that only repeats the question
        if g in COMMON_SINGLE_TOKEN_FALSE_POSITIVES:
            return seg[0] == g
        if seg[0] == g:
            return not (len(seg) >= 2 and seg[1] in ROMAN_NUMERAL)
        return g in seg
    return _contains(seg, gold_toks, fuzzy=True)


def is_correct_v3(answer: str, accepted: list[str], question: str = "") -> bool:
    if not answer:
        return False
    refusal = REFUSAL_PATTERN.search(answer)
    if refusal:
        # Only the text before a refusal phrase can count: a refusal that repeats
        # gold words is wrong, but a real answer followed by a disclaimer is kept.
        answer = answer[: refusal.start()]
    golds = []
    for gold in accepted:
        golds.append(gold)
        # Drop a bracketed aside only when it holds a number or a short symbol:
        # "1.5 m (4 ft 11 in)" -> "1.5 m", "copper (Cu)" -> "copper". Brackets that
        # disambiguate, like "Retriever (Labrador)" or "Bat (cricket)", are kept.
        without_aside = re.sub(r"\((?=[^)]*\d)[^)]*\)|\([A-Za-z]{1,3}\)", " ", gold).strip()
        if without_aside and without_aside != gold:
            golds.append(without_aside)
    for segment in answer_segments_v3(answer):
        for gold in golds:
            if segment_matches_gold_v3(segment, gold, question):
                return True
    return False
