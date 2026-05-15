from __future__ import annotations

import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "with",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def keywords(text: str, limit: int = 12) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", text.lower())
    counts = Counter(word for word in words if word not in STOPWORDS)
    return [word for word, _ in counts.most_common(limit)]


def extractive_summary(text: str, max_sentences: int = 3) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return sentences

    key_terms = set(keywords(text, limit=16))
    scored = []
    for index, sentence in enumerate(sentences):
        terms = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", sentence.lower())
        overlap = sum(1 for term in terms if term in key_terms)
        position_bonus = 1.0 if index == 0 else 0.0
        scored.append((overlap + position_bonus, index, sentence))

    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:max_sentences]
    return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]

