"""
Shared helpers: text normalisation, EM / F1, seeding.

The SQuAD v2 normalisation logic is the canonical one published by the
original SQuAD authors (lower-casing, article removal, punctuation
stripping, whitespace collapsing).
"""
from __future__ import annotations

import random
import re
import string
from collections import Counter
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# SQuAD-style text normalisation (Rajpurkar et al., 2016)
# ---------------------------------------------------------------------------
_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lower-case, strip articles + punctuation, collapse whitespace."""
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCT)
    text = _ARTICLES.sub(" ", text)
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between two strings after normalisation."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not p_tokens or not g_tokens:
        return float(p_tokens == g_tokens)
    common = Counter(p_tokens) & Counter(g_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def best_match_against_golds(pred: str, golds: Iterable[str]) -> tuple[int, float]:
    """Return (best_em, best_f1) of `pred` against any of the gold answers."""
    golds = list(golds) or [""]
    em = max(exact_match(pred, g) for g in golds)
    f1 = max(f1_score(pred, g) for g in golds)
    return em, f1


def is_unanswerable(golds: Iterable[str]) -> bool:
    """SQuAD v2: a question is unanswerable iff every gold answer is empty."""
    golds = list(golds)
    return len(golds) == 0 or all(not g.strip() for g in golds)
