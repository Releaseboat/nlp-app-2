"""
Ranking / retrieval metrics — implemented from scratch.

All functions take a list of *ranked* prediction lists and the
corresponding list of relevant items (set per query) and return a single
float in [0, 1].  Implementations follow the canonical definitions used in
Manning et al., *Introduction to Information Retrieval*, and the original
TREC evaluation guidelines.

Two relevance regimes are supported through the same code path:

    Document-level (RAG retrieval)
        Items are document IDs.  An item is relevant if it equals the
        gold document ID.

    Answer-level (Extractive Top-K)
        Items are predicted answer strings (normalised).  An item is
        relevant if it matches any of the gold answer strings under the
        SQuAD-style normalisation.

Both regimes simply differ in how `relevant` sets are constructed before
being passed in here.  The metrics themselves are oblivious.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Recall @ K
# ---------------------------------------------------------------------------
def recall_at_k(
    predictions: Sequence[Sequence[str]],
    relevants: Sequence[Iterable[str]],
    k: int,
) -> float:
    """
    Recall@K (binary, per query):
        1.0 if any of the top-K predictions is in the relevant set,
        else 0.0.  Averaged across queries.

    This is the standard "hit rate" interpretation used by the open-domain
    QA community (Karpukhin et al., 2020, DPR).
    """
    if not predictions:
        return 0.0
    hits = 0
    for preds, rels in zip(predictions, relevants):
        rel_set = set(rels)
        if not rel_set:
            continue  # unanswerable / no gold — skip, see note in __all__
        if set(preds[:k]) & rel_set:
            hits += 1
    n = sum(1 for r in relevants if set(r))
    return hits / n if n else 0.0


def recall_at_k_curve(
    predictions: Sequence[Sequence[str]],
    relevants: Sequence[Iterable[str]],
    max_k: int = 10,
) -> List[float]:
    """Convenience helper returning Recall@1..Recall@max_k."""
    return [recall_at_k(predictions, relevants, k) for k in range(1, max_k + 1)]


# ---------------------------------------------------------------------------
# Mean Reciprocal Rank
# ---------------------------------------------------------------------------
def reciprocal_rank(preds: Sequence[str], rels: Iterable[str]) -> float:
    rel_set = set(rels)
    for i, p in enumerate(preds, start=1):
        if p in rel_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    predictions: Sequence[Sequence[str]],
    relevants: Sequence[Iterable[str]],
) -> float:
    """
    MRR averaged over queries that have at least one relevant item.
    Unanswerable queries (empty relevance set) are excluded from the
    denominator — including them would conflate "system mis-ranked" with
    "no correct answer exists".
    """
    rrs = []
    for preds, rels in zip(predictions, relevants):
        if not set(rels):
            continue
        rrs.append(reciprocal_rank(preds, rels))
    return float(np.mean(rrs)) if rrs else 0.0


# ---------------------------------------------------------------------------
# Mean Average Precision
# ---------------------------------------------------------------------------
def average_precision(preds: Sequence[str], rels: Iterable[str]) -> float:
    """
    Average Precision = mean of precision values computed at each rank
    where a *previously unseen* relevant item appears.

        AP = (1 / |R|) * Σ_{k: pred_k is relevant} Precision@k

    where |R| is the number of distinct relevant items.  Duplicate
    relevant items in the ranked list are ignored after the first
    occurrence (standard IR convention — Manning et al. 2008).
    """
    rel_set = set(rels)
    if not rel_set:
        return 0.0
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for i, p in enumerate(preds, start=1):
        if p in rel_set and p not in seen:
            seen.add(p)
            hits += 1
            precision_sum += hits / i
    return precision_sum / len(rel_set)


def mean_average_precision(
    predictions: Sequence[Sequence[str]],
    relevants: Sequence[Iterable[str]],
) -> float:
    """MAP across queries with at least one relevant item."""
    aps = []
    for preds, rels in zip(predictions, relevants):
        if not set(rels):
            continue
        aps.append(average_precision(preds, rels))
    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# SQuAD-style EM / F1 over Top-K (best-of-K)
# ---------------------------------------------------------------------------
def topk_best_em_f1(predictions, golds_per_query) -> tuple[float, float]:
    """
    For each question take the best EM and best F1 across all K
    candidates.  Mirrors the SQuAD oracle metric used to upper-bound
    extractive performance.
    """
    from src.utils import exact_match, f1_score
    em_scores, f1_scores = [], []
    for preds, golds in zip(predictions, golds_per_query):
        if not golds:
            continue
        best_em = max(exact_match(p, g) for p in preds for g in golds) if preds else 0
        best_f1 = max(f1_score(p, g) for p in preds for g in golds) if preds else 0.0
        em_scores.append(best_em)
        f1_scores.append(best_f1)
    if not em_scores:
        return 0.0, 0.0
    return float(np.mean(em_scores)), float(np.mean(f1_scores))


# ---------------------------------------------------------------------------
# No-answer (SQuAD v2) classification metrics
# ---------------------------------------------------------------------------
def no_answer_classification(
    is_unanswerable_gold: Sequence[bool],
    predicted_no_answer: Sequence[bool],
) -> dict:
    """
    Treat the no-answer flag as a binary classifier.

    Returns precision, recall, F1 and accuracy for the 'unanswerable' class.
    """
    tp = fp = tn = fn = 0
    for g, p in zip(is_unanswerable_gold, predicted_no_answer):
        if g and p:
            tp += 1
        elif g and not p:
            fn += 1
        elif (not g) and p:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, tp + fp + tn + fn)
    return {
        "no_answer_precision": prec,
        "no_answer_recall": rec,
        "no_answer_f1": f1,
        "no_answer_accuracy": acc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
