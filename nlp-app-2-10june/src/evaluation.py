"""
End-to-end evaluation helpers — orchestrates metric computation against
the predictions produced by the two pipelines and writes machine-readable
outputs (JSON / CSV) for downstream analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd

from src.utils import normalize_answer, exact_match, f1_score
from src.metrics import (
    recall_at_k,
    recall_at_k_curve,
    mean_reciprocal_rank,
    mean_average_precision,
    topk_best_em_f1,
    no_answer_classification,
)


# ---------------------------------------------------------------------------
# Extractive evaluation
# ---------------------------------------------------------------------------
def evaluate_extractive(examples, predictions, max_k: int = 10) -> dict:
    """
    Compute answer-level retrieval metrics by treating the top-K
    candidate spans as a ranked list.

    A prediction matches a gold answer iff their normalised forms are
    equal — same equivalence used by the official SQuAD evaluation
    script.
    """
    pred_lists: list[list[str]] = []
    gold_sets: list[list[str]] = []

    answerable_examples = []
    answerable_preds = []

    for ex, pred in zip(examples, predictions):
        # treat each prediction's normalised text as its "doc id"
        pred_norm = [normalize_answer(c.text) for c in pred.candidates]
        gold_norm = [normalize_answer(a) for a in ex.answers if a.strip()]
        pred_lists.append(pred_norm)
        gold_sets.append(gold_norm)
        if not ex.is_unanswerable:
            answerable_examples.append(ex)
            answerable_preds.append(pred)

    curve = recall_at_k_curve(pred_lists, gold_sets, max_k=max_k)
    metrics = {
        "recall_at_k": {k: v for k, v in zip(range(1, max_k + 1), curve)},
        "mrr": mean_reciprocal_rank(pred_lists, gold_sets),
        "map": mean_average_precision(pred_lists, gold_sets),
    }

    # Best-of-K oracle EM / F1 over the answerable subset
    em, f1 = topk_best_em_f1(
        [[c.text for c in p.candidates] for p in answerable_preds],
        [ex.answers for ex in answerable_examples],
    )
    metrics["best_of_k_em"] = em
    metrics["best_of_k_f1"] = f1

    # Top-1 EM / F1 over the full sample (including no-answers)
    top1_em, top1_f1 = [], []
    for ex, pred in zip(examples, predictions):
        if ex.is_unanswerable:
            # SQuAD v2 rule: correct iff predicted no-answer
            ok = int(pred.no_answer)
            top1_em.append(ok)
            top1_f1.append(float(ok))
        else:
            pred_text = "" if pred.no_answer or not pred.candidates else pred.candidates[0].text
            em_s = max(exact_match(pred_text, g) for g in ex.answers) if ex.answers else 0
            f1_s = max(f1_score(pred_text, g) for g in ex.answers) if ex.answers else 0.0
            top1_em.append(em_s)
            top1_f1.append(f1_s)
    metrics["top1_em"] = sum(top1_em) / len(top1_em)
    metrics["top1_f1"] = sum(top1_f1) / len(top1_f1)

    # No-answer classification
    gold_no_ans = [ex.is_unanswerable for ex in examples]
    pred_no_ans = [p.no_answer for p in predictions]
    metrics.update(no_answer_classification(gold_no_ans, pred_no_ans))

    return metrics


# ---------------------------------------------------------------------------
# RAG retrieval evaluation
# ---------------------------------------------------------------------------
def evaluate_retrieval(examples, rag_results, qid_to_gold_doc, max_k: int = 10) -> dict:
    pred_lists = [r.retrieved_doc_ids for r in rag_results]
    gold_sets = [[qid_to_gold_doc[ex.qid]] for ex in examples]

    curve = recall_at_k_curve(pred_lists, gold_sets, max_k=max_k)
    return {
        "recall_at_k": {k: v for k, v in zip(range(1, max_k + 1), curve)},
        "mrr": mean_reciprocal_rank(pred_lists, gold_sets),
        "map": mean_average_precision(pred_lists, gold_sets),
    }


# ---------------------------------------------------------------------------
# RAG answer evaluation
# ---------------------------------------------------------------------------
def evaluate_rag_answers(examples, rag_results) -> dict:
    em_scores, f1_scores = [], []
    for ex, r in zip(examples, rag_results):
        if ex.is_unanswerable:
            ok = int(r.no_answer)
            em_scores.append(ok)
            f1_scores.append(float(ok))
        else:
            if r.no_answer:
                em_scores.append(0)
                f1_scores.append(0.0)
                continue
            em_s = max(exact_match(r.answer, g) for g in ex.answers) if ex.answers else 0
            f1_s = max(f1_score(r.answer, g) for g in ex.answers) if ex.answers else 0.0
            em_scores.append(em_s)
            f1_scores.append(f1_s)

    gold_no_ans = [ex.is_unanswerable for ex in examples]
    pred_no_ans = [r.no_answer for r in rag_results]
    metrics = {
        "em": sum(em_scores) / len(em_scores),
        "f1": sum(f1_scores) / len(f1_scores),
    }
    metrics.update(no_answer_classification(gold_no_ans, pred_no_ans))
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)


def predictions_to_dataframe(examples, ext_preds, rag_preds) -> pd.DataFrame:
    """Side-by-side table of predictions for qualitative inspection."""
    rows = []
    for ex, ep, rp in zip(examples, ext_preds, rag_preds):
        rows.append(
            {
                "qid": ex.qid,
                "question": ex.question,
                "gold_answers": " | ".join(ex.answers) if ex.answers else "<unanswerable>",
                "is_unanswerable": ex.is_unanswerable,
                "extractive_top1": (ep.candidates[0].text if ep.candidates and not ep.no_answer else "<no-answer>"),
                "extractive_top1_score": (ep.candidates[0].score if ep.candidates and not ep.no_answer else 0.0),
                "extractive_no_answer": ep.no_answer,
                "rag_answer": rp.answer,
                "rag_no_answer": rp.no_answer,
                "rag_top1_doc": rp.retrieved_doc_ids[0] if rp.retrieved_doc_ids else "",
            }
        )
    return pd.DataFrame(rows)
