"""
Offline smoke tests — exercise the whole evaluation flow on synthetic
data so the structure can be validated without downloading model
weights.  Two purposes:

    1. CI-style sanity check: run `python -m tests.test_smoke` to confirm
       data loading, metric computation, plotting and serialisation all
       work on this machine.
    2. Reference: the synthetic examples illustrate the exact shapes the
       real pipelines emit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import config
from src.utils import set_seed, normalize_answer, exact_match, f1_score
from src.metrics import (
    recall_at_k,
    recall_at_k_curve,
    mean_reciprocal_rank,
    mean_average_precision,
    average_precision,
    no_answer_classification,
    topk_best_em_f1,
)


def test_metrics() -> None:
    # canonical IR example (Manning et al., 2008)
    preds = ["a", "b", "c", "d", "e"]
    rels = {"b", "d"}
    ap = average_precision(preds, rels)
    assert math.isclose(ap, (1 / 2 + 2 / 4) / 2)

    # batch versions
    p_batch = [["a", "b", "c"], ["x", "y", "z"]]
    r_batch = [{"b"}, {"z"}]
    assert recall_at_k(p_batch, r_batch, 1) == 0.0
    assert recall_at_k(p_batch, r_batch, 3) == 1.0
    assert math.isclose(
        mean_reciprocal_rank(p_batch, r_batch), (1 / 2 + 1 / 3) / 2
    )
    assert recall_at_k_curve(p_batch, r_batch, 3) == [0.0, 0.5, 1.0]

    # unanswerable queries are skipped from the denominator
    assert recall_at_k(
        [["a"], ["x"]], [set(), {"x"}], 1
    ) == 1.0
    print("[ok] metrics")


def test_utils() -> None:
    assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"
    assert exact_match("the cat", "a cat") == 1
    assert f1_score("the brown fox", "a quick brown fox") > 0.7
    print("[ok] utils")


def test_no_answer_classification() -> None:
    m = no_answer_classification(
        [True, True, False, False, True],
        [True, False, False, True, True],
    )
    assert m["tp"] == 2 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1
    assert math.isclose(
        m["no_answer_f1"], 2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3)
    )
    print("[ok] no_answer_classification")


def test_synthetic_extractive_eval() -> None:
    """End-to-end metric flow with synthetic ExtractiveResult-like dicts."""
    from dataclasses import dataclass
    from src.evaluation import evaluate_extractive

    @dataclass
    class _Span:
        text: str
        score: float

    @dataclass
    class _Res:
        qid: str
        question: str
        candidates: list
        null_score: float = 0.1
        best_non_null_score: float = 0.9
        no_answer: bool = False
        no_answer_margin: float = -0.8

        def to_dict(self):
            return {"qid": self.qid}

    @dataclass
    class _Ex:
        qid: str
        title: str
        question: str
        context: str
        answers: list
        is_unanswerable: bool

    examples = [
        _Ex("q1", "t", "Q1?", "ctx", ["Paris"], False),
        _Ex("q2", "t", "Q2?", "ctx", ["1969"], False),
        _Ex("q3", "t", "Q3?", "ctx", [], True),
    ]
    preds = [
        _Res("q1", "Q1?", [_Span("Paris", 0.9), _Span("Lyon", 0.05)]),
        _Res("q2", "Q2?", [_Span("1970", 0.4), _Span("1969", 0.3)]),
        _Res("q3", "Q3?", [_Span("Paris", 0.1)], no_answer=True),
    ]
    m = evaluate_extractive(examples, preds, max_k=3)
    # Recall@1: q1 hits, q2 misses (top1 is wrong) -> 1/2 over answerable
    assert math.isclose(m["recall_at_k"][1], 0.5)
    # Recall@2: both hit -> 1.0
    assert math.isclose(m["recall_at_k"][2], 1.0)
    # Best-of-K F1 on answerable subset
    assert m["best_of_k_em"] == 1.0
    # Top-1 EM: q1 right (1), q2 wrong (0), q3 right via no_answer (1)
    assert math.isclose(m["top1_em"], 2 / 3)
    print("[ok] evaluate_extractive on synthetic data")


def test_recall_plot() -> None:
    """Plot generation works headlessly."""
    import matplotlib
    matplotlib.use("Agg")
    from run_pipeline import plot_recall_at_k

    out = config.FIGURES_DIR / "smoke_recall.png"
    plot_recall_at_k(
        {
            "system A": [0.1, 0.3, 0.5, 0.7, 0.85, 0.9, 0.93, 0.95, 0.96, 0.97],
            "system B": [0.2, 0.4, 0.55, 0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88],
        },
        out,
    )
    assert out.exists() and out.stat().st_size > 0
    print(f"[ok] plot_recall_at_k -> {out}")


def test_data_loader_json() -> None:
    """JSON fallback works when SQuAD v2 dev file is cached."""
    cache_path = Path(".cache/squad_v2_dev.json")
    if not cache_path.exists():
        print("[skip] data_loader (cache file not present)")
        return

    from src.data_loader import load_squad_v2, build_corpus
    set_seed(123)
    examples = load_squad_v2(
        "rajpurkar/squad_v2",
        "validation",
        sample_size=20,
        answerable_ratio=0.5,
        seed=123,
        source="json",
    )
    assert len(examples) == 20
    n_ans = sum(1 for e in examples if not e.is_unanswerable)
    n_unans = sum(1 for e in examples if e.is_unanswerable)
    assert n_ans == 10 and n_unans == 10
    docs, _, qid2gold = build_corpus(examples)
    assert all(qid in qid2gold for qid in (e.qid for e in examples))
    print(f"[ok] data_loader (20 examples, {len(docs)} unique docs)")


def main() -> None:
    print("running smoke tests...\n")
    test_metrics()
    test_utils()
    test_no_answer_classification()
    test_synthetic_extractive_eval()
    test_recall_plot()
    test_data_loader_json()
    print("\nALL SMOKE TESTS PASS")


if __name__ == "__main__":
    main()
