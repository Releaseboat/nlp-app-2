"""
Main entry point — runs the entire assignment end-to-end.

Usage:
    python run_pipeline.py                       # full run, 500 examples
    python run_pipeline.py --sample-size 100     # quick smoke test
    python run_pipeline.py --skip-rag            # extractive only
    python run_pipeline.py --skip-judge          # skip LLM-as-Judge

All intermediate outputs are written to outputs/ as JSON / CSV so the
notebook can pick them up without re-running the models.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils import set_seed
from src.data_loader import load_squad_v2, build_corpus
from src.extractive_qa import ExtractiveQA
from src.retriever import TFIDFRetriever, DenseRetriever
from src.rag_qa import RAGPipeline
from src.llm_judge import LLMJudge, aggregate_judge_scores
from src.evaluation import (
    evaluate_extractive,
    evaluate_retrieval,
    evaluate_rag_answers,
    save_json,
    predictions_to_dataframe,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-size", type=int, default=config.SAMPLE_SIZE)
    p.add_argument("--retriever", choices=["tfidf", "dense"], default="tfidf",
                   help="Primary retriever used by the RAG pipeline.")
    p.add_argument("--skip-extractive", action="store_true")
    p.add_argument("--skip-rag", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-subset", type=int, default=100,
                   help="Number of RAG answers to send to the LLM judge (slow).")
    return p.parse_args()


def plot_recall_at_k(curves: dict, save_path: Path) -> None:
    """Overlay Recall@K curves for the systems we evaluate."""
    plt.figure(figsize=(8, 5))
    for label, curve in curves.items():
        ks = list(range(1, len(curve) + 1))
        plt.plot(ks, curve, marker="o", label=label)
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall@K — Extractive vs RAG Retrieval")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(len(c) for c in curves.values()) + 1))
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(config.SEED)
    device = config.get_device()
    print(f"[setup] device={device}, sample_size={args.sample_size}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    t0 = time.time()
    examples = load_squad_v2(
        config.DATASET_NAME,
        config.DATASET_SPLIT,
        sample_size=args.sample_size,
        answerable_ratio=config.ANSWERABLE_RATIO,
        seed=config.SEED,
    )
    doc_ids, doc_texts, qid_to_gold_doc = build_corpus(examples)
    print(f"[data] loaded {len(examples)} questions, {len(doc_ids)} unique docs "
          f"({time.time()-t0:.1f}s)")
    save_json([ex.to_dict() for ex in examples], config.OUTPUT_DIR / "sample.json")

    summary: dict = {"sample_size": len(examples), "n_docs": len(doc_ids)}
    recall_curves: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # Pipeline A — Extractive QA
    # ------------------------------------------------------------------
    if not args.skip_extractive:
        print("\n[Pipeline A] Loading extractive QA model…")
        ext = ExtractiveQA(
            model_name=config.EXTRACTIVE_MODEL,
            device=device,
            max_answer_len=config.EXTRACTIVE_MAX_ANSWER_LEN,
            top_k=config.EXTRACTIVE_TOP_K,
            null_threshold=config.EXTRACTIVE_NULL_THRESHOLD,
        )
        ext_preds = ext.predict_batch(examples)
        save_json([p.to_dict() for p in ext_preds],
                  config.OUTPUT_DIR / "extractive_predictions.json")

        ext_metrics = evaluate_extractive(examples, ext_preds,
                                          max_k=config.RETRIEVER_MAX_K_EVAL)
        save_json(ext_metrics, config.OUTPUT_DIR / "extractive_metrics.json")
        summary["extractive"] = ext_metrics
        recall_curves["Extractive Top-K answers"] = list(
            ext_metrics["recall_at_k"].values()
        )
        print(f"[Pipeline A] Top-1 EM={ext_metrics['top1_em']:.3f}  "
              f"F1={ext_metrics['top1_f1']:.3f}  MRR={ext_metrics['mrr']:.3f}  "
              f"MAP={ext_metrics['map']:.3f}")
    else:
        ext_preds = None

    # ------------------------------------------------------------------
    # Pipeline B — RAG
    # ------------------------------------------------------------------
    if not args.skip_rag:
        print(f"\n[Pipeline B] Fitting {args.retriever} retriever…")
        if args.retriever == "tfidf":
            retriever = TFIDFRetriever().fit(doc_ids, doc_texts)
        else:
            retriever = DenseRetriever(device=device).fit(doc_ids, doc_texts)

        print(f"[Pipeline B] Loading generator {config.GENERATIVE_MODEL}…")
        rag = RAGPipeline(
            retriever=retriever,
            generator_model_name=config.GENERATIVE_MODEL,
            device=device,
            top_k=config.RETRIEVER_TOP_K,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS,
            num_beams=config.GEN_NUM_BEAMS,
        )
        rag_preds = rag.predict_batch(examples)
        save_json([r.to_dict() for r in rag_preds],
                  config.OUTPUT_DIR / "rag_predictions.json")

        retrieval_metrics = evaluate_retrieval(examples, rag_preds,
                                               qid_to_gold_doc,
                                               max_k=config.RETRIEVER_MAX_K_EVAL)
        answer_metrics = evaluate_rag_answers(examples, rag_preds)
        save_json({"retrieval": retrieval_metrics, "answers": answer_metrics},
                  config.OUTPUT_DIR / "rag_metrics.json")

        summary["rag"] = {"retrieval": retrieval_metrics, "answers": answer_metrics}
        recall_curves[f"RAG retrieval ({args.retriever.upper()})"] = list(
            retrieval_metrics["recall_at_k"].values()
        )

        print(f"[Pipeline B] Retrieval MRR={retrieval_metrics['mrr']:.3f}  "
              f"MAP={retrieval_metrics['map']:.3f}  "
              f"Recall@1={retrieval_metrics['recall_at_k'][1]:.3f}  "
              f"Recall@5={retrieval_metrics['recall_at_k'][5]:.3f}")
        print(f"[Pipeline B] Answer EM={answer_metrics['em']:.3f}  "
              f"F1={answer_metrics['f1']:.3f}")
    else:
        rag_preds = None

    # ------------------------------------------------------------------
    # Recall@K plot
    # ------------------------------------------------------------------
    if recall_curves:
        plot_path = config.FIGURES_DIR / "recall_at_k.png"
        plot_recall_at_k(recall_curves, plot_path)
        print(f"[plot] saved {plot_path}")

    # ------------------------------------------------------------------
    # Side-by-side prediction table
    # ------------------------------------------------------------------
    if ext_preds is not None and rag_preds is not None:
        df = predictions_to_dataframe(examples, ext_preds, rag_preds)
        df.to_csv(config.OUTPUT_DIR / "predictions_side_by_side.csv", index=False)
        print(f"[table] wrote {len(df)} side-by-side predictions")

    # ------------------------------------------------------------------
    # LLM-as-Judge on a RAG subset
    # ------------------------------------------------------------------
    if not args.skip_judge and rag_preds is not None:
        print(f"\n[Judge] Loading judge model {config.JUDGE_MODEL}…")
        judge = LLMJudge(model_name=config.JUDGE_MODEL, device=device,
                         max_new_tokens=config.JUDGE_MAX_NEW_TOKENS)

        # Build records: pair each RAG answer with its retrieved (not gold!)
        # context, which is what was actually shown to the generator.
        records = []
        for ex, r in zip(examples, rag_preds):
            ctx = "\n\n".join(
                doc_texts[doc_ids.index(d)] for d in r.retrieved_doc_ids
                if d in doc_ids
            )
            records.append({
                "qid": r.qid,
                "question": r.question,
                "answer": r.answer,
                "context": ctx,
                "no_answer": r.no_answer,
            })
        # Subset to keep judging tractable
        records = records[: args.judge_subset]
        scores = judge.score_batch(records, skip_no_answer=True)
        save_json([s.to_dict() for s in scores],
                  config.OUTPUT_DIR / "judge_scores.json")

        agg = aggregate_judge_scores(scores)
        save_json(agg, config.OUTPUT_DIR / "judge_summary.json")
        summary["judge"] = agg
        print(f"[Judge] faithfulness(LLM)={agg['faithfulness_llm_mean']:.2f}/5  "
              f"relevance(LLM)={agg['relevance_llm_mean']:.2f}/5  "
              f"(n={agg['n_judged']})")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    save_json(summary, config.OUTPUT_DIR / "summary.json")
    print("\n[done] wrote outputs/summary.json")


if __name__ == "__main__":
    main()
