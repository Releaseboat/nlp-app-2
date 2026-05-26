"""
LLM-as-Judge evaluation for the generative pipeline.

Two qualities are scored on a 1-5 ordinal scale:

    Faithfulness (Groundedness)
        Is every claim in the answer supported by the supplied context?
        A faithful but irrelevant answer should still get a high
        faithfulness score; we measure faithfulness independently of
        relevance.

    Answer Relevance
        Does the answer actually address what the question asks?
        Penalises irrelevant, redundant or hallucinated content even when
        the surface form looks plausible.

We use the same small instruction-tuned model (flan-t5-base) as both
generator and judge.  Self-judging is biased upward — the model tends to
forgive its own outputs — so we additionally compute deterministic
heuristic baselines (token-overlap groundedness and question-answer
lexical relevance) and report both in the final analysis.  Hybrid
scoring is the most academically defensible choice when the same model
plays both roles.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from src.utils import normalize_answer


FAITHFULNESS_PROMPT = """You are an expert evaluator.
Read the CONTEXT and the ANSWER below.  Decide how well every claim in
the ANSWER is directly supported by the CONTEXT.  A score of 5 means
fully supported; 1 means contradicted or unsupported (hallucinated).

CONTEXT:
{context}

ANSWER:
{answer}

Reply with a single integer from 1 to 5.  Score:"""


RELEVANCE_PROMPT = """You are an expert evaluator.
Read the QUESTION and the ANSWER below.  Decide how well the ANSWER
directly addresses the QUESTION.  A score of 5 means the ANSWER answers
exactly what is asked; 1 means it is irrelevant, redundant, or off-topic.

QUESTION:
{question}

ANSWER:
{answer}

Reply with a single integer from 1 to 5.  Score:"""


@dataclass
class JudgeScore:
    qid: str
    faithfulness_llm: Optional[int]      # 1-5 from the LLM judge
    relevance_llm: Optional[int]         # 1-5 from the LLM judge
    faithfulness_heuristic: float        # [0,1] token overlap with context
    relevance_heuristic: float           # [0,1] token overlap with question

    def to_dict(self) -> dict:
        return asdict(self)


class LLMJudge:
    """Self-judge wrapper around flan-t5 with a deterministic fallback."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 8,
        max_context_chars: int = 3000,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device
        self.model.to(device).eval()
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ask(self, prompt: str) -> str:
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=1,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    @staticmethod
    def _parse_score(text: str) -> Optional[int]:
        m = re.search(r"[1-5]", text)
        return int(m.group(0)) if m else None

    # ------------------------------------------------------------------
    # Heuristic backstops
    # ------------------------------------------------------------------
    @staticmethod
    def token_overlap(a: str, b: str) -> float:
        """Fraction of unique tokens in `a` that also appear in `b`."""
        a_tokens = set(normalize_answer(a).split())
        b_tokens = set(normalize_answer(b).split())
        if not a_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens)

    # ------------------------------------------------------------------
    def score(
        self, qid: str, question: str, answer: str, context: str
    ) -> JudgeScore:
        # truncate to keep within model context window
        ctx = context[: self.max_context_chars]

        faith_text = self._ask(
            FAITHFULNESS_PROMPT.format(context=ctx, answer=answer)
        )
        rel_text = self._ask(
            RELEVANCE_PROMPT.format(question=question, answer=answer)
        )

        return JudgeScore(
            qid=qid,
            faithfulness_llm=self._parse_score(faith_text),
            relevance_llm=self._parse_score(rel_text),
            faithfulness_heuristic=self.token_overlap(answer, context),
            relevance_heuristic=self.token_overlap(answer, question),
        )

    def score_batch(
        self,
        records: List[dict],
        skip_no_answer: bool = True,
    ) -> List[JudgeScore]:
        """
        Score a list of records of the form
            {"qid", "question", "answer", "context", "no_answer"}.

        If `skip_no_answer` is True, we don't bother judging answers the
        system flagged as unanswerable — their content is the sentinel
        string, not a real answer.  They get None LLM scores and 0.0
        heuristic scores so they can still be aggregated downstream.
        """
        results: List[JudgeScore] = []
        for r in tqdm(records, desc="LLM Judge"):
            if skip_no_answer and r.get("no_answer", False):
                results.append(
                    JudgeScore(
                        qid=r["qid"],
                        faithfulness_llm=None,
                        relevance_llm=None,
                        faithfulness_heuristic=0.0,
                        relevance_heuristic=0.0,
                    )
                )
                continue
            results.append(
                self.score(r["qid"], r["question"], r["answer"], r["context"])
            )
        return results


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_judge_scores(scores: List[JudgeScore]) -> dict:
    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0

    return {
        "faithfulness_llm_mean": _mean([s.faithfulness_llm for s in scores]),
        "relevance_llm_mean": _mean([s.relevance_llm for s in scores]),
        "faithfulness_heuristic_mean": _mean([s.faithfulness_heuristic for s in scores]),
        "relevance_heuristic_mean": _mean([s.relevance_heuristic for s in scores]),
        "n_judged": sum(1 for s in scores if s.faithfulness_llm is not None),
        "n_total": len(scores),
    }
