"""
Pipeline A — Extractive QA.

Wraps a BERT/RoBERTa span-prediction model (default
`deepset/roberta-base-squad2`) and exposes a clean API that returns the
top-K candidate spans together with calibrated confidence scores and an
explicit "no answer" flag for SQuAD v2 unanswerable questions.

Why a custom wrapper instead of the bare `pipeline("question-answering")`?
The HF pipeline returns top-K but its scoring conflates the start/end
logits in a way that makes calibration hard.  We re-implement the standard
span-enumeration algorithm so the null-vs-best-non-null comparison used to
flag unanswerable questions is transparent and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from tqdm import tqdm


@dataclass
class SpanPrediction:
    """A single candidate answer span."""
    text: str
    score: float                 # softmax probability of (start, end) pair
    start_char: int
    end_char: int
    start_logit: float
    end_logit: float


@dataclass
class ExtractiveResult:
    """The full output for one question."""
    qid: str
    question: str
    candidates: List[SpanPrediction] = field(default_factory=list)
    null_score: float = 0.0       # score of the [CLS] / null span
    best_non_null_score: float = 0.0
    no_answer: bool = False       # True if model predicts SQuAD v2 unanswerable
    no_answer_margin: float = 0.0 # null_score - best_non_null_score

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class ExtractiveQA:
    """Span-prediction QA with calibrated Top-K and no-answer detection."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_answer_len: int = 30,
        top_k: int = 10,
        null_threshold: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_answer_len = max_answer_len
        self.top_k = top_k
        self.null_threshold = null_threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.model.to(device).eval()

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, question: str, context: str):
        enc = self.tokenizer(
            question,
            context,
            return_tensors="pt",
            truncation="only_second",
            max_length=384,
            stride=128,
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        sequence_ids = enc.sequence_ids(0)
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**inputs)
        start_logits = out.start_logits[0].detach().cpu().numpy()
        end_logits = out.end_logits[0].detach().cpu().numpy()
        return start_logits, end_logits, offsets, sequence_ids

    def _enumerate_spans(
        self,
        start_logits: np.ndarray,
        end_logits: np.ndarray,
        offsets: list,
        sequence_ids: list,
        context: str,
    ) -> tuple[list, float]:
        """
        Return (candidates, null_score).

        Candidates is a list of dicts with raw logits; null_score is the
        sum of start/end logits at position 0 ([CLS]), interpreted as the
        model's preference for "no answer".
        """
        n_best = 20  # consider top-20 start/end indices each
        start_idx = np.argsort(start_logits)[::-1][:n_best]
        end_idx = np.argsort(end_logits)[::-1][:n_best]

        candidates: list[dict] = []
        for s in start_idx:
            for e in end_idx:
                # span must lie inside the context (sequence_id == 1)
                if sequence_ids[s] != 1 or sequence_ids[e] != 1:
                    continue
                if e < s or (e - s + 1) > self.max_answer_len:
                    continue
                start_char, _ = offsets[s]
                _, end_char = offsets[e]
                if end_char <= start_char:
                    continue
                candidates.append(
                    dict(
                        start_char=int(start_char),
                        end_char=int(end_char),
                        start_logit=float(start_logits[s]),
                        end_logit=float(end_logits[e]),
                        text=context[start_char:end_char],
                    )
                )

        null_score = float(start_logits[0] + end_logits[0])
        return candidates, null_score

    @staticmethod
    def _softmax_calibrate(candidates: list, null_score: float) -> list:
        """
        Turn raw logit sums into probabilities via a single softmax over
        all candidate spans + the null span.  This gives directly
        comparable confidence numbers.
        """
        logits = np.array([c["start_logit"] + c["end_logit"] for c in candidates] + [null_score])
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        for c, p in zip(candidates, probs[:-1]):
            c["score"] = float(p)
        return candidates, float(probs[-1])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, qid: str, question: str, context: str) -> ExtractiveResult:
        start_logits, end_logits, offsets, seq_ids = self._forward(question, context)
        cands, null_logit = self._enumerate_spans(
            start_logits, end_logits, offsets, seq_ids, context
        )

        if not cands:
            return ExtractiveResult(
                qid=qid, question=question,
                candidates=[],
                null_score=1.0,
                best_non_null_score=0.0,
                no_answer=True,
                no_answer_margin=1.0,
            )

        cands, null_prob = self._softmax_calibrate(cands, null_logit)
        cands.sort(key=lambda c: c["score"], reverse=True)

        # de-duplicate identical answer strings, keep the highest-scoring
        seen, dedup = set(), []
        for c in cands:
            key = c["text"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(c)
            if len(dedup) >= self.top_k:
                break

        best = dedup[0]["score"]
        no_ans = null_prob > best + self.null_threshold

        return ExtractiveResult(
            qid=qid,
            question=question,
            candidates=[SpanPrediction(**c) for c in dedup],
            null_score=null_prob,
            best_non_null_score=best,
            no_answer=bool(no_ans),
            no_answer_margin=null_prob - best,
        )

    def predict_batch(self, examples) -> List[ExtractiveResult]:
        """Iterate predict() over a list of QAExample objects with a progress bar."""
        results: List[ExtractiveResult] = []
        for ex in tqdm(examples, desc="Extractive QA"):
            results.append(self.predict(ex.qid, ex.question, ex.context))
        return results
