"""
Pipeline B — Retrieval-Augmented Generation QA.

Three-stage pipeline:
    1. Retrieve top-K context paragraphs (delegated to retriever.py)
    2. Inject them into a prompt that *explicitly* permits a refusal
    3. Generate the answer with a seq2seq LLM (default flan-t5-base)

The refusal sentinel ("unanswerable") is post-processed into a boolean
no-answer flag mirroring the extractive pipeline so the two are directly
comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

from src.retriever import RetrievalResult


# The exact sentinel string the model is instructed to emit when the
# retrieved context does not support an answer.  We compare answers
# against this string (case-insensitive) to set the no_answer flag.
NO_ANSWER_SENTINEL = "unanswerable"


PROMPT_TEMPLATE = """You are a careful question-answering assistant.
Use ONLY the information in the context below.  If the answer cannot be
found in the context, reply with exactly the single word: {sentinel}.
Do not invent facts that are not stated in the context.

Context:
{context}

Question: {question}

Answer:"""


@dataclass
class RAGResult:
    qid: str
    question: str
    answer: str
    no_answer: bool
    retrieved_doc_ids: List[str] = field(default_factory=list)
    retrieved_scores: List[float] = field(default_factory=list)
    prompt: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class RAGPipeline:
    """End-to-end RAG QA with explicit unanswerable handling."""

    def __init__(
        self,
        retriever,
        generator_model_name: str,
        device: str = "cpu",
        top_k: int = 5,
        max_new_tokens: int = 64,
        num_beams: int = 4,
        max_context_chars: int = 3500,
    ) -> None:
        self.retriever = retriever
        self.device = device
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.max_context_chars = max_context_chars

        self.tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.model.to(device).eval()

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str, retrieval: RetrievalResult) -> str:
        # concatenate retrieved docs with separators, hard-cap total chars
        joined = "\n\n---\n\n".join(
            f"[Doc {i+1}] {txt}" for i, txt in enumerate(retrieval.doc_texts)
        )
        if len(joined) > self.max_context_chars:
            joined = joined[: self.max_context_chars]
        return PROMPT_TEMPLATE.format(
            sentinel=NO_ANSWER_SENTINEL,
            context=joined,
            question=question.strip(),
        )

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
            do_sample=False,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    def predict(self, qid: str, question: str) -> RAGResult:
        retrieval = self.retriever.rank(question, self.top_k)
        prompt = self._build_prompt(question, retrieval)
        answer = self._generate(prompt)
        no_ans = answer.strip().lower().startswith(NO_ANSWER_SENTINEL)
        return RAGResult(
            qid=qid,
            question=question,
            answer=answer,
            no_answer=no_ans,
            retrieved_doc_ids=list(retrieval.doc_ids),
            retrieved_scores=list(retrieval.scores),
            prompt=prompt,
        )

    def predict_batch(self, examples) -> List[RAGResult]:
        # Pre-retrieve in batch (faster for the dense retriever).
        questions = [ex.question for ex in examples]
        retrievals = self.retriever.rank_batch(questions, self.top_k)

        results: List[RAGResult] = []
        for ex, retrieval in tqdm(
            list(zip(examples, retrievals)), desc="RAG generation"
        ):
            prompt = self._build_prompt(ex.question, retrieval)
            answer = self._generate(prompt)
            no_ans = answer.strip().lower().startswith(NO_ANSWER_SENTINEL)
            results.append(
                RAGResult(
                    qid=ex.qid,
                    question=ex.question,
                    answer=answer,
                    no_answer=no_ans,
                    retrieved_doc_ids=list(retrieval.doc_ids),
                    retrieved_scores=list(retrieval.scores),
                    prompt=prompt,
                )
            )
        return results
