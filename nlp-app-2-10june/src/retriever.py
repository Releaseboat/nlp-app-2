"""
Retrievers used by the RAG pipeline.

Two retrievers are implemented and benchmarked against each other in the
notebook:

    1. TFIDFRetriever  — sparse, deterministic, instantaneous to fit.
       This is the classical IR baseline and demonstrates how far a
       carefully tokenised BoW model can go.
    2. DenseRetriever  — sentence-transformers MiniLM cosine retrieval.
       Captures paraphrastic and semantic matches the sparse model misses.

Both expose the same `rank(query, k)` API so the rest of the pipeline is
agnostic to which one is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    doc_ids: List[str]
    doc_texts: List[str]
    scores: List[float]


class TFIDFRetriever:
    """Sparse TF-IDF cosine retriever."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_matrix = None

    def fit(self, doc_ids: List[str], doc_texts: List[str]) -> "TFIDFRetriever":
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.doc_matrix = self.vectorizer.fit_transform(doc_texts)
        return self

    def rank(self, query: str, k: int) -> RetrievalResult:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix)[0]
        order = np.argsort(sims)[::-1][:k]
        return RetrievalResult(
            doc_ids=[self.doc_ids[i] for i in order],
            doc_texts=[self.doc_texts[i] for i in order],
            scores=[float(sims[i]) for i in order],
        )

    def rank_batch(self, queries: List[str], k: int) -> List[RetrievalResult]:
        q_vecs = self.vectorizer.transform(queries)
        sims = cosine_similarity(q_vecs, self.doc_matrix)
        results = []
        for row in sims:
            order = np.argsort(row)[::-1][:k]
            results.append(
                RetrievalResult(
                    doc_ids=[self.doc_ids[i] for i in order],
                    doc_texts=[self.doc_texts[i] for i in order],
                    scores=[float(row[i]) for i in order],
                )
            )
        return results


class DenseRetriever:
    """
    Dense semantic retriever using sentence-transformers MiniLM.

    We pre-encode the corpus once, then cosine-rank query embeddings.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.doc_emb: np.ndarray | None = None

    def fit(self, doc_ids: List[str], doc_texts: List[str]) -> "DenseRetriever":
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(doc_texts)
        self.doc_emb = self.model.encode(
            doc_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        return self

    def rank(self, query: str, k: int) -> RetrievalResult:
        return self.rank_batch([query], k)[0]

    def rank_batch(self, queries: List[str], k: int) -> List[RetrievalResult]:
        q_emb = self.model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        sims = q_emb @ self.doc_emb.T   # both normalised -> cosine
        results = []
        for row in sims:
            order = np.argsort(row)[::-1][:k]
            results.append(
                RetrievalResult(
                    doc_ids=[self.doc_ids[i] for i in order],
                    doc_texts=[self.doc_texts[i] for i in order],
                    scores=[float(row[i]) for i in order],
                )
            )
        return results
