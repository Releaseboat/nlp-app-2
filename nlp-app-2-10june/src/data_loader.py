"""
Load and sample the SQuAD v2.0 dev set.

Two loaders are supported:

    1. HuggingFace `datasets`  — preferred, returns the official split.
    2. Direct JSON download    — fallback for networks that block the
       HF dataset CDN (e.g. corporate proxies that drop xethub.hf.co).
       The file is the canonical `dev-v2.0.json` hosted on the SQuAD
       project page by the original authors.

Both produce identical QAExample objects so downstream code is oblivious
to which source was used.
"""
from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

from src.utils import is_unanswerable


SQUAD_V2_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


@dataclass
class QAExample:
    """A single SQuAD v2 question, normalised for our pipelines."""
    qid: str
    title: str
    question: str
    context: str
    answers: List[str]          # may be empty for unanswerable
    is_unanswerable: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _iter_from_huggingface(dataset_name: str, split: str):
    """Yield rows from the HF Hub dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    for row in ds:
        yield {
            "id": row["id"],
            "title": row.get("title", ""),
            "question": row["question"],
            "context": row["context"],
            "answers": list(row["answers"]["text"]),
        }


def _iter_from_json(json_path: Path):
    """Yield rows from the canonical Stanford dev-v2.0.json file."""
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for article in data["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                # SQuAD v2 marks unanswerable QAs with is_impossible=True
                # and an empty "answers" list (plausible_answers may be set
                # but those are *not* gold answers).
                if qa.get("is_impossible", False):
                    answers = []
                else:
                    answers = [a["text"] for a in qa["answers"]]
                yield {
                    "id": qa["id"],
                    "title": title,
                    "question": qa["question"],
                    "context": context,
                    "answers": answers,
                }


def _download_squad_v2(cache_dir: Path) -> Path:
    """Fetch dev-v2.0.json into the cache dir if not already present."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "squad_v2_dev.json"
    if target.exists():
        return target

    print(f"[data] downloading {SQUAD_V2_DEV_URL} -> {target}")
    # Respect SSL_CERT_FILE / REQUESTS_CA_BUNDLE for corporate networks
    import ssl
    ca_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    ctx = ssl.create_default_context(cafile=ca_file) if ca_file else None
    with urllib.request.urlopen(SQUAD_V2_DEV_URL, context=ctx) as resp, \
            open(target, "wb") as out:
        out.write(resp.read())
    return target


def load_squad_v2(
    dataset_name: str,
    split: str,
    sample_size: int,
    answerable_ratio: float,
    seed: int = 42,
    cache_dir: Path | None = None,
    source: str = "auto",
) -> List[QAExample]:
    """
    Return a stratified random sample of QAExample objects.

    `answerable_ratio` controls how many of the returned examples have at
    least one gold answer.  The rest are SQuAD v2 unanswerable items.

    `source` ∈ {"auto", "hf", "json"}:
        - "auto": try HuggingFace, fall back to JSON on failure.
        - "hf": force HuggingFace `datasets`.
        - "json": force the direct dev-v2.0.json download.
    """
    cache_dir = cache_dir or Path(".cache")

    iterator = None
    if source in ("auto", "hf"):
        try:
            iterator = _iter_from_huggingface(dataset_name, split)
            # Materialise lazily — we still want to catch network errors
            # during iteration, so wrap the generator.
            iterator = list(iterator)
        except Exception as e:
            if source == "hf":
                raise
            print(f"[data] HuggingFace load failed ({type(e).__name__}); "
                  f"falling back to direct JSON download")
            iterator = None
    if iterator is None:
        json_path = _download_squad_v2(cache_dir)
        iterator = list(_iter_from_json(json_path))

    answerable_pool: List[QAExample] = []
    unanswerable_pool: List[QAExample] = []

    for row in iterator:
        answers = list(row["answers"])
        ex = QAExample(
            qid=row["id"],
            title=row.get("title", ""),
            question=row["question"],
            context=row["context"],
            answers=answers,
            is_unanswerable=is_unanswerable(answers),
        )
        (unanswerable_pool if ex.is_unanswerable else answerable_pool).append(ex)

    rng = random.Random(seed)
    rng.shuffle(answerable_pool)
    rng.shuffle(unanswerable_pool)

    n_ans = int(round(sample_size * answerable_ratio))
    n_unans = sample_size - n_ans

    sample = answerable_pool[:n_ans] + unanswerable_pool[:n_unans]
    rng.shuffle(sample)
    return sample


def build_corpus(examples: List[QAExample]) -> tuple[List[str], List[str], dict]:
    """
    Build the retrieval corpus from the sampled examples.

    Every distinct context paragraph becomes a "document". We return:
        doc_ids:   stable IDs for each context (cN)
        doc_texts: parallel list of context strings
        qid_to_gold_doc: question id -> gold document id

    Multiple questions in SQuAD v2 share the same paragraph, so the
    corpus is typically much smaller than the question set.
    """
    seen: dict[str, str] = {}
    doc_ids: List[str] = []
    doc_texts: List[str] = []
    qid_to_gold_doc: dict[str, str] = {}

    for ex in examples:
        if ex.context not in seen:
            doc_id = f"c{len(doc_ids)}"
            seen[ex.context] = doc_id
            doc_ids.append(doc_id)
            doc_texts.append(ex.context)
        qid_to_gold_doc[ex.qid] = seen[ex.context]

    return doc_ids, doc_texts, qid_to_gold_doc
