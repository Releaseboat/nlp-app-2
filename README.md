# SQuAD v2.0 — Extractive vs RAG Question Answering

End-to-end implementation and evaluation of two QA systems on the Stanford
Question Answering Dataset v2.0:

- **Pipeline A — Extractive QA** using `deepset/roberta-base-squad2`, with a
  custom Top-K span decoder and calibrated no-answer handling.
- **Pipeline B — Retrieval-Augmented Generation** using a TF-IDF (and
  optional dense MiniLM) retriever feeding a `flan-t5-base` generator that is
  explicitly instructed to refuse when context is insufficient.

Both pipelines are evaluated with **from-scratch implementations** of
Recall@K, MRR and MAP (unit-tested against textbook examples), and the
generative outputs are additionally scored on **Faithfulness** and **Answer
Relevance** by an LLM-as-Judge supplemented with deterministic heuristics.

## Repository layout

```
.
├── README.md                        ← you are here
├── report/REPORT.md                 ← academic write-up (start here)
├── notebooks/QA_Analysis.ipynb      ← runner notebook with plots + analysis
├── notebooks/_build_notebook.py     ← regenerates the notebook from source
├── run_pipeline.py                  ← main entry point
├── config.py                        ← all hyper-parameters
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_loader.py               ← SQuAD v2 loader (HF Hub + JSON fallback)
│   ├── extractive_qa.py             ← Pipeline A
│   ├── retriever.py                 ← TF-IDF + dense retrievers
│   ├── rag_qa.py                    ← Pipeline B
│   ├── llm_judge.py                 ← Faithfulness + relevance judging
│   ├── metrics.py                   ← Recall@K / MRR / MAP from scratch
│   ├── evaluation.py                ← orchestration of metric computation
│   └── utils.py                     ← normalisation, EM, F1, seeding
├── tests/test_smoke.py              ← offline unit tests (no model downloads)
└── outputs/                         ← populated by run_pipeline.py
    ├── sample.json
    ├── extractive_predictions.json
    ├── extractive_metrics.json
    ├── rag_predictions.json
    ├── rag_metrics.json
    ├── judge_scores.json
    ├── judge_summary.json
    ├── predictions_side_by_side.csv
    ├── summary.json
    └── figures/recall_at_k.png
```

## Quick start

```bash
# 1.  Create a venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2.  Run the offline smoke tests (no model downloads required)
python -m tests.test_smoke

# 3.  Run the full pipeline (downloads models on first run, ~1.5 GB total)
python run_pipeline.py                     # 500 examples, both pipelines
# or smaller for iteration:
python run_pipeline.py --sample-size 100 --judge-subset 30

# 4.  Open the notebook to see the analysis with plots and tables
jupyter lab notebooks/QA_Analysis.ipynb
```

`run_pipeline.py` writes everything the notebook needs to `outputs/`. The
notebook is read-only on those files — re-running the notebook does not
re-run the models.

## CLI flags

```
--sample-size INT         number of SQuAD v2 dev questions to evaluate
                          (default: 500, stratified 50/50 answerable / unanswerable)
--retriever {tfidf,dense} primary retriever for the RAG pipeline
                          (default: tfidf; dense uses sentence-transformers/all-MiniLM-L6-v2)
--skip-extractive         only run the RAG pipeline
--skip-rag                only run the extractive pipeline
--skip-judge              skip the LLM-as-Judge step (saves 5-10 min)
--judge-subset INT        number of RAG answers to send to the judge
                          (default: 100)
```

## Reproducibility

- All randomness is seeded via `config.SEED = 42` propagated through
  `src.utils.set_seed`.
- Beam search uses `do_sample=False` and `num_beams=4`.
- The sampled question list is fully determined by the seed.
- Re-running on the same machine + checkpoint yields byte-identical
  outputs.

## Network considerations

The pipeline pulls three model checkpoints and one dataset from HuggingFace
Hub on first run (~1.5 GB total):

| Resource                                    | Size    |
|---------------------------------------------|---------|
| `deepset/roberta-base-squad2`               | ~500 MB |
| `google/flan-t5-base`                       | ~990 MB |
| `sentence-transformers/all-MiniLM-L6-v2`    | ~80 MB  |
| `rajpurkar/squad_v2` (validation split)     | ~5 MB   |

### Corporate networks

Two common corporate-network issues and their work-arounds:

1. **TLS interception** — the company injects its own root CA. Symptom:
   `SSLCertVerificationError`. Fix:

   ```bash
   # Build a combined bundle of system + keychain CAs (macOS example)
   {
     cat /etc/ssl/cert.pem
     security find-certificate -a -p /Library/Keychains/System.keychain
     security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain
   } > .cache/corp-cabundle.pem

   export SSL_CERT_FILE=$(pwd)/.cache/corp-cabundle.pem
   export REQUESTS_CA_BUNDLE=$(pwd)/.cache/corp-cabundle.pem
   ```

2. **Xet CDN block** — HuggingFace serves large model files via
   `cas-bridge.xethub.hf.co`, which some corporate proxies block at the
   domain level. Symptom: `403 Forbidden` on model download. There is no
   client-side fix beyond moving to a non-blocked network or downloading
   the model checkpoint manually on a different machine and pointing
   `HF_HOME` at it.

The data loader (`src/data_loader.py`) has a built-in fallback that
downloads the canonical `dev-v2.0.json` directly from the SQuAD project
page (`rajpurkar.github.io`) when the Hub is unreachable, so the
**dataset** side works on any network with general internet access. The
**models**, unfortunately, only ship through HF.

## What was implemented from scratch

Per the assignment brief, the following are implemented directly rather
than imported from a library:

- `recall_at_k`, `recall_at_k_curve`
- `mean_reciprocal_rank` (and `reciprocal_rank`)
- `average_precision` (with the canonical IR de-duplication of relevant
  items), `mean_average_precision`
- SQuAD-style answer normalisation, EM, token-level F1
- `no_answer_classification` (TP/FP/TN/FN -> precision/recall/F1/accuracy)
- The extractive span decoder (rather than relying on
  `pipeline('question-answering')`'s opaque internals)

All are unit-tested in `tests/test_smoke.py` against worked examples from
Manning, Raghavan & Schütze (2008).

## Report

The academic write-up — methodology, results tables, diminishing-returns
analysis, MRR-vs-Recall@1 discussion, hallucination vs extractive-error
analysis, suggested improvements, references — is in
**[`report/REPORT.md`](report/REPORT.md)**.
