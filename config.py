"""
Central configuration for the QA assignment.
All hyper-parameters, model identifiers, and paths live here so the rest of
the codebase remains free of magic numbers.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = ROOT_DIR / ".cache"

for d in (OUTPUT_DIR, FIGURES_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
DATASET_NAME = "rajpurkar/squad_v2"          # SQuAD v2.0 on the Hub
DATASET_SPLIT = "validation"                 # we evaluate on the dev set
SAMPLE_SIZE = 500                            # examples used for evaluation
ANSWERABLE_RATIO = 0.5                       # fraction of answerable Qs in sample

# ---------------------------------------------------------------------------
# Extractive QA (Pipeline A)
# ---------------------------------------------------------------------------
EXTRACTIVE_MODEL = "deepset/roberta-base-squad2"
EXTRACTIVE_TOP_K = 10                        # candidate spans returned per Q
EXTRACTIVE_MAX_ANSWER_LEN = 30               # tokens
EXTRACTIVE_NULL_THRESHOLD = 0.0              # tune for no-answer trade-off

# ---------------------------------------------------------------------------
# Retriever (Pipeline B - RAG)
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = 5                          # contexts injected into the LLM
RETRIEVER_MAX_K_EVAL = 10                    # max K used in Recall@K curve

# ---------------------------------------------------------------------------
# Generator (Pipeline B - RAG)
# ---------------------------------------------------------------------------
GENERATIVE_MODEL = "google/flan-t5-base"
GEN_MAX_NEW_TOKENS = 64
GEN_NUM_BEAMS = 4

# ---------------------------------------------------------------------------
# LLM-as-Judge
# ---------------------------------------------------------------------------
JUDGE_MODEL = GENERATIVE_MODEL               # self-judge (same model)
JUDGE_MAX_NEW_TOKENS = 8

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Pick the best available accelerator (MPS > CUDA > CPU)."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
