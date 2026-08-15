"""
Retrieval configuration (Phase 2 / Phase 3)
---------------------------------------------
All retrieval settings are read from the root ``.env`` file at import time so
models, paths and top-k values are configurable and never hardcoded in the
pipeline.

See ``.env`` / ``.env.example`` under the ``# Retrieval configuration`` section.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Hybrid retrieval -----------------------------------------------------
# RETRIEVAL_MODE:
#   "dense"  -> dense-only (Chroma similarity search), used as the baseline
#   "hybrid" -> dense + BM25 + RRF (+ optional cross-encoder rerank)
#   "table"  -> hybrid + table-aware retrieval (TableStore rows/cells)
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid").strip().lower()

# Number of evidence chunks finally returned to the downstream graph.
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))

# Candidate counts fetched from each retrieval leg before RRF fusion.
DENSE_CANDIDATE_K = int(os.getenv("DENSE_CANDIDATE_K", "20"))
BM25_CANDIDATE_K = int(os.getenv("BM25_CANDIDATE_K", "20"))

# Reciprocal Rank Fusion smoothing constant k.
RRF_K = int(os.getenv("RRF_K", "60"))

# --- Cross-encoder reranker ----------------------------------------------
# RERANKER_ENABLED toggles the cross-encoder stage. The reranker model is
# configurable. If loading the model fails (e.g. offline), the pipeline
# degrades gracefully to the RRF ordering instead of crashing.
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "false").strip().lower() in ("1", "true", "yes")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2"
)

# --- Table-aware retrieval (Phase 3) -------------------------------------
TABLE_STORE_PATH = os.getenv("TABLE_STORE_PATH", "vectorstore/tables")
TABLE_DENSE_INDEX_PATH = os.getenv("TABLE_DENSE_INDEX_PATH", "vectorstore/tables_index")
# Whether table rows are embedded densely into a separate Chroma collection.
TABLE_DENSE_ENABLED = os.getenv("TABLE_DENSE_ENABLED", "true").strip().lower() in ("1", "true", "yes")

# Number of rows retained per table when building the table store.
TABLE_MAX_ROWS = int(os.getenv("TABLE_MAX_ROWS", "200"))