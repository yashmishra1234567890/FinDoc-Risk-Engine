"""
HybridRetriever
---------------
Orchestrates the retrieval legs and fusion:

    query
      |-- Dense (ChromaDB)      +------------------+
      |-- BM25 (lexical)  ----->|  RRF Fusion      |--> cross-encoder
      |-- Table rows (Phase 3)  +------------------+    rerank (optional)
                                                            |--> top-k evidence

The pipeline is fully configurable through ``config/retrieval_config.py``:
  - ``mode="dense"``   -> dense-only (Chroma similarity search, the baseline)
  - ``mode="hybrid"``  -> dense + BM25 + RRF (+ optional rerank)
  - ``mode="table"``   -> hybrid + table-aware retrieval (rows/cells)

The persisted Chroma collection is never rebuilt or emptied; it is read as-is
and used as the dense leg. Existing Chroma data is therefore preserved.
"""
import logging
from typing import Dict, List, Optional

from config.retrieval_config import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    RERANKER_ENABLED,
    RERANKER_MODEL,
    RETRIEVAL_MODE,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from retrieval.bm25 import BM25Index
from retrieval.citations import make_citation
from retrieval.dense import ChromaDenseIndex
from retrieval.reranker import Reranker, build_reranker
from retrieval.rrf import rrf_fuse

logger = logging.getLogger(__name__)

_VALID_MODES = {"dense", "hybrid", "table"}


class HybridRetriever:
    """Retrieves top-k evidence using dense + BM25 + RRF (+ reranker)."""

    def __init__(
        self,
        vectorstore,
        table_store: Optional[object] = None,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
        dense_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.vectorstore = vectorstore
        self.table_store = table_store
        self.mode = (mode or RETRIEVAL_MODE).strip().lower()
        if self.mode not in _VALID_MODES:
            logger.warning("Unknown RETRIEVAL_MODE %r; falling back to 'hybrid'.", self.mode)
            self.mode = "hybrid"
        self.top_k = top_k or RETRIEVAL_TOP_K
        self.dense_k = dense_k or DENSE_CANDIDATE_K
        self.bm25_k = bm25_k or BM25_CANDIDATE_K
        self.rrf_k = rrf_k or RRF_K
        self.reranker = reranker or build_reranker(RERANKER_ENABLED, RERANKER_MODEL)

        self.dense = ChromaDenseIndex(vectorstore)
        self.bm25 = self._build_bm25(vectorstore) if self.mode != "dense" else None
        self.corpus_count = (len(self.bm25) if self.bm25 else 0)

    # ------------------------------------------------------------- building
    @staticmethod
    def _build_bm25(vectorstore) -> BM25Index:
        """Build a BM25 index over the docs already persisted in Chroma."""
        try:
            collection = vectorstore._collection
        except Exception:  # noqa: BLE001
            collection = None

        if collection is not None and hasattr(collection, "get"):
            data = collection.get(include=["documents", "metadatas"])
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
        else:
            ids = [str(i) for i in range(len(getattr(vectorstore, "index", []) or []))]
            docs = [d.page_content for d in (getattr(vectorstore, "index", []) or [])]
            metas = [dict(d.metadata or {}) for d in (getattr(vectorstore, "index", []) or [])]

        entries = [
            {
                "id": (meta or {}).get("_logical_id") or doc_id,
                "text": text or "",
                "metadata": {k: v for k, v in (meta or {}).items() if k != "_logical_id"},
            }
            for doc_id, text, meta in zip(ids, docs, metas)
        ]
        index = BM25Index()
        index.add_documents(entries)
        logger.info("Built BM25 index over %d documents from Chroma.", len(index))
        return index

    # ------------------------------------------------------------- retrieval
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Run the configured pipeline and return top-k evidence dicts."""
        k = k or self.top_k
        ranked_lists: List[List[Dict]] = []

        # 1. dense (Chroma) - always present so the baseline is preserved
        dense_entries = self.dense.search(query, k=self.dense_k)
        ranked_lists.append(dense_entries)

        if self.mode != "dense":
            # 2. BM25 leg
            bm25_entries = self.bm25.search(query, k=self.bm25_k) if self.bm25 else []
            ranked_lists.append(bm25_entries)

            # 3. table-aware leg (Phase 3): BM25 + dense legs kept separate so
            #    table rows get the same RRF representation as text chunks.
            if self.mode == "table" and self.table_store is not None:
                try:
                    ranked_lists.append(self.table_store.search_bm25(query, k=self.bm25_k))
                    dense_table = self.table_store.search_dense(query, k=self.dense_k)
                    if dense_table:
                        ranked_lists.append(dense_table)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Table-aware retrieval failed (%s); skipping.", exc)

        # 4. RRF fusion
        fused = rrf_fuse(ranked_lists, k=self.rrf_k) if self.mode != "dense" else dense_entries

        # 5. rerank (optional; NoopReranker preserves RRF order)
        if self.reranker is not None and fused:
            fused = self.reranker.rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]

        # 6. finalize shape for the downstream graph & citation
        for i, entry in enumerate(fused):
            if "rank" not in entry:
                entry["rank"] = i + 1
            entry["content"] = entry.get("text") or entry.get("content") or ""
            meta = entry.get("metadata") or {}
            for key in (
                "page_no", "source_id", "document_id", "table_id", "row_idx",
                "has_table", "is_table", "header", "cells", "best_column",
            ):
                if key not in entry and key in meta:
                    entry[key] = meta[key]
            entry["citation"] = make_citation(entry)
        return fused


# ------------------------------------------------------------------ cache
_RETRIEVER_CACHE: Dict[tuple, HybridRetriever] = {}


def get_hybrid_retriever(
    vectorstore,
    table_store: Optional[object] = None,
    mode: Optional[str] = None,
    top_k: Optional[int] = None,
) -> HybridRetriever:
    """Return a cached HybridRetriever for the given vectorstore/mode.

    The BM25 index is built once per (vectorstore, mode) and reused across
    queries, so repeated calls stay cheap.
    """
    key = (id(vectorstore), mode, top_k, id(table_store))
    if key not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[key] = HybridRetriever(
            vectorstore, table_store=table_store, mode=mode, top_k=top_k
        )
    return _RETRIEVER_CACHE[key]
