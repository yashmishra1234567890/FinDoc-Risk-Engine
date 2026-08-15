"""
Retrieval entry points.

``retrieve_chunks`` is the backward-compatible function used by the LangGraph
retriever agent. It still returns dicts with ``content``/``page_no``/``has_table``
but now routes through the configurable hybrid pipeline:

  - RETRIEVAL_MODE=dense  -> original Chroma similarity search (baseline)
  - RETRIEVAL_MODE=hybrid -> dense + BM25 + RRF (+ optional cross-encoder rerank)
  - RETRIEVAL_MODE=table  -> hybrid + table-aware retrieval (rows/cells)

Pass ``mode=...`` explicitly to force a mode (used by tests and the benchmark).
"""

from config.retrieval_config import RETRIEVAL_MODE
from retrieval.hybrid import get_hybrid_retriever

try:
    from ingestion.table_store import get_table_store as _get_table_store
except Exception:  # pragma: no cover - table store optional
    _get_table_store = None


def _dense_retrieve_chunks(vectorstore, query: str, k: int = 15):
    """Original dense-only retrieval (unchanged behavior)."""
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "page_no": doc.metadata["page_no"],
            "has_table": doc.metadata["has_table"]
        }
        for doc in results
    ]


def retrieve_chunks(vectorstore, query: str, k: int = 15, mode=None):
    """
    Retrieve top-k relevant chunks for a query.

    Preserves the original output shape (content/page_no/has_table) while
    supporting the hybrid and table-aware pipelines. Extra keys (source_id,
    table_id, row_idx, header, cells, citation, scores, rank) are added when
    available so downstream analysis/citation code can use them.
    """
    mode = (mode or RETRIEVAL_MODE).strip().lower()

    if mode == "dense":
        return _dense_retrieve_chunks(vectorstore, query, k=k)

    table_store = None
    if mode == "table" and _get_table_store is not None:
        table_store = _get_table_store()

    retriever = get_hybrid_retriever(vectorstore, table_store=table_store, mode=mode, top_k=k)
    results = retriever.retrieve(query, k=k)

    out = []
    for e in results:
        row = {
            "content": e.get("content", ""),
            "page_no": e.get("page_no"),
            "has_table": bool(e.get("has_table", e.get("is_table", False))),
        }
        for extra in (
            "id", "source_id", "document_id", "table_id", "row_idx",
            "is_table", "header", "cells", "citation", "scores", "rank",
            "rrf_score", "rrf_contributions",
        ):
            if extra in e:
                row[extra] = e[extra]
        out.append(row)
    return out
