"""
TableStore (Phase 3)
--------------------
A lightweight persistent store of normalized financial tables. Tables are
stored column-wise (header + rows + cells) so row/column relationships are
preserved, and every row becomes its own retrievable unit (row-level
representation).

Retrieval options:
  - BM25 over row units (always available, offline)
  - optional dense embedding of row units into a separate Chroma collection
    (built separately so the existing text Chroma is never touched/mutated)

Persistence is JSONL-based (small, diff-friendly). Test/build artifacts are
written to ``TABLE_STORE_PATH`` / ``TABLE_DENSE_INDEX_PATH`` and never into
``data/financial_docs/``.
"""
import json
import logging
import os
from typing import Dict, List, Optional

from ingestion.table_extract import find_cell, row_to_unit
from retrieval.bm25 import BM25Index
from retrieval.citations import make_citation
from retrieval.rrf import rrf_fuse

logger = logging.getLogger(__name__)

_TABLES_FILE = "tables.jsonl"
_ROWS_FILE = "rows.jsonl"


class TableStore:
    """Normalized table store with row-level retrieval and citations."""

    def __init__(self, path: Optional[str] = None, dense_index: Optional[object] = None):
        self.path = path or "vectorstore/tables"
        self.dense_index = dense_index  # optional ChromaDenseIndex over table rows
        self.tables: List[Dict] = []
        self._rows: List[Dict] = []
        self._bm25 = BM25Index()
        self._finalized = False
        self._maybe_load()

    # ----------------------------------------------------------------- ingest
    def add_table(self, table: Dict) -> None:
        """Add a normalized table (see ``ingestion.table_extract.normalize_table``)."""
        self.tables.append(table)
        docs = []
        for row in table["rows"]:
            unit = row_to_unit(table, row)
            self._rows.append(unit)
            docs.append(
                {
                    "id": unit["id"],
                    "text": unit["text"],
                    "metadata": unit["metadata"],
                }
            )
        self._bm25.add_documents(docs)

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def table_count(self) -> int:
        return len(self.tables)

    def finalize(self) -> None:
        self._finalized = True

    # ------------------------------------------------------------------ search
    def _decorate(self, entries: List[Dict]) -> List[Dict]:
        """Add rank/content/is_table/citation to raw leg entries."""
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
            entry["content"] = entry.get("text") or entry.get("content") or ""
            meta = entry.get("metadata") or {}
            for key in ("source_id", "page_no", "table_id", "row_idx", "header", "cells"):
                if key not in entry and key in meta:
                    entry[key] = meta[key]
            entry["is_table"] = True
            entry["citation"] = make_citation(entry)
        return entries

    def search_bm25(self, query: str, k: int = 20) -> List[Dict]:
        """Raw BM25 leg over table row units (top-level is_table/citation added)."""
        if self._finalized and not self._bm25:
            return []
        return self._decorate(self._bm25.search(query, k=k))

    def search_dense(self, query: str, k: int = 20) -> List[Dict]:
        """Raw dense leg over table row units (empty when no dense index)."""
        if self.dense_index is None:
            return []
        try:
            return self._decorate(self.dense_index.search(query, k=k))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Table dense search failed (%s); empty dense leg.", exc)
            return []

    def search(self, query: str, k: int = 20) -> List[Dict]:
        """Fused table retrieval (dense + BM25 via RRF), decorated with citations."""
        bm25_entries = self.search_bm25(query, k=k)
        dense_entries = self.search_dense(query, k=k)
        if dense_entries:
            fused = rrf_fuse([dense_entries, bm25_entries], k=60)
        else:
            fused = bm25_entries
        return fused

    def build_dense_index(self, embedder, persist_directory: Optional[str] = None,
                          collection_name: str = "table_rows"):
        """Embed every row unit densely and attach a ChromaDenseIndex.

        ``persist_directory=None`` keeps the collection in memory (tests /
        offline evaluation); a path persists it to ``TABLE_DENSE_INDEX_PATH``
        for production table-aware retrieval. ``collection_name`` isolates the
        collection when several in-memory stores exist in one process.
        """
        if not self._rows:
            return None
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from retrieval.dense import ChromaDenseIndex

        docs = []
        for unit in self._rows:
            meta = dict(unit["metadata"])
            meta["_logical_id"] = unit["id"]  # align dense leg with BM25 leg ids
            docs.append(Document(page_content=unit["text"], metadata=meta))
        vectorstore = Chroma.from_documents(
            docs,
            embedder,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self.dense_index = ChromaDenseIndex(vectorstore)
        return self.dense_index

    def cell_evidence(self, query: str, table_id: Optional[str] = None) -> Dict:
        """Best-effort cell-level evidence (row/column/value) for a query."""
        candidates = self.tables
        if table_id is not None:
            candidates = [t for t in self.tables if t["table_id"] == table_id]
        best = {}
        best_score = 0
        for table in candidates:
            hit = find_cell(query, table)
            if hit and hit.get("score", 0) > best_score:
                best = hit
                best_score = hit["score"]
        return best

    # ------------------------------------------------------------- persistence
    def persist(self, path: Optional[str] = None) -> str:
        target = path or self.path
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, _TABLES_FILE), "w", encoding="utf-8") as f:
            for table in self.tables:
                f.write(json.dumps(table, ensure_ascii=False) + "\n")
        with open(os.path.join(target, _ROWS_FILE), "w", encoding="utf-8") as f:
            for unit in self._rows:
                f.write(json.dumps(unit, ensure_ascii=False) + "\n")
        logger.info("Persisted %d tables / %d rows to %s", len(self.tables), len(self._rows), target)
        return target

    def _maybe_load(self) -> None:
        tables_path = os.path.join(self.path, _TABLES_FILE)
        if not os.path.exists(tables_path):
            return
        with open(tables_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.add_table(json.loads(line))
        logger.info("Loaded %d tables from %s", len(self.tables), self.path)

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        path: Optional[str] = None,
        max_rows: Optional[int] = None,
    ) -> "TableStore":
        """Build a TableStore directly from a real financial PDF (streamed page-wise)."""
        from ingestion.loader import load_pdf
        from ingestion.table_extract import normalize_table

        store = cls(path=path)
        source_id = os.path.basename(pdf_path)
        for page in load_pdf(pdf_path):
            for idx, raw in enumerate(page.get("tables") or []):
                store.add_table(normalize_table(raw, page["page_no"], source_id, idx, max_rows=max_rows))
        store.finalize()
        return store


# ------------------------------------------------------------------ loader
_TABLE_STORE_CACHE: Dict[str, "TableStore"] = {}


def get_table_store(path: Optional[str] = None) -> "TableStore":
    """
    Lazily load (and cache) the persisted TableStore used by table-aware retrieval.

    If a dense table index exists at ``TABLE_DENSE_INDEX_PATH`` it is attached
    as an additional dense leg; otherwise retrieval falls back to BM25 over the
    row units (still fully functional offline).
    """
    from config.retrieval_config import (
        TABLE_DENSE_ENABLED,
        TABLE_DENSE_INDEX_PATH,
        TABLE_STORE_PATH,
    )

    target = path or TABLE_STORE_PATH
    if target not in _TABLE_STORE_CACHE:
        dense_index = None
        if TABLE_DENSE_ENABLED:
            dense_path = os.path.abspath(TABLE_DENSE_INDEX_PATH)
            if os.path.exists(os.path.join(dense_path, "chroma.sqlite3")):
                try:
                    from langchain_community.vectorstores import Chroma
                    from retrieval.dense import ChromaDenseIndex
                    from ingestion.embeddings import get_embedding_model

                    vectorstore = Chroma(persist_directory=dense_path, embedding_function=get_embedding_model())
                    dense_index = ChromaDenseIndex(vectorstore)
                except Exception as exc:  # noqa: BLE001 - degrade to BM25
                    logger.warning("Table dense index unavailable (%s); BM25 only.", exc)
                    dense_index = None
        store = TableStore(path=target, dense_index=dense_index)
        store._finalized = True
        _TABLE_STORE_CACHE[target] = store
    return _TABLE_STORE_CACHE[target]