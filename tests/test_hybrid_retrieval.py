"""
Phase 2 - Hybrid retrieval integration tests (dense / hybrid / table modes).

Runs against the REAL financial PDFs already present in data/financial_docs/.
No network is used: Chroma is built in-memory with the offline embedder, and
BM25 is pure Python.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from ingestion.chunking import chunk_financial_pages, chunk_pages_excluding_tables
from ingestion.embeddings import LocalHashEmbeddings
from ingestion.loader import load_pdf
from ingestion.table_extract import normalize_table
from ingestion.table_store import TableStore
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import NoopReranker


_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_docs")

CORPUS_PDFS = [
    os.path.join(_DATA, "Financial Performance Report.pdf"),
    os.path.join(_DATA, "VIL-QR-Q1FY25.pdf"),
]


def _build_docs(exclude_tables=False):
    """Build text documents from the real PDFs.

    ``exclude_tables=True`` uses the Phase-3 table-aware layout (paragraph text
    only, tables live in the TableStore). Otherwise it mirrors the legacy
    production layout where tables are inlined into the text chunks.
    """
    docs = []
    for path in CORPUS_PDFS:
        source_id = os.path.basename(path)
        pages = list(load_pdf(path))
        chunks = (
            chunk_pages_excluding_tables(pages, source_id=source_id)
            if exclude_tables
            else chunk_financial_pages(pages)
        )
        for c_idx, chunk in enumerate(chunks):
            doc_id = f"{source_id}:p{chunk['page_no']}:c{c_idx}"
            docs.append(
                Document(
                    page_content=chunk["content"],
                    metadata={
                        "page_no": chunk["page_no"],
                        "source_id": source_id,
                        "has_table": bool(chunk.get("has_table", False)),
                        "is_table": bool(chunk.get("is_table", False)),
                        "_logical_id": doc_id,
                    },
                )
            )
    return docs


def _build_table_store():
    store = TableStore(path=None)
    for path in CORPUS_PDFS:
        source_id = os.path.basename(path)
        for page in load_pdf(path):
            for t_idx, raw in enumerate(page.get("tables") or []):
                tbl = normalize_table(raw, page["page_no"], source_id, t_idx)
                if tbl["row_count"] > 0:
                    store.add_table(tbl)
    store.finalize()
    # table-aware mode uses dense + BM25 legs for rows, just like text
    store.build_dense_index(LocalHashEmbeddings())
    return store


def _retriever(mode, vs=None, table_store=None, top_k=5):
    vs = vs or Chroma.from_documents(
        _build_docs(), LocalHashEmbeddings(), collection_name="test_vs_hybrid"
    )
    return HybridRetriever(
        vs,
        table_store=table_store,
        mode=mode,
        top_k=top_k,
        dense_k=20,
        bm25_k=20,
        reranker=NoopReranker(),
    )


def test_dense_mode_returns_valid_evidence():
    vs = Chroma.from_documents(_build_docs(), LocalHashEmbeddings(), collection_name="test_vs_dense")
    retriever = _retriever("dense", vs=vs)
    results = retriever.retrieve("total revenue", k=5)
    assert len(results) <= 5
    assert results
    assert all("content" in r for r in results)
    assert all("page_no" in r for r in results)
    assert all(isinstance(r["page_no"], int) for r in results)


def test_hybrid_mode_has_scores_and_citations():
    retriever = _retriever("hybrid")
    results = retriever.retrieve("total revenue fiscal 2025", k=5)
    assert results
    assert all("citation" in r for r in results)
    assert all("scores" in r for r in results)
    # hybrid should record rrf (+ per-leg) scores
    assert any(r.get("score") is not None or r.get("rrf_score") for r in results)
    # ranks sequential 1..len
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


def test_hybrid_respects_top_k():
    retriever = _retriever("hybrid", top_k=3)
    assert len(retriever.retrieve("borrowings", k=3)) <= 3


def test_table_mode_surfaces_row_units():
    # table-aware layout: table content lives ONLY in the TableStore, not in
    # the text chunks -- so table mode must return row units for table queries.
    vs = Chroma.from_documents(_build_docs(exclude_tables=True), LocalHashEmbeddings(),
                               collection_name="test_vs_table")
    store = _build_table_store()
    retriever = _retriever("table", vs=vs, table_store=store)
    results = retriever.retrieve("total revenue", k=5)
    assert results
    table_rows = [r for r in results if r.get("is_table")]
    # table mode should include at least one structured row from the real tables
    assert table_rows, "expected a table-aware row unit in results"
    row = table_rows[0]
    assert row["citation"]["table_id"] is not None
    assert row["citation"]["page_no"] is not None


def test_hybrid_rrf_metric_ordering():
    # dense + BM25 offer complementary signals that RRF can fuse; verify the
    # fusion returns a single merged, deduplicated ranking.
    retriever = _retriever("hybrid")
    results = retriever.retrieve("profit for the period", k=10)
    ids = [r["id"] for r in results]
    assert len(ids) == len(set(ids))  # no duplicates after fusion