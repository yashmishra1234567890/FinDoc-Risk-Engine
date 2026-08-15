"""
Phase 3 - Large-document behavior (real PDFs).

No 300+ page financial PDF ships in the repo, so this test exercises the
memory-safe streaming behavior on the LARGEST real document available,
``sa-fy23-annual-finstatement.pdf`` (annual financial statement). It also
exercises the guardrails that keep large documents workable: streaming loader
(generator, not a full in-memory load), per-page table extraction and a BM25
index over all row units without loading the whole PDF at once.
"""
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.chunking import chunk_pages_excluding_tables
from ingestion.loader import load_pdf
from ingestion.table_extract import normalize_table
from ingestion.table_store import TableStore


_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_docs")
LARGE_PDF = os.path.join(_DATA, "sa-fy23-annual-finstatement.pdf")


def test_loader_is_streaming_generator():
    assert isinstance(load_pdf(LARGE_PDF), types.GeneratorType)


def test_large_document_streams_page_wise():
    # iterate the generator without materializing the whole document
    pages_processed = 0
    table_page_count = 0
    for page in load_pdf(LARGE_PDF):
        pages_processed += 1
        if page.get("tables"):
            table_page_count += 1
        assert isinstance(page["page_no"], int)
    # annual statement is the largest real doc (>= 50 pages)
    assert pages_processed >= 50
    assert table_page_count > 0


def test_large_document_table_store_and_bm25():
    store = TableStore(path=None)
    for page in load_pdf(LARGE_PDF):
        for t_idx, raw in enumerate(page.get("tables") or []):
            tbl = normalize_table(raw, page["page_no"], "sa-fy23.pdf", t_idx)
            if tbl["row_count"] > 0:
                store.add_table(tbl)
    store.finalize()
    assert store.table_count > 0
    assert store.row_count > 0
    # BM25 over all row units works at this scale
    results = store.search("total trade payables", k=5)
    assert isinstance(results, list)


def test_large_document_text_chunking_no_tables_in_text():
    pages = list(load_pdf(LARGE_PDF))
    chunks = chunk_pages_excluding_tables(pages, source_id="sa-fy23.pdf")
    assert chunks
    assert all(c["source_id"] == "sa-fy23.pdf" for c in chunks)
    assert all(c["is_table"] is False for c in chunks)
    # no table markers leak into pure text chunks
    assert not any("[TABLE" in c["content"] for c in chunks)