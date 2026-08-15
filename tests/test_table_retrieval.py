"""
Phase 3 - TableStore retrieval tests (real financial PDFs).

Verifies table-specific retrieval, row/cell retrieval, and citations
(source / page / table / row / column header).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.loader import load_pdf
from ingestion.table_extract import normalize_table
from ingestion.table_store import TableStore


_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_docs")
REPORT_PDF = os.path.join(_DATA, "Financial Performance Report.pdf")


def _build_report_store():
    store = TableStore(path=None)
    for page in load_pdf(REPORT_PDF):
        for t_idx, raw in enumerate(page.get("tables") or []):
            tbl = normalize_table(raw, page["page_no"], "report.pdf", t_idx)
            if tbl["row_count"] > 0:
                store.add_table(tbl)
    store.finalize()
    return store


def test_table_specific_retrieval_finds_relevant_row():
    store = _build_report_store()
    results = store.search("Total Revenue FY 2025", k=5)
    assert results
    # the row containing Total Revenue should surface
    joined = " ".join(r.get("content", "") for r in results)
    assert "Total Revenue" in joined or "45,200,000" in joined


def test_row_unit_citation_fields():
    store = _build_report_store()
    results = store.search("Gross Profit FY 2024", k=5)
    assert results
    row = results[0]
    cit = row["citation"]
    assert cit["source_id"] == "report.pdf"
    assert cit["page_no"] is not None
    assert cit["table_id"] is not None
    assert cit["row_idx"] is not None
    # citation should expose the column header for the best cell
    assert cit.get("column_header") is not None


def test_cell_level_evidence():
    store = _build_report_store()
    evidence = store.cell_evidence("Total Revenue FY 2025")
    assert evidence
    assert evidence["value"] == "45,200,000"
    assert evidence["column_header"] == "FY 2025 ($)"
    assert evidence["source_id"] == "report.pdf"


def test_table_persist_and_reload(tmp_path):
    store = _build_report_store()
    store.persist(str(tmp_path))
    reloaded = TableStore(path=str(tmp_path))
    assert reloaded.table_count == store.table_count
    assert reloaded.row_count == store.row_count
    assert reloaded.search("Total Revenue", k=3)


def test_row_retrieval_respects_top_k():
    store = _build_report_store()
    assert len(store.search("revenue", k=2)) <= 2