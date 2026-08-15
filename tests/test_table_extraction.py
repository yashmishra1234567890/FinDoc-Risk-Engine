"""
Phase 3 - Table extraction tests (real financial PDFs).

Verifies header/row/cell preservation, table/page/source ids, None-cell
handling and the header detection heuristic.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ingestion.loader import load_pdf
from ingestion.table_extract import (
    find_cell,
    is_numeric,
    make_table_id,
    make_row_id,
    normalize_table,
    row_to_unit,
)


_DATA = os.path.join(os.path.dirname(__file__), "..", "data", "financial_docs")
REPORT_PDF = os.path.join(_DATA, "Financial Performance Report.pdf")


def _first_real_table():
    """First table found in the real 'Financial Performance Report' PDF."""
    for page in load_pdf(REPORT_PDF):
        for raw in page.get("tables") or []:
            if raw and any(any(c for c in r) for r in raw):
                return page, raw
    raise AssertionError("No table found in the Financial Performance Report PDF")


def test_header_and_rows_preserved():
    page, raw = _first_real_table()
    table = normalize_table(raw, page["page_no"], "report.pdf", 0)
    assert table["header"]  # header preserved
    assert any("FY" in str(h) for h in table["header"]), f"expected year columns, got {table['header']}"
    assert table["row_count"] > 0
    assert table["column_count"] >= 2
    # rows start at 1 and keep their cells
    assert table["rows"][0]["row_idx"] == 1
    assert all(isinstance(r["cells"], list) for r in table["rows"])


def test_table_metadata_ids():
    page, raw = _first_real_table()
    table = normalize_table(raw, page["page_no"], "report.pdf", 2)
    assert table["table_id"] == make_table_id("report.pdf", page["page_no"], 2)
    assert table["page_no"] == page["page_no"]
    assert table["source_id"] == "report.pdf"
    assert make_row_id(table["table_id"], 3) == f"{table['table_id']}:r3"


def test_none_cells_become_empty_strings():
    raw = [
        ["Category", "FY 2025", "FY 2024"],
        ["Revenue", "45,200,000", None],
        ["Costs", None, "10,000"],
    ]
    table = normalize_table(raw, 1, "doc.pdf", 0)
    assert table["rows"][0]["cells"] == ["Revenue", "45,200,000", ""]
    assert table["rows"][1]["cells"] == ["Costs", "", "10,000"]


def test_fully_numeric_first_row_gets_synthetic_header():
    raw = [["1,000", "2,000"], ["3,000", "4,000"]]
    table = normalize_table(raw, 1, "doc.pdf", 0)
    assert table["header"] == ["col_0", "col_1"]
    assert table["row_count"] == 2


def test_is_numeric():
    assert is_numeric("45,200,000")
    assert is_numeric("(18,100,000)")
    assert is_numeric("+18.0%")
    assert is_numeric("1,234.56")
    assert not is_numeric("Total Revenue")
    assert not is_numeric("")


def test_find_cell_returns_cell_level_evidence():
    raw = [
        ["Category", "FY 2025 ($)", "FY 2024 ($)"],
        ["Total Revenue", "45,200,000", "38,300,000"],
        ["Gross Profit", "27,100,000", "22,300,000"],
    ]
    table = normalize_table(raw, 1, "doc.pdf", 0)
    hit = find_cell("Total Revenue FY 2025", table)
    assert hit
    assert hit["row_idx"] == 1
    assert hit["value"] == "45,200,000"
    assert hit["column_header"] == "FY 2025 ($)"
    assert hit["table_id"] == table["table_id"]
    assert hit["page_no"] == 1


def test_row_to_unit_keeps_structure():
    raw = [["Category", "FY 2025"], ["Revenue", "45,200,000"]]
    table = normalize_table(raw, 1, "doc.pdf", 0)
    unit = row_to_unit(table, table["rows"][0])
    assert unit["id"] == make_row_id(table["table_id"], 1)
    assert unit["metadata"]["header"] == table["header"]
    assert unit["metadata"]["cells"] == ["Revenue", "45,200,000"]
    assert unit["metadata"]["is_table"] is True
    assert "45,200,000" in unit["text"]