"""
Phase 2 - BM25 unit tests (pure-Python Okapi BM25).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.bm25 import BM25Index, tokenize


def test_tokenize_normalizes_financial_text():
    assert tokenize("Total Revenue 45,200,000 (FY 2025)") == [
        "total", "revenue", "45", "200", "000", "fy", "2025"
    ]


def test_tokenize_handles_empty_and_none():
    assert tokenize("") == []
    assert tokenize(None) == []


def _sample_index():
    index = BM25Index()
    index.add_documents(
        [
            {"id": "d1", "text": "The quick brown fox jumps over the lazy dog", "metadata": {"page_no": 1}},
            {"id": "d2", "text": "Total Revenue increased to 45,200,000 in FY 2025", "metadata": {"page_no": 2, "source_id": "report.pdf"}},
            {"id": "d3", "text": "The weather today is rainy and cold", "metadata": {"page_no": 3}},
            {"id": "d4", "text": "Total borrowings stood at 1,000,000 as of FY 2025", "metadata": {"page_no": 4, "has_table": True}},
        ]
    )
    return index


def test_bm25_ranks_relevant_document_first():
    index = _sample_index()
    results = index.search("total revenue fy 2025", k=3)
    assert results[0]["id"] == "d2"


def test_bm25_returns_requested_k():
    index = _sample_index()
    assert len(index.search("total revenue fy 2025", k=2)) == 2
    assert len(index.search("total revenue fy 2025", k=10)) == 2  # only matches


def test_bm25_preserves_metadata():
    index = _sample_index()
    results = index.search("total borrowings", k=1)
    assert results[0]["metadata"]["has_table"] is True
    assert results[0]["metadata"]["page_no"] == 4


def test_bm25_empty_query():
    assert _sample_index().search("", k=5) == []


def test_bm25_empty_corpus():
    assert BM25Index().search("anything", k=5) == []


def test_bm25_len():
    assert len(_sample_index()) == 4