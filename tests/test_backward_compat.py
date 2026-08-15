"""
Phase 2 - Backward compatibility.

The EXISTING dense-only retrieval entry point and the LangGraph retriever agent
must keep working with the original chunk shape (content/page_no/has_table),
and the legacy indexer must still produce valid documents.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import retrieval.retriever as retriever_mod
from retrieval.retriever import retrieve_chunks
from ingestion.embeddings import LocalHashEmbeddings


def _fake_chunks_docs():
    docs = [
        Document(page_content="Total debt and borrowings rose in FY 2023.", metadata={"page_no": 4, "has_table": True}),
        Document(page_content="Equity increased steadily across the year.", metadata={"page_no": 5, "has_table": False}),
        Document(page_content="The management discussion talks about growth.", metadata={"page_no": 6, "has_table": False}),
    ]
    return Chroma.from_documents(
        docs, LocalHashEmbeddings(), collection_name="bc_" + uuid.uuid4().hex[:8]
    )


def test_dense_retrieve_chunks_preserves_original_shape():
    vs = _fake_chunks_docs()
    results = retrieve_chunks(vs, "total debt", k=3, mode="dense")
    assert results
    for r in results:
        assert "content" in r and "page_no" in r and "has_table" in r
        assert r["page_no"] in (4, 5, 6)


def test_retriever_agent_backward_compatible(monkeypatch):
    import agents.retriever_agent

    # force the legacy dense path so the agent behaves exactly as before
    monkeypatch.setattr(retriever_mod, "RETRIEVAL_MODE", "dense")
    vs = _fake_chunks_docs()
    chunks = agents.retriever_agent.retrieve_content(["total debt and equity"], vs)
    assert isinstance(chunks, list)
    assert chunks, "retriever agent returned no chunks"
    assert all("content" in c and "page_no" in c for c in chunks)
    assert len(chunks) <= 6  # agent caps context


def test_legacy_indexer_still_builds_docs():
    from ingestion.indexer import create_documents_from_chunks

    chunks = [
        {"content": "Revenue for FY25 was 45.2M.", "page_no": 1, "has_table": False},
        {"content": "Borrowings: 1,000,000", "page_no": 2, "has_table": True},
    ]
    docs = create_documents_from_chunks(chunks)
    assert len(docs) == 2
    assert docs[0].metadata["page_no"] == 1
    assert docs[1].metadata["has_table"] is True


def test_retrieve_chunks_accepts_explicit_hybrid(monkeypatch):
    monkeypatch.setattr(retriever_mod, "RETRIEVAL_MODE", "hybrid")
    vs = _fake_chunks_docs()
    results = retrieve_chunks(vs, "total debt borrowings", k=3)
    assert results
    assert all("content" in r for r in results)
    # bonus fields preserved when available
    assert any(r.get("citation") is not None for r in results)