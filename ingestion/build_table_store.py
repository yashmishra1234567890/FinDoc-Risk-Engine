"""
Table store builder (Phase 3)
-----------------------------
Builds the persisted TableStore (row-level representations) from the REAL
financial PDFs in ``data/financial_docs/``. Optionally embeds row units into a
separate dense Chroma index (``TABLE_DENSE_INDEX_PATH``) so table-aware
retrieval gets a dense leg too.

The original PDFs and the existing text Chroma index are never modified.

Usage:
    python -m ingestion.build_table_store                 # build from all PDFs
    python -m ingestion.build_table_store <pdf1> [pdf2..] # specific PDFs
"""
import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_table_store")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_pdfs():
    folder = os.path.join(_ROOT, "data", "financial_docs")
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pdf") and not f.startswith(".")
    )


def build_table_store(pdf_paths, store_path=None, dense_enabled=True, embedder=None, max_rows=None):
    """Build a TableStore (and optional dense index) from the given PDFs."""
    from config.retrieval_config import TABLE_DENSE_INDEX_PATH, TABLE_STORE_PATH
    from ingestion.table_store import TableStore, get_table_store
    from ingestion.table_extract import normalize_table
    from ingestion.loader import load_pdf

    store_path = store_path or TABLE_STORE_PATH
    store = TableStore(path=store_path)

    for pdf_path in pdf_paths:
        source_id = os.path.basename(pdf_path)
        n_pages = 0
        for page in load_pdf(pdf_path):  # streamed page-wise (memory-safe)
            n_pages += 1
            for t_idx, raw in enumerate(page.get("tables") or []):
                tbl = normalize_table(raw, page["page_no"], source_id, t_idx, max_rows=max_rows)
                if tbl["row_count"] > 0:
                    store.add_table(tbl)
        logger.info("PDF %s (%d pages): %d tables stored", source_id, n_pages, store.table_count)

    store.finalize()
    store.persist(store_path)

    if dense_enabled and store.row_count > 0:
        embedder = embedder or _real_embedder()
        try:
            _embed_rows(store, embedder, TABLE_DENSE_INDEX_PATH)
            logger.info("Dense table index written to %s", TABLE_DENSE_INDEX_PATH)
        except Exception as exc:  # noqa: BLE001 - dense is optional
            logger.warning("Dense table index skipped (%s) - BM25 table retrieval still works.", exc)

    return store


def _real_embedder():
    from ingestion.embeddings import get_embedding_model
    return get_embedding_model()


def _embed_rows(store, embedder, path):
    """Embed every row unit into a separate persisted Chroma collection."""
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    docs = []
    for unit in store._rows:  # row-level representations
        doc_id = unit["id"]
        meta = dict(unit["metadata"])
        meta["_logical_id"] = doc_id
        docs.append(Document(page_content=unit["text"], metadata=meta))
    Chroma.from_documents(docs, embedder, persist_directory=os.path.abspath(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="*", help="PDF paths (default: all PDFs in data/financial_docs)")
    args = parser.parse_args(argv)

    pdfs = args.pdfs or _default_pdfs()
    if not pdfs:
        logger.error("No PDFs found in data/financial_docs/")
        return 1
    store = build_table_store(pdfs)
    print(f"\nTable store built: {store.table_count} tables / {store.row_count} row units")
    return 0


if __name__ == "__main__":
    sys.exit(main())