"""
Retrieval benchmark (Phase 2 / Phase 3)
----------------------------------------
Compares the retrieval tiers on the REAL financial documents already present in
``data/financial_docs/``:

  1. dense-only (existing Chroma similarity search -> the baseline)
  2. hybrid (dense + BM25 + RRF + optional cross-encoder rerank)
  3. table-aware (hybrid + TableStore rows/cells)

The corpus is built by ingesting the original PDFs page-by-page (nothing is
downloaded, nothing is modified). Correctness is measured against where the
answer actually lives inside every index:

  - table queries:  ground-truth = the table row (page/table/row recorded)
  - paragraph queries: ground-truth = the paragraph chunk

Metrics: Recall@5, MRR, numerical accuracy. Every result row records the PDF,
page, table and query that produced it so numbers are auditable.

Run:  python -m evaluation.retrieval_eval

Uses the offline deterministic embedder (LocalHashEmbeddings) so results are
reproducible without network calls; the same documents are used for all tiers.
"""
import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retrieval_eval")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_pdfs():
    folder = os.path.join(_ROOT, "data", "financial_docs")
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pdf") and not f.startswith(".")
    )


from typing import Dict, List, Optional

from langchain_core.documents import Document

from ingestion.chunking import chunk_financial_pages, chunk_pages_excluding_tables
from ingestion.embeddings import LocalHashEmbeddings
from ingestion.loader import load_pdf
from ingestion.table_extract import normalize_table
from ingestion.table_store import TableStore
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import NoopReranker
from evaluation.datasets import dataset_status, build_synthetic_queries
from evaluation.metrics import (
    mean_reciprocal_rank,
    numerical_accuracy,
    recall_at_k,
)


def _chunk_id(source_id: str, page_no: int, index: int) -> str:
    return f"{source_id}:p{page_no}:c{index}"


def build_corpus(pdf_paths: List[str]):
    """
    Ingest the real PDFs (page-wise, streamed) into:
      - ``chunks_a`` : paragraphs with tables inlined (mirrors the current
                       production dense pipeline)
      - ``chunks_b`` : paragraphs only (tables live in the TableStore)
      - ``table_store`` : TableStore built from the same PDFs
    """
    chunks_a: List[Dict] = []
    chunks_b: List[Dict] = []
    table_store = TableStore(path=os.path.join(_ROOT, "vectorstore", "tables"))

    for pdf_path in pdf_paths:
        source_id = os.path.basename(pdf_path)
        pages = list(load_pdf(pdf_path))  # streamed generator -> page dicts

        loaded_a = chunk_financial_pages(pages)  # tables inlined (production style)
        for a_idx, chunk in enumerate(loaded_a):
            chunk["id"] = _chunk_id(source_id, chunk["page_no"], a_idx)
            chunk["source_id"] = source_id
            chunks_a.append(chunk)

        chunks = chunk_pages_excluding_tables(pages, source_id=source_id)
        for b_idx, chunk in enumerate(chunks):
            chunk["id"] = _chunk_id(source_id, chunk["page_no"], b_idx)
            chunks_b.append(chunk)

        for page in pages:
            for t_idx, raw in enumerate(page.get("tables") or []):
                tbl = normalize_table(raw, page["page_no"], source_id, t_idx)
                if tbl["row_count"] > 0:
                    table_store.add_table(tbl)

    table_store.finalize()
    return {
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "table_store": table_store,
    }


def _make_docs(chunks: List[Dict]) -> List[Document]:
    docs = []
    for c in chunks:
        meta = {
            "page_no": c["page_no"],
            "source_id": c.get("source_id", ""),
            "has_table": bool(c.get("has_table", c.get("is_table", False))),
            "_logical_id": c["id"],
        }
        docs.append(Document(page_content=c["content"], metadata=meta))
    return docs


def _truncate(text: str, size: int = 120) -> str:
    text = " ".join((text or "").split())
    return text[:size] + ("..." if len(text) > size else "")


def _relevant_for_a(query: Dict, chunks_a: List[Dict]) -> List[str]:
    """Find corpus-A chunk ids containing the query's ground-truth content."""
    out = []
    pdf = query.get("pdf")
    page = query.get("page_no")
    answer = query.get("answer")
    if answer is not None:
        ans_s = str(answer).strip()
        for c in chunks_a:
            if c.get("source_id") == pdf and c.get("page_no") == page and ans_s in (c.get("content") or ""):
                out.append(c["id"])
        return out
    phrase = " ".join((query.get("question") or "").split()[:8])
    for c in chunks_a:
        if c.get("source_id") == pdf and c.get("page_no") == page and phrase in (c.get("content") or ""):
            out.append(c["id"])
    return out


def _relevant_for_b(query: Dict) -> List[str]:
    return list(query.get("relevant_ids", []))


def _evaluate(queries, vs_a, vs_b, table_store, top_k):
    """Run all three retrieval tiers and measure Recall@5 / MRR / numerical accuracy."""
    retrievers = {
        "dense-only": HybridRetriever(vs_a, mode="dense", top_k=top_k,
                                      dense_k=max(20, top_k)),
        "hybrid": HybridRetriever(vs_a, mode="hybrid", top_k=top_k,
                                  dense_k=max(20, top_k), bm25_k=max(20, top_k)),
        "table-aware": HybridRetriever(vs_b, table_store=table_store, mode="table",
                                       top_k=top_k, dense_k=max(20, top_k), bm25_k=max(20, top_k)),
    }

    # keep the Chroma collections alive (in-memory) via closure refs
    results = {"rows": [], "summary": {}}
    for mode, retriever in retrievers.items():
        relevant_key = "relevant_a" if mode != "table-aware" else "relevant_b"
        rec, mrrs, nacc = [], [], []
        details = []
        for q in queries:
            rel = q.get(relevant_key) or []
            if not rel:
                continue
            try:
                res = retriever.retrieve(q["question"], k=top_k)
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s] retrieval failed for %r: %s", mode, q["question"], exc)
                res = []
            ids = [e["id"] for e in res]
            content_top = [e.get("content", "") for e in res]
            perc = recall_at_k(ids, rel, k=top_k)
            mrr = mean_reciprocal_rank(ids, rel)
            if q.get("answer") is not None:
                na = numerical_accuracy(q["answer"], content_top)
                nacc.append(na)
            else:
                na = None
            rec.append(perc)
            mrrs.append(mrr)
            details.append({
                "mode": mode,
                "query": q["question"],
                "pdf": q.get("pdf"),
                "page_no": q.get("page_no"),
                "table_id": q.get("table_id"),
                "row_idx": q.get("row_idx"),
                "answer": q.get("answer"),
                "recall_at_5": perc,
                "mrr": mrr,
                "num_acc": na,
                "top1_source": _truncate(content_top[0]) if content_top else "",
                "retrieved_ids": ids[:top_k],
            })
        results["rows"].extend(details)
        results["summary"][mode] = {
            "recall_at_5": round(sum(rec) / len(rec), 4) if rec else 0.0,
            "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
            "num_accuracy": round(sum(nacc) / len(nacc), 4) if nacc else 0.0,
            "num_queries": len(nacc),
        }
    return results


def run_benchmark(pdf_paths: Optional[List[str]] = None, top_k: int = 5) -> Dict:
    """Build the corpus from real PDFs, then run all three retrieval tiers."""
    from langchain_community.vectorstores import Chroma

    pdf_paths = pdf_paths or _default_pdfs()
    logger.info("Corpus PDFs: %s", ", ".join(os.path.basename(p) for p in pdf_paths))

    corpus = build_corpus(pdf_paths)
    embedder = LocalHashEmbeddings()

    # table-aware tier uses dense + BM25 legs for rows (same as text)
    if corpus["table_store"].row_count > 0:
        corpus["table_store"].build_dense_index(embedder)

    # Paragraph queries are derived from corpus_B paragraphs (their own ground
    # truth). Table queries are derived from the same real tables.
    queries = build_synthetic_queries(
        corpus["table_store"].tables, corpus["chunks_b"], max_per_table=2, max_paragraphs=50
    )
    logger.info("Generated %d labeled queries from real documents.", len(queries))

    for q in queries:
        q["relevant_a"] = _relevant_for_a(q, corpus["chunks_a"])
        q["relevant_b"] = _relevant_for_b(q)
    queries = [q for q in queries if q["relevant_a"] or q["relevant_b"]]
    logger.info("%d queries kept after ground-truth filtering.", len(queries))

    vs_a = Chroma.from_documents(_make_docs(corpus["chunks_a"]), embedder, collection_name="bench_a")
    vs_b = Chroma.from_documents(_make_docs(corpus["chunks_b"]), embedder, collection_name="bench_b")

    results = _evaluate(queries, vs_a, vs_b, corpus["table_store"], top_k)
    results["corpus"] = {
        "pdfs": [os.path.basename(p) for p in pdf_paths],
        "chunks_a": len(corpus["chunks_a"]),
        "chunks_b": len(corpus["chunks_b"]),
        "tables": corpus["table_store"].table_count,
        "rows": corpus["table_store"].row_count,
        "queries": len(queries),
        "dataset_status": dataset_status(),
    }
    return results


def write_report(results: Dict, top_k: int = 5) -> str:
    """Write the benchmark report (summary + per-query provenance rows)."""
    lines = []
    lines.append("# Retrieval Benchmark - Phase 2 / Phase 3")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    corpus = results["corpus"]
    lines.append("## Corpus (real financial PDFs from data/financial_docs/)")
    lines.append("")
    for p in corpus["pdfs"]:
        lines.append(f"- {p}")
    lines.append("")
    lines.append(f"- Paragraph chunks (production layout, tables inlined): {corpus['chunks_a']}")
    lines.append(f"- Paragraph chunks (table-aware layout): {corpus['chunks_b']}")
    lines.append(f"- Normalized tables: {corpus['tables']} | row units: {corpus['rows']}")
    lines.append(f"- Labeled queries: {corpus['queries']}")
    lines.append("")
    lines.append("Embedder: LocalHashEmbeddings (offline deterministic) | "
                 "Reranker: per .env config (default: none/RRF order) | top_k = %d" % top_k)
    lines.append("")
    lines.append("External datasets: %s" % ", ".join(
        f"{k}={v}" for k, v in corpus["dataset_status"].items()
    ))
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Retrieval tier | Recall@5 | MRR | Numerical accuracy |")
    lines.append("|---|---|---|---|")
    for mode, stats in results["summary"].items():
        lines.append(
            f"| {mode} | {stats['recall_at_5']} | {stats['mrr']} | {stats['num_accuracy']} "
            f"({stats['num_queries']} numeric q) |"
        )
    lines.append("")
    lines.append("## Per-query results (pdf / page / table / query provenance)")
    lines.append("")
    lines.append("| Mode | Query | PDF | Page | Table | Row | Answer | R@5 | MRR | NumAcc | Top-1 source |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for row in results["rows"]:
        answer = str(row.get("answer")) if row.get("answer") is not None else "-"
        na = "%.2f" % row["num_acc"] if row["num_acc"] is not None else "-"
        table_id = row.get("table_id") or "-"
        row_idx = row.get("row_idx") if row.get("row_idx") is not None else "-"
        source = " ".join((row.get("top1_source") or "").split())[:60]
        lines.append(
            f"| {row['mode']} | {row['query'][:60]} | {row.get('pdf')} | {row.get('page_no')} "
            f"| {table_id} | {row_idx} | {answer} | {row['recall_at_5']} | {row['mrr']} | {na} | {source} |"
        )
    lines.append("")

    report_path = os.path.join(_ROOT, "evaluation", "retrieval_results.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


if __name__ == "__main__":
    benchmark = run_benchmark()
    path = write_report(benchmark)
    print("\n=== Retrieval benchmark summary ===")
    for mode, stats in benchmark["summary"].items():
        print(f"{mode:>12}: Recall@5={stats['recall_at_5']}  MRR={stats['mrr']}  "
              f"NumAcc={stats['num_accuracy']} ({stats['num_queries']} numeric q)")
    print(f"\nReport written to {path}")