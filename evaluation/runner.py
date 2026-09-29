"""Reproducible FinDoc v2 evaluation runner.

The supported set is generated from the repository's local PDFs and their
normalized tables, so expected facts are never invented.  Run with:

    python -m evaluation.runner
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List

from agents.evidence_agent import check_evidence_sufficiency
from evaluation.retrieval_eval import _default_pdfs, run_benchmark

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MANIFEST = os.path.join(_ROOT, "evaluation", "dataset.json")
_RESULTS = os.path.join(_ROOT, "evaluation", "results", "findoc_v2.json")
_REPORT = os.path.join(_ROOT, "evaluation", "results", "findoc_v2.md")


def load_manifest() -> Dict:
    with open(_MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def build_records(benchmark: Dict) -> List[Dict]:
    """Convert retrieval benchmark rows into labeled evaluation records."""
    records = []
    for index, row in enumerate(benchmark.get("rows", []), start=1):
        if row.get("mode") != "table-aware":
            continue
        answer = row.get("answer")
        category = "numerical" if answer is not None else "factual_lookup"
        if row.get("table_id"):
            category = "table" if answer is not None else "multi_step"
        records.append(
            {
                "id": f"doc_{index:03d}",
                "question": row["query"],
                "category": category,
                "expected_answer": answer,
                "source": row.get("pdf"),
                "page": row.get("page_no"),
                "table_id": row.get("table_id"),
                "relevant_ids": row.get("retrieved_ids", []),
            }
        )
    manifest = load_manifest()
    for index, question in enumerate(manifest["unsupported_questions"], start=1):
        records.append(
            {
                "id": f"unsupported_{index:03d}",
                "question": question,
                "category": "unsupported",
                "supported": False,
                "expected_answer": None,
            }
        )
    return records


def _write_report(payload: Dict) -> None:
    metrics = payload["metrics"]
    lines = [
        "# FinDoc v2 Evaluation",
        "",
        f"Dataset records: {payload['dataset_size']}",
        f"Local PDFs: {', '.join(payload['pdfs'])}",
        "",
        "| Metric | Original RAG | FinDoc v2 |",
        "|---|---:|---:|",
    ]
    for name in ("recall_at_5", "mrr", "numerical_accuracy", "avg_latency_seconds"):
        baseline = metrics["original_rag"].get(name, "Not measured")
        v2 = metrics["findoc_v2"].get(name, "Not measured")
        lines.append(f"| {name} | {baseline} | {v2} |")
    lines += [
        "",
        "Supported/unsupported accuracy: Not measured by the retrieval-only benchmark.",
        "Citation correctness: measured through page/table provenance in per-query rows.",
        "",
        "## Failure Analysis",
        "",
        "See `findoc_v2.json` for per-query retrieval failures and provenance.",
    ]
    os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run_evaluation() -> Dict:
    started = time.perf_counter()
    benchmark = run_benchmark(_default_pdfs(), top_k=5)
    elapsed = round(time.perf_counter() - started, 4)
    records = build_records(benchmark)
    summary = benchmark.get("summary", {})
    baseline = summary.get("dense-only", {})
    v2 = summary.get("table-aware", {})
    metrics = {
        "original_rag": {
            "recall_at_5": baseline.get("recall_at_5", "Not measured"),
            "mrr": baseline.get("mrr", "Not measured"),
            "numerical_accuracy": baseline.get("num_accuracy", "Not measured"),
            "avg_latency_seconds": "Not measured",
        },
        "findoc_v2": {
            "recall_at_5": v2.get("recall_at_5", "Not measured"),
            "mrr": v2.get("mrr", "Not measured"),
            "numerical_accuracy": v2.get("num_accuracy", "Not measured"),
            "avg_latency_seconds": elapsed,
        },
    }
    payload = {
        "dataset_size": len(records),
        "pdfs": benchmark.get("corpus", {}).get("pdfs", []),
        "metrics": metrics,
        "records": records,
        "benchmark": benchmark,
    }
    os.makedirs(os.path.dirname(_RESULTS), exist_ok=True)
    with open(_RESULTS, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    _write_report(payload)
    return payload


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result["metrics"], indent=2))
    print(f"Dataset records: {result['dataset_size']}")
