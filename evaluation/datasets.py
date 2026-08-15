"""
Dataset adapters for retrieval evaluation (FinQA, TAT-QA, T2-RAGBench).

None of these datasets ships with the repository. Instead of automatically
downloading large archives, this module provides *adapters* that load the
datasets when a local copy is placed on disk, and logs clearly when they are
absent.

When no external dataset is present, ``build_synthetic_queries`` derives a
small labeled query set from the *real* financial documents being benchmarked
(tables + paragraphs actually ingested), so every number reported by the
benchmark is a real measurement against real content -- never fabricated.
"""
import json
import logging
import os
from typing import Dict, List, Optional

from evaluation.metrics import normalize_number

logger = logging.getLogger(__name__)

# Overridable locations for datasets if a user provides them locally.
_DEFAULTS = {
    "finqa": os.getenv("FINQA_PATH", "data/datasets/finqa"),
    "tatqa": os.getenv("TATQA_PATH", "data/datasets/tatqa"),
    "t2ragbench": os.getenv("T2RAGBENCH_PATH", "data/datasets/t2ragbench"),
}

SUPPORTED = tuple(_DEFAULTS.keys())


def _read_jsonl(path: str) -> List[Dict]:
    records = []
    if os.path.isdir(path):
        candidates = sorted(
            f for f in os.listdir(path) if f.endswith(".jsonl") or f.endswith(".json")
        )
        path = os.path.join(path, candidates[0]) if candidates else ""
    if not path or not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _normalize_finqa(rec: Dict) -> Optional[Dict]:
    pre = rec.get("pre_text", "") or ""
    post = rec.get("post_text", "") or ""
    if isinstance(pre, list):
        pre = " ".join(pre)
    if isinstance(post, list):
        post = " ".join(post)
    table = rec.get("table")
    if isinstance(table, list):
        table_text = " | ".join(" ; ".join(str(c) for c in row) for row in table)
    else:
        table_text = ""
    context = f"{pre}\n{table_text}\n{post}".strip()
    return {
        "source": "finqa",
        "question": rec.get("question", ""),
        "answer": rec.get("answer", ""),
        "context": context,
        "relevant_ids": [],
    }


def _normalize_tatqa(rec: Dict) -> Optional[Dict]:
    context = ""
    for key in ("table", "paragraph", "title"):
        part = rec.get(key)
        if isinstance(part, list):
            context += " | ".join(" ; ".join(str(c) for c in row) for row in part) + "\n"
        elif isinstance(part, str):
            context += part + "\n"
    return {
        "source": "tatqa",
        "question": rec.get("question", ""),
        "answer": rec.get("answer", rec.get("answer_value", "")),
        "context": context.strip(),
        "relevant_ids": rec.get("relevant_ids", []),
    }


def _normalize_t2ragbench(rec: Dict) -> Optional[Dict]:
    return {
        "source": "t2ragbench",
        "question": rec.get("question", rec.get("query", "")),
        "answer": rec.get("answer", rec.get("gold_answer", "")),
        "context": rec.get("context", rec.get("table", "")),
        "relevant_ids": rec.get("relevant_ids", []),
    }


_NORMALIZERS = {
    "finqa": _normalize_finqa,
    "tatqa": _normalize_tatqa,
    "t2ragbench": _normalize_t2ragbench,
}


def load_dataset(name: str, path: Optional[str] = None) -> List[Dict]:
    """
    Load a local dataset via its adapter. Returns [] and logs when absent.

    ``name`` is one of: finqa, tatqa, t2ragbench.
    """
    if name not in _NORMALIZERS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {SUPPORTED}")
    target = path or _DEFAULTS[name]
    records = _read_jsonl(target)
    if not records:
        logger.info(
            "Dataset %r not found at %s -- skipping (no automatic download).",
            name, target,
        )
        return []
    normalized = []
    for rec in records:
        out = _NORMALIZERS[name](rec)
        if out and out["question"]:
            normalized.append(out)
    logger.info("Loaded %d %s records from %s", len(normalized), name, target)
    return normalized


def dataset_status() -> Dict[str, str]:
    """Report which external datasets are available locally."""
    status = {}
    for name in SUPPORTED:
        target = _DEFAULTS[name]
        present = os.path.isdir(target) and any(
            f.endswith((".jsonl", ".json")) for f in os.listdir(target)
        )
        status[name] = "present" if present else "not-present"
    return status


def build_synthetic_queries(
    tables: List[Dict],
    paragraphs: Optional[List[Dict]] = None,
    max_per_table: int = 2,
    max_paragraphs: int = 50,
) -> List[Dict]:
    """
    Build a labeled query set from the *real* ingested documents.

    Table queries: "What is {row-label} {col-label}?" -> the answer is the cell
    value; the relevant id is the row unit id.
    Paragraph queries: a leading clause of a paragraph chunk -> the relevant id
    is the paragraph chunk id.

    Every query records the PDF source, page and table that produced it so the
    benchmark can report exactly where each result came from.
    """
    queries: List[Dict] = []

    for table in tables:
        header = table.get("header") or []
        if not header or len(header) < 2:
            continue
        row_label_col = 0
        data_cols = range(1, len(header))
        for row in table.get("rows", [])[:max_per_table]:
            cells = row.get("cells", [])
            row_label = cells[row_label_col] if row_label_col < len(cells) else ""
            if not row_label or str(row_label).strip().startswith("col_"):
                continue
            for col in data_cols:
                if col >= len(cells):
                    continue
                value = cells[col]
                if normalize_number(value) is None:
                    continue  # numeric answer required
                col_header = header[col] if col < len(header) else ""
                question = f"What is the value of {row_label} for {col_header}?"
                queries.append(
                    {
                        "source": "synthetic-table",
                        "question": question,
                        "answer": value,
                        "relevant_ids": [
                            f"{table['table_id']}:r{row['row_idx']}"
                        ],
                        "pdf": table.get("source_id"),
                        "page_no": table.get("page_no"),
                        "table_id": table.get("table_id"),
                        "row_idx": row.get("row_idx"),
                    }
                )

    for chunk in (paragraphs or [])[:max_paragraphs]:
        text = (chunk.get("content") or "").strip()
        if len(text) < 20:
            continue
        tokens = text.split()
        # Use a distinctive leading clause as the query so the chunk is its
        # own ground-truth: the full chunk must be retrievable from its query.
        query = " ".join(tokens[:12])
        if len(query) < 10:
            continue
        queries.append(
            {
                "source": "synthetic-paragraph",
                "question": query,
                "answer": None,
                "relevant_ids": [chunk["id"]],
                "pdf": chunk.get("source_id"),
                "page_no": chunk.get("page_no"),
                "table_id": None,
                "row_idx": None,
            }
        )
    return queries