"""
Table-aware extraction (Phase 3)
---------------------------------
Financial tables are extracted from pdfplumber's list-of-rows structure and
normalized so that headers, rows and cells are preserved independently instead
of being flattened into one generic embedding.

Target representation::

    Table
      |-- header (one row of column labels)
      |-- row <row_idx>
      |     |-- cell(col_0)
      |     |-- cell(col_1)
      |     `-- cell(col_N)
      `-- row ...

Every table is assigned a stable ``table_id`` and every row a ``row_id`` so
page number, source/document id, table id and row/column information are
retained for citation. Nothing is reduced to a single table-level embedding.
"""
import re
from typing import Dict, List, Optional

from retrieval.bm25 import tokenize  # shared pure-python tokenizer

_NUM_RE = re.compile(r"^[\(\[\-]?[\d,]+(?:\.\d+)?[\)\]]?%?$")


def is_numeric(value: Optional[str]) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    t = s.replace(",", "").replace(" ", "").lstrip("+-")  # allow +18.0% / -45
    return bool(_NUM_RE.match(t))


def make_table_id(source_id: str, page_no: int, table_idx: int) -> str:
    return f"{source_id}:p{page_no}:t{table_idx}"


def make_row_id(table_id: str, row_idx: int) -> str:
    return f"{table_id}:r{row_idx}"


def _split_header(rows: List[List[str]]) -> tuple:
    """Heuristically split header row from data rows.

    The first row is treated as the header when it contains mostly non-numeric
    labels (typical of financial statements). If the first row is fully
    numeric we synthesize column labels and treat every row as data.
    """
    if not rows:
        return [], []
    first = rows[0]
    non_empty = [c for c in first if str(c).strip()]
    if non_empty:
        numeric_ratio = sum(1 for c in non_empty if is_numeric(c)) / len(non_empty)
    else:
        numeric_ratio = 1.0
    if non_empty and numeric_ratio <= 0.5:
        return [str(c) for c in first], rows[1:]
    width = max((len(r) for r in rows), default=0)
    return [f"col_{i}" for i in range(width)], rows


def normalize_table(raw_rows: List[List], page_no: int, source_id: str,
                    table_idx: int, max_rows: Optional[int] = None) -> Dict:
    """
    Normalise a pdfplumber table (list of lists) into the canonical structure.

    Returns a dict with source_id, page_no, table_id, header, rows and
    row/column counts. ``None`` cells are converted to empty strings.
    """
    if not raw_rows:
        raw_rows = []
    rows_clean = [[(c if c is not None else "") for c in (r or [])] for r in raw_rows]
    rows_clean = [r for r in rows_clean if any(str(c).strip() for c in r)]

    header, data_rows = _split_header(rows_clean)

    if max_rows is not None and max_rows > 0:
        data_rows = data_rows[:max_rows]

    width = max((len(r) for r in data_rows), default=(len(header) if header else 0))

    rows = [
        {
            "row_idx": ri,
            "cells": [str(c) for c in r] + [""] * (width - len(r)),
        }
        for ri, r in enumerate(data_rows, start=1)
    ]

    table_id = make_table_id(source_id, int(page_no), table_idx)
    return {
        "source_id": source_id,
        "page_no": int(page_no),
        "table_id": table_id,
        "table_idx": table_idx,
        "header": [str(c) for c in header][:width],
        "rows": rows,
        "column_count": width,
        "row_count": len(rows),
    }


def row_to_unit(table: Dict, row: Dict) -> Dict:
    """Build a retrievable row-level representation preserving header + cells."""
    header = table["header"]
    cells = row["cells"]
    labels = [header[i] if i < len(header) else f"col_{i}" for i in range(len(cells))]
    line = " ; ".join(f"{labels[i]}: {cells[i]}" for i in range(len(cells)) if str(cells[i]).strip())
    text = f"[table {table['table_id']} on page {table['page_no']}] {line}"
    return {
        "id": make_row_id(table["table_id"], row["row_idx"]),
        "source_id": table["source_id"],
        "page_no": table["page_no"],
        "table_id": table["table_id"],
        "row_idx": row["row_idx"],
        "header": labels,
        "cells": cells,
        "text": text,
        "metadata": {
            "source_id": table["source_id"],
            "page_no": table["page_no"],
            "table_id": table["table_id"],
            "row_idx": row["row_idx"],
            "is_table": True,
            "has_table": True,
            "header": labels,
            "cells": cells,
        },
    }


def find_cell(query: str, table: Dict) -> Dict:
    """Best-effort cell-level evidence: which row/column answers a query.

    Scores each row by its strongest *data* column: token overlap between the
    query and (column header + cell text) plus a bonus for numeric cells, so
    the value cell is returned instead of the row-label cell.
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return {}
    best = None
    best_score = -1.0
    for row in table["rows"]:
        cells = row["cells"]
        header = table["header"]
        labels = [header[i] if i < len(header) else f"col_{i}" for i in range(len(cells))]
        best_col = 0
        col_score = -1.0
        for ci, cell in enumerate(cells):
            label_toks = set(tokenize(labels[ci])) if ci < len(labels) else set()
            overlap = len(q_tokens & (label_toks | set(tokenize(cell))))
            # numeric bonus steers the pick towards data columns, not the label
            score = overlap + (1.5 if is_numeric(cell) else 0.0)
            if score > col_score:
                col_score = score
                best_col = ci
        if col_score > best_score:
            best_score = col_score
            best = {
                "table_id": table["table_id"],
                "page_no": table["page_no"],
                "source_id": table["source_id"],
                "row_idx": row["row_idx"],
                "column_idx": best_col,
                "column_header": labels[best_col] if best_col < len(labels) else None,
                "value": cells[best_col] if best_col < len(cells) else None,
                "score": round(best_score, 2),
            }
    return best or {}

