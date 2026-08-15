"""
Citation helper for retrieved evidence.

Builds a stable, human/API friendly citation (source_id, page, table, row,
column header) from a retrieval result so that page numbers, source ids, table
ids and row/column information are always preserved and surfaced.
"""


def make_citation(entry: dict) -> dict:
    """Extract a citation dict from a retrieval result entry."""
    meta = entry.get("metadata") or {}

    def field(*names):
        for n in names:
            val = entry.get(n)
            if val is not None:
                return val
            val = meta.get(n)
            if val is not None:
                return val
        return None

    column_header = None
    header = field("header")
    best_col = None
    if isinstance(header, list) and header:
        best_col = field("best_column")
        if isinstance(best_col, int) and 0 <= best_col < len(header):
            column_header = header[best_col]
        elif best_col is None and len(header) >= 1:
            # fall back to the first (label) column
            column_header = header[0]

    cells = field("cells")
    value = None
    if isinstance(cells, list) and isinstance(best_col, int) and 0 <= best_col < len(cells):
        value = cells[best_col]

    return {
        "source_id": field("source_id", "document_id", "doc_id") or entry.get("id"),
        "page_no": field("page_no"),
        "table_id": field("table_id"),
        "row_idx": field("row_idx"),
        "column_header": column_header,
        "value": value,
        "is_table": bool(field("is_table", "has_table")),
    }