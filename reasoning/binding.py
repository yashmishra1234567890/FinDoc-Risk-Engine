"""
Numerical reasoning variable/evidence binding (Phase 4).

Binds retrieved text/table evidence to the structured variable representation
produced by the intent detector. Every variable traceability links back to a
specific document, page, table, and cell.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured variable representation (produced after binding).
# ---------------------------------------------------------------------------
@dataclass
class BoundVariable:
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    period: Optional[str] = None
    source_id: Optional[str] = None
    page_no: Optional[int] = None
    table_id: Optional[str] = None
    row_idx: Optional[int] = None
    column_idx: Optional[int] = None
    citation: Optional[dict] = None
    raw_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"[\(\[\-]?[\d,]+(?:\.\d+)?[\)\]]?%?$")


def _strip_units(text: str) -> Tuple[Optional[float], Optional[str]]:
    s = text.strip()
    t = s.replace("Rs.", "").replace("Rs", "").strip()
    t = t.rstrip("% ()")
    m = _NUM_RE.search(t)
    if not m:
        return None, None
    raw_num = m.group()
    try:
        val = float(m.group().replace(",", ""))
    except ValueError:
        return None, None
    unit_text = s[len(m.group()):].strip()
    unit = None
    if unit_text:
        if re.search(r"\b(mn|billion|million)\b", unit_text, re.I):
            unit = "Rs mn"
        elif re.search(r"\bbn\b", unit_text, re.I):
            unit = "Rs bn"
        elif re.search(r"\b%\b", unit_text):
            unit = "%"
        else:
            unit = unit_text
    return val, unit


def _is_numeric_cell(cell_text: str) -> bool:
    t = cell_text.strip()
    if not t:
        return False
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    t = t.replace(",", "").replace(" ", "")
    return bool(_NUM_RE.match(t))


# ---------------------------------------------------------------------------
# Text extraction helpers used by binding routines.
# ---------------------------------------------------------------------------
def _extract_all_text(evidence: dict) -> List[str]:
    """Return every printable string found anywhere in the evidence dict."""
    out: List[str] = []
    for _k, v in evidence.items():
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            out.extend(_extract_all_text(v))
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    out.extend(_extract_all_text(item))
    return out


def _extract_clean_text(evidence: dict) -> str:
    """Concatenate all text fields into a single string."""
    parts = []
    for _k, v in evidence.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, dict):
            sub = _extract_clean_text(v)
            if sub:
                parts.append(sub)
    return " | ".join(parts) if parts else ""


def make_bound(name: str, value: float, unit: Optional[str] = None,
               source: dict = None, extra_period: Optional[str] = None) -> BoundVariable:
    """Convenience constructor preserving citation/source provenance."""
    return BoundVariable(
        name=name,
        value=value,
        unit=unit,
        period=extra_period,
        source_id=(source or {}).get("source_id") if source else None,
        page_no=(source or {}).get("page_no") if source else None,
        table_id=(source or {}).get("table_id") if source else None,
        citation=(source or {}).get("citation") if source else None,
    )


# ---------------------------------------------------------------------------
# Value extraction from a single evidence chunk/table row.
# ---------------------------------------------------------------------------
def extract_numeric_pairs(text: str) -> List[Tuple[float, Optional[str]]]:
    """Return (value, unit) pairs in the order they appear in ``text``.

    Only matches numbers that are NOT glued to other alphanumerics (avoids
    picking up "1" from "Q1FY25" or "24" from "FY24"). Trailing unit words
    such as "mn", "billion", "%" are captured when present.
    """
    pairs: List[Tuple[float, Optional[str]]] = []
    # A number must be delimited: preceded by start/space/( ; - and followed by
    # a space, comma-with-space, %, or a unit keyword.
    pattern = re.compile(
        r"(?<![A-Za-z0-9_])([\d,]+(?:\.\d+)?)\s*(%|Rs\.?|mn|bn|million|billion|"
        r"million\s*subscribers|subscribers)?",
        re.I,
    )
    for mtch in pattern.finditer(text):
        raw = mtch.group(1)
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        unit_text = (mtch.group(2) or "").strip()
        unit = None
        if unit_text:
            unit = (
                "Rs mn" if unit_text.lower() in ("mn", "million")
                else "Rs bn" if unit_text.lower() in ("bn", "billion")
                else "%" if unit_text == "%"
                else unit_text
            )
        pairs.append((value, unit))
    return pairs


def bind_values_from_chunks(chunks: List[dict]) -> List[BoundVariable]:
    """
    Collect all numeric values found across retrieved chunks/rows.

    Each value is turned into a variable with citation provenance pointing at
    the chunk it came from (page / source / table when available).
    """
    bound: List[BoundVariable] = []
    for chunk in chunks or []:
        source = {
            "source_id": chunk.get("source_id") or chunk.get("document_id"),
            "page_no": chunk.get("page_no"),
            "table_id": chunk.get("table_id"),
            "citation": chunk.get("citation"),
        }
        text = chunk.get("content") or chunk.get("text") or ""
        for value, unit in extract_numeric_pairs(text):
            bound.append(make_bound(name=f"v{len(bound)+1}", value=value, unit=unit, source=source))
    return bound