"""
Phase 5 - Simple Agentic Retrieval: evidence sufficiency & query refinement.

Lightweight, deterministic logic layered on top of the existing retrieval
pipeline. After each retrieval the system decides whether the retrieved
evidence is sufficient to answer the user's question. If not, a focused
refined query is produced and retrieval runs again (maximum 3 attempts).

No new retrieval engine, no planner, no long-running loops. Reuses the chunk
metadata (content / page / source / scores) produced by the Phase 2/3 hybrid
retrieval pipeline.
"""
import os
import re
from typing import Dict, List, Optional

# Max total retrieval attempts before the graph proceeds with the best
# available evidence (configurable via .env).
MAX_RETRIEVAL_ATTEMPTS = int(os.getenv("AGENTIC_MAX_RETRIEVAL_ATTEMPTS", "3"))

_STOPWORDS = {
    "what", "was", "is", "were", "are", "the", "a", "an", "of", "to", "in",
    "for", "and", "or", "on", "by", "from", "with", "at", "as", "does", "do",
    "did", "has", "have", "how", "much", "many", "it", "its", "their", "this",
    "that", "please", "give", "me", "about", "between", "using",
}

_FINANCE_TERMS = [
    "revenue", "sales", "income", "ebitda", "ebit", "profit", "loss", "margin",
    "subscriber", "arpu", "debt", "borrowing", "equity", "asset", "liabilit",
    "cash", "capex", "depreciation", "tax", "interest", "expense", "opex",
    "receivable", "payable", "value", "share", "holding", "eps", "growth",
    "cagr", "ratio", "balance", "statement", "cost", "net", "gross",
]

_PERIOD_RE = re.compile(r"[0-9]{4}|q[1-4]|fy[0-9]{2}", re.I)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def significant_tokens(text: str) -> List[str]:
    """Lower-cased tokens from ``text`` minus stopwords and noise."""
    tokens = [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]
    # drop single-character noise but keep period/keys such as "q1"/"fy25"
    return [t for t in tokens if len(t) > 1 or t in ("q1", "q2", "q3", "q4")]


def _content_text(chunks: List[Dict]) -> str:
    parts = []
    for chunk in chunks or []:
        text = chunk.get("content") or chunk.get("text") or ""
        if text:
            parts.append(text.lower())
    return " ".join(parts)


def _num_evidence_chunks(chunks: List[Dict]) -> int:
    """Count chunks carrying any non-empty evidence text."""
    return sum(
        1 for c in (chunks or [])
        if (c.get("content") or c.get("text") or "").strip()
    )


def check_evidence_sufficiency(
    user_query: str,
    retrieved_chunks: Optional[List[Dict]],
    min_chunks: int = 1,
    coverage_threshold: float = 0.3,
) -> Dict:
    """
    Decide whether the retrieved evidence is sufficient to answer the question.

    Returns::

        {
            "sufficient": bool,
            "reason": str,
            "overlap_score": float,
            "num_chunks": int,
            "refined_query": str | None,
        }

    The decision is deliberately simple and deterministic: a question's
    significant tokens must be reasonably covered by the retrieved evidence and
    at least ``min_chunks`` chunks must carry content.
    """
    required = significant_tokens(user_query)
    content = _content_text(retrieved_chunks)
    num_chunks = _num_evidence_chunks(retrieved_chunks)

    covered = [t for t in required if t in content]
    coverage = len(covered) / len(required) if required else 1.0

    sufficient = num_chunks >= min_chunks and coverage >= coverage_threshold
    if sufficient:
        reason = f"evidence sufficient ({len(covered)}/{len(required)} tokens covered, {num_chunks} chunks)"
        refined_query: Optional[str] = None
    else:
        reason = f"evidence insufficient ({len(covered)}/{len(required)} tokens covered, {num_chunks} chunks)"
        refined_query = refine_query(user_query)

    return {
        "sufficient": sufficient,
        "reason": reason,
        "overlap_score": round(coverage, 3),
        "num_chunks": num_chunks,
        "required_tokens": required,
        "covered_tokens": covered,
        "refined_query": refined_query,
    }


def refine_query(user_query: str, evidence_check: Optional[Dict] = None) -> str:
    """
    Build a focused refinement of the original question for a retry.

    The refined query favours the financial metric terms and period tokens
    already present in the question (e.g. "subscriber", "q1fy25") so the next
    retrieval pass targets the same subject -- it never introduces unrelated
    sub-questions.
    """
    tokens = significant_tokens(user_query)
    if not tokens:
        return user_query

    def is_metric(t: str) -> bool:
        # substring match so "subscribers" counts under "subscriber"
        return any(m in t for m in _FINANCE_TERMS)

    metrics = [t for t in tokens if is_metric(t)]
    periods = [m.group(0).lower() for m in _PERIOD_RE.finditer(user_query)]
    remaining = [t for t in tokens if not is_metric(t) and not t in periods]

    focus = metrics + periods + remaining
    # de-duplicate, cap length
    focus = list(dict.fromkeys(focus))[:6]
    if not focus:
        return user_query
    return " ".join(focus)