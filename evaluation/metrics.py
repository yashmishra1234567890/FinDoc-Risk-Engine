"""
Evaluation Metrics Module
-------------------------
This module defines functions to calculate performance metrics for the financial agent.
How to run (import usage):
    from evaluation.metrics import calculate_latency, check_keywords

"""
import time
from typing import List

def calculate_latency(start_time: float, end_time: float) -> float:
    """Calculates execution time in seconds."""
    return round(end_time - start_time, 2)

def check_keyword_presence(answer: str, keywords: List[str]) -> float:
    """
    Checks what percentage of expected financial keywords appear in the answer.
    Returns a score between 0.0 and 1.0.
    """
    if not answer or not keywords:
        return 0.0
    
    answer_lower = answer.lower()
    found = sum(1 for k in keywords if k.lower() in answer_lower)
    return round(found / len(keywords), 2)


# ---------------------------------------------------------------------------
# Retrieval metrics (Phase 2 / Phase 3 evaluation)
# ---------------------------------------------------------------------------
import re as _re


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = 5) -> float:
    """Recall@k: fraction of relevant items present in the top-k retrieved ids."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hit = sum(1 for rid in relevant_ids if rid in top_k)
    return hit / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR over the first relevant item (1/rank) for a single query."""
    relevant = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


_NUM_PATTERN = _re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number(text: str):
    """
    Extract a comparable float from a financial cell string.

    Handles '45,200,000', '(18,100,000)', '+18.0%', '1,234.56', '-35' etc.
    Returns None when the text holds no usable number.
    """
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]
    s = s.replace(",", "").replace("\u00a0", "").replace(" ", "")
    m = _NUM_PATTERN.search(s)
    if not m:
        return None
    try:
        value = float(m.group())
    except ValueError:
        return None
    return -value if negative else value


def numerical_accuracy(answer_cell: str, evidence_texts: List[str]) -> float:
    """
    Numerical accuracy: 1.0 if the numeric value of ``answer_cell`` appears in
    any of the ``evidence_texts``, else 0.0. Non-numeric answers return 0.0.
    """
    expected = normalize_number(answer_cell)
    if expected is None:
        return 0.0
    for text in evidence_texts:
        for token in str(text or "").split():
            found = normalize_number(token)
            if found is not None and abs(found - expected) < 1e-6:
                return 1.0
    return 0.0
