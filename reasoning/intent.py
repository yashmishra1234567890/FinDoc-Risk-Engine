"""
Numerical reasoning intent detection (Phase 4).

Determines whether a user query requires numerical computation (as opposed to a
factual lookup or a non-numerical question). Rule-based, deterministic, and
LLM-free.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calculation-type keyword patterns. Order matters: first match wins.
# ---------------------------------------------------------------------------
_PATTERNS: List[tuple] = [
    ("cagr", re.compile(r"\bcagr\b", re.I)),
    ("yoy_growth", re.compile(r"year[-\s]over[-\s]year|\byoy\b", re.I)),
    ("qoq_growth", re.compile(r"quarter[-\s]over[-\s]quarter|\bqoq\b", re.I)),
    ("percentage_change", re.compile(
        r"(?:percent|percentage|%)\s*(?:change|growth|increase|decrease|decline|rise|fall)|"
        r"(?:change|growth|increase|decrease|decline|rise|fall)\s*(?:in|of).*(?:percent|%)", re.I)),
    ("percentage_of_total", re.compile(
        r"(?:what\s*)?(?:percent|percentage|proportion|share).*(?:of)\b", re.I)),
    ("margin", re.compile(r"\bmargin\b", re.I)),
    ("debt_ratio", re.compile(
        r"debt[-\s]to[-\s]equity|gearing|leverage|debt\s*ratio|solvency", re.I)),
    ("profitability", re.compile(
        r"\b(roa|roe|return on equity|return on assets|net margin|profit margin|"
        r"return on capital employed|roce)\b", re.I)),
    ("ratio", re.compile(r"\bratio\b|\bper\s+revenue\b", re.I)),
    ("average", re.compile(r"\baverage\b|\bmean\b", re.I)),
    ("sum", re.compile(r"\b(sum of|total of|combined)\b", re.I)),
    ("difference", re.compile(
        r"\b(difference\s*(?:between|in)|change in|change from|increase in|decrease in|"
        r"how much (?:did|has|have)|increase from|decline from|improved|rose|fell|"
        r"declined|grew|increased|decreased)\b", re.I)),
]

# Explicit arithmetic verbs that always require computation.
_STRONG_CALC_VERBS = re.compile(
    r"\b(calculate|compute|determine|what\s*is\s*the\s*(?:%|percentage|ratio|margin|growth))"
    r"\b", re.I)

# Neutral wording that usually indicates a factual lookup.
_FACTUAL_MARKERS = re.compile(
    r"\b(what was|what is|what were|who|which company|provide|list|describe|"
    r"explain|give me|does the company offer)\b", re.I)

# Questions asking for a single reported figure (factual, not computed).
_NO_CALC_NUMBERS = re.compile(
    r"\b(what (?:was|is|were|are) the (?:revenue|sales|profit|subscribers|arpu|"
    r"debt|borrowings|ebitda|assets|liabilities|margin|ratio|eps|cash))\b", re.I)

# Compound / multi-step calculations.
_MULTI_STEP_MARKERS = re.compile(
    r"\band\s+how\s+(?:did|does|that|this)|compare|relative to|over\s+time", re.I)
@dataclass
class IntentResult:
    requires_calculation: bool
    calculation_type: Optional[str] = None
    confidence: float = 0.0
    matched_pattern: Optional[str] = None
    multi_step: bool = False
    metric_hints: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requires_calculation": self.requires_calculation,
            "calculation_type": self.calculation_type,
            "confidence": round(self.confidence, 3),
            "matched_pattern": self.matched_pattern,
            "multi_step": self.multi_step,
            "metric_hints": self.metric_hints,
            "reasons": self.reasons,
        }


def metric_hints(question: str) -> List[str]:
    """Return financial metric terms mentioned in the question."""
    q = question.lower()
    return [term for term in METRIC_LEXICON if term in q]


def detect_intent(question: str) -> IntentResult:
    """Classify the question as requiring numerical computation or not."""
    if not question or not question.strip():
        return IntentResult(requires_calculation=False, confidence=0.0, reasons=["empty question"])

    q = question.strip()

    # 1. Strong arithmetic verbs -> definitely a calculation
    if _STRONG_CALC_VERBS.search(q):
        calc_type, conf, matched = _match_pattern(q)
        return IntentResult(
            requires_calculation=True,
            calculation_type=calc_type,
            confidence=max(conf, 0.9),
            matched_pattern=matched,
            multi_step=bool(_MULTI_STEP_MARKERS.search(q)),
            metric_hints=metric_hints(q),
            reasons=[f"strong calculation verb matched ({matched or 'verb'})"],
        )

    # 2. Explicit calculation-type pattern (checked before the factual
    #    fallback so "what was the revenue difference between..." is treated
    #    as a calculation, not a plain lookup).
    calc_type, conf, matched = _match_pattern(q)
    if calc_type:
        return IntentResult(
            requires_calculation=True,
            calculation_type=calc_type,
            confidence=conf,
            matched_pattern=matched,
            multi_step=bool(_MULTI_STEP_MARKERS.search(q)),
            metric_hints=metric_hints(q),
            reasons=[f"calculation pattern matched: {matched}"],
        )

    # 3. Pure factual lookup of a single reported figure -> no calculation
    if _NO_CALC_NUMBERS.match(q):
        return IntentResult(
            requires_calculation=False,
            confidence=0.9,
            metric_hints=metric_hints(q),
            reasons=["factual single-figure lookup"],
        )

    # 4. Factual wording with no growth/change/% signal
    if _FACTUAL_MARKERS.search(q) and not any(
        w in q.lower() for w in ("growth", "change", "increase", "decrease", "difference", "%")
    ):
        return IntentResult(
            requires_calculation=False,
            confidence=0.7,
            metric_hints=metric_hints(q),
            reasons=["factual wording, no calculation signal"],
        )

    return IntentResult(
        requires_calculation=False,
        confidence=0.5,
        metric_hints=metric_hints(q),
        reasons=["no numerical computation signal detected"],
    )


def _match_pattern(q: str):
    """Return (calculation_type, confidence, matched_pattern) for the question."""
    for calc_type, pattern in _PATTERNS:
        m = pattern.search(q)
        if m:
            conf = 0.85 if calc_type in ("yoy_growth", "qoq_growth", "cagr", "margin") else 0.8
            return calc_type, conf, f"{calc_type}:{m.group(0)}"
    return None, 0.0, None

# Financial metric terms used to describe what is being measured.
METRIC_LEXICON = [
    "revenue", "sales", "income", "ebitda", "ebit", "profit", "loss", "margin",
    "subscriber", "arpu", "debt", "borrowing", "equity", "asset", "liabilit",
    "cash", "capex", "depreciation", "tax", "interest", "expense", "opex",
    "receivable", "payable", "value", "share", "holding", "eps",
]