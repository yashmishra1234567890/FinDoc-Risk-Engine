"""
Numerical reasoning engine (Phase 4).

Orchestrates the full reasoning flow for a single question:

    question
      -> intent detection
      -> bind values from retrieved evidence
      -> calculate (deterministic, safe ops only)
      -> verify (optional reported answer support)
      -> final answer + provenance

The engine never executes LLM-generated code; it only calls the explicit
operations implemented in :mod:`reasoning.calculator`.
"""
import logging
import math
from typing import Dict, List, Optional

from config.numerical_config import NUMERICAL_VERIFY_TOLERANCE
from reasoning.binding import bind_values_from_chunks, BoundVariable
from reasoning.calculator import calculate
from reasoning.intent import detect_intent

logger = logging.getLogger(__name__)

# Map intent calculation types to the calculator operation names.
TYPE_TO_OPERATION = {
    "yoy_growth": "yoy_growth",
    "qoq_growth": "yoy_growth",        # same formula
    "percentage_change": "percentage_change",
    "difference": "difference",
    "margin": "margin",
    "percentage_of_total": "percentage_of_total",
    "ratio": "ratio",
    "debt_ratio": "debt_to_equity",
    "average": "average",
    "sum": "sum",
    "cagr": "cagr",
}


def verify_result(computed: float, reported: Optional[float],
                  tolerance: float = NUMERICAL_VERIFY_TOLERANCE) -> Dict:
    """Compare a computed result against a reported value within tolerance.

    Returns::

        {"status": "consistent" | "discrepancy" | "not-checked",
         "computed_value", "reported_value", "difference", "tolerance"}
    """
    if reported is None:
        return {
            "status": "not-checked",
            "computed_value": computed,
            "reported_value": None,
            "difference": None,
            "tolerance": tolerance,
        }
    try:
        reported_f = float(reported) if not isinstance(reported, (int, float)) else reported
    except (TypeError, ValueError):
        return {
            "status": "not-checked",
            "computed_value": computed,
            "reported_value": reported,
            "difference": None,
            "tolerance": tolerance,
        }
    difference = abs(computed - reported_f) if computed is not None else None
    status = "consistent" if difference is not None and difference <= tolerance else "discrepancy"
    return {
        "status": status,
        "computed_value": computed,
        "reported_value": reported_f,
        "difference": difference,
        "tolerance": tolerance,
    }


def _select_operands(operation: str, variables: List) -> Dict[str, float]:
    """Pick the operands required by ``operation`` from bound variables.

    Variables whose name matches the operand key (current/previous/numerator/
    denominator/begin/end) are used first; otherwise the last two values in
    evidence order are used as current/previous.
    """
    by_name = {v.name: v.value for v in variables if v.value is not None}
    values = [v.value for v in variables if v.value is not None]

    if operation in ("difference", "percentage_change", "yoy_growth", "qoq_growth"):
        if "current" in by_name and "previous" in by_name:
            return {"current": by_name["current"], "previous": by_name["previous"]}
        if len(values) >= 2:
            return {"current": values[-1], "previous": values[-2]}
        return {"current": values[0]} if len(values) == 1 else {}

    if operation in ("margin", "percentage_of_total", "ratio", "debt_to_equity"):
        if "numerator" in by_name and "denominator" in by_name:
            return {"numerator": by_name["numerator"], "denominator": by_name["denominator"]}
        if len(values) >= 2:
            a, b = values[0], values[1]
            if abs(a) <= abs(b):
                return {"numerator": a, "denominator": b}
            return {"numerator": b, "denominator": a}
        return {"numerator": values[0]} if len(values) == 1 else {}

    if operation in ("average", "sum"):
        return {f"value{i+1}": v for i, v in enumerate(values)} if values else {}

    if operation == "cagr":
        if "begin" in by_name and "end" in by_name:
            return {"begin": by_name["begin"], "end": by_name["end"], "years": by_name.get("years")}
        if len(values) >= 2:
            return {"begin": values[0], "end": values[-1], "years": None}
        return {}

    return {}


def _missing_operands(operation: str, operands: Dict[str, object]) -> List[str]:
    """Return the operand keys required by ``operation`` that are absent/None."""
    expected = {
        "difference": ("current", "previous"),
        "percentage_change": ("current", "previous"),
        "yoy_growth": ("current", "previous"),
        "qoq_growth": ("current", "previous"),
        "margin": ("numerator", "denominator"),
        "percentage_of_total": ("numerator", "denominator"),
        "ratio": ("numerator", "denominator"),
        "debt_to_equity": ("numerator", "denominator"),
        "cagr": ("begin", "end", "years"),
    }.get(operation, ())
    return [k for k in expected if k not in operands or operands.get(k) is None]


def _make_explicit(name: str, value: float) -> BoundVariable:
    """Wrap an explicit variable into a BoundVariable."""
    return BoundVariable(name=name, value=float(value), unit=None)


def run_numerical_reasoning(
    question: str,
    chunks: Optional[List[Dict]] = None,
    reported_value: Optional[float] = None,
    explicit_variables: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Run the full numerical reasoning pipeline for a question.

    Args:
        question: the user question (used for intent detection).
        chunks: retrieved evidence rows (content / source_id / page_no / ...).
        reported_value: optional known answer (e.g. from the document
            narrative) used for tolerance verification.
        explicit_variables: optional pre-bound variable dict to use directly
            instead of extracting from chunks (for tests / integrations where
            retrieval is not available).

    Returns a structured dict with intent, variables, formula, result,
    verification and a natural-language answer.
    """
    intent = detect_intent(question)
    operation = TYPE_TO_OPERATION.get(intent.calculation_type or "")
    if not intent.requires_calculation or operation is None:
        return {
            "requires_calculation": False,
            "intent": intent.to_dict(),
            "answer": None,
            "reason": "non-numerical or unsupported question; using standard path",
        }

    # 1. Bind values (explicit variables override extraction)
    if explicit_variables:
        variables = [_make_explicit(name, value) for name, value in explicit_variables.items()]
    else:
        variables = bind_values_from_chunks(chunks or [])

    # 2. Select operands
    operands = _select_operands(operation, variables)
    if not operands:
        return {
            "requires_calculation": True,
            "intent": intent.to_dict(),
            "calculation_type": intent.calculation_type,
            "answer": None,
            "result": None,
            "variables": {v.name: v.value for v in variables},
            "error": "no numeric values found in retrieved evidence",
            "missing": "operands",
            "verification": None,
        }

    # CAGR needs a year count; default to 1 when not derivable.
    if operation == "cagr" and operands.get("years") is None:
        operands["years"] = 1.0

    # 3. Calculate
    calc = calculate(operation, operands)
    if not calc.get("success"):
        missing_keys = _missing_operands(operation, operands)
        return {
            "requires_calculation": True,
            "intent": intent.to_dict(),
            "calculation_type": intent.calculation_type,
            "answer": None,
            "result": None,
            "variables": {v.name: v.value for v in variables},
            "operands": operands,
            "error": calc.get("error"),
            "missing": missing_keys or None,
            "verification": None,
        }

    # 4. Runtime sanity checks (no clamping - report honestly)
    value = calc.get("value")
    runtime_issues = []
    if value is None or not isinstance(value, (int, float)):
        runtime_issues.append("non-numeric result")
    else:
        if math.isnan(value):
            runtime_issues.append("NaN result")
        if math.isinf(value):
            runtime_issues.append("infinite result")
    runtime_ok = not runtime_issues

    # 5. Verification
    verification = verify_result(value, reported_value)

    # 6. Provenance + natural answer
    proof = _build_provenance(operation, operands, calc, variables, verification, runtime_issues)
    answer = _build_narrative(operation, operands, calc, verification)

    return {
        "requires_calculation": True,
        "intent": intent.to_dict(),
        "calculation_type": intent.calculation_type,
        "operation": operation,
        "result": value,
        "unit": calc.get("unit"),
        "formula": calc.get("formula"),
        "operands": operands,
        "variables": {v.name: v.value for v in variables},
        "runtime_ok": runtime_ok,
        "runtime_issues": runtime_issues,
        "verification": verification,
        "proof": proof,
        "answer": answer,
    }
def _build_provenance(operation: str, operands: Dict, calc: Dict, variables: List,
                      verification: Dict, runtime_issues: List[str]) -> List[Dict]:
    """Return a list of provenance steps describing the calculation."""
    var_entries = [
        {
            "name": v.name,
            "value": v.value,
            "unit": v.unit,
            "period": v.period,
            "source_id": v.source_id,
            "page_no": v.page_no,
            "table_id": v.table_id,
            "citation": v.citation,
        }
        for v in variables
    ]
    steps = [
        {"step": "operation", "operation": operation, "formula": calc.get("formula") or ""},
        {"step": "operands", "operands": operands},
        {"step": "variables", "variables": var_entries},
        {"step": "execution", "engine": "reasoning.calculator (explicit Python arithmetic)"},
        {"step": "result", "value": calc.get("value")},
        {"step": "verification", "status": verification.get("status"),
         "reported_value": verification.get("reported_value")},
    ]
    if runtime_issues:
        steps.append({"step": "runtime_warnings", "issues": runtime_issues})
    return steps


def _build_narrative(operation: str, operands: Dict, calc: Dict, verification: Dict) -> str:
    """Build a short natural-language answer with the calculation shown."""
    value = calc.get("value")
    unit = calc.get("unit") or ""
    formula = calc.get("formula") or ""

    if operation == "difference":
        lead = f"The difference is {value:,.2f} {unit}."
    elif operation == "average":
        lead = f"The average is {value:,.2f} {unit}."
    elif operation == "sum":
        lead = f"The total is {value:,.2f} {unit}."
    elif operation == "cagr":
        lead = f"The CAGR is {value:,.2f}%."
    elif operation in ("yoy_growth", "percentage_change"):
        lead = f"The percentage change is {value:,.2f}%."
    elif operation in ("margin", "percentage_of_total"):
        lead = f"The result is {value:,.2f}%."
    elif operation == "ratio":
        lead = f"The ratio is {value:,.4f}."
    else:
        lead = f"The result is {value:,.4f} {unit}."

    narrative = f"{lead}\nCalculation: {formula}"
    status = verification.get("status") if isinstance(verification, dict) else None
    if status == "consistent":
        narrative += (
            f"\nVerification: consistent with reported value "
            f"~{verification.get('reported_value')}."
        )
    elif status == "discrepancy":
        narrative += (
            f"\nVerification: discrepancy vs reported value "
            f"{verification.get('reported_value')} "
            f"(difference {verification.get('difference')})."
        )
    elif status == "not-checked":
        narrative += "\nVerification: not checked (no reported value in evidence)."
    return narrative