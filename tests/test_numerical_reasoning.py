"""
Phase 4 - Numerical reasoning tests.

Covers the intent detector, the safe deterministic calculator, the reasoning
engine (binding -> calculate -> verify -> provenance), and a graph routing
smoke test. All tests are offline: no LLM, no API key, no network.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reasoning.calculator import calculate
from reasoning.engine import run_numerical_reasoning, verify_result
from reasoning.intent import detect_intent, metric_hints
from reasoning.binding import extract_numeric_pairs


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
def test_intent_yoy_growth():
    result = detect_intent("What was Vodafone Idea's YoY revenue growth in Q1FY25?")
    assert result.requires_calculation is True
    assert result.calculation_type == "yoy_growth"


def test_intent_margin():
    result = detect_intent("What was Vodafone Idea's EBITDA margin in Q1FY25?")
    assert result.requires_calculation is True
    assert result.calculation_type == "margin"


def test_intent_difference():
    result = detect_intent("How much did 4G subscribers increase from Q1FY24 to Q1FY25?")
    assert result.requires_calculation is True
    assert result.calculation_type == "difference"


def test_intent_factual_lookup_no_calculation():
    result = detect_intent("What was Vodafone Idea's revenue in Q1FY25?")
    assert result.requires_calculation is False


def test_intent_non_numerical():
    result = detect_intent("What products does the company offer?")
    assert result.requires_calculation is False


def test_intent_average():
    result = detect_intent("What is the average ARPU over the four quarters?")
    assert result.requires_calculation is True
    assert result.calculation_type == "average"


def test_intent_cagr():
    result = detect_intent("Calculate the CAGR of revenue over the last three years.")
    assert result.requires_calculation is True
    assert result.calculation_type == "cagr"


def test_metric_hints():
    hints = metric_hints("What was the YoY revenue growth?")
    assert "revenue" in hints


# ---------------------------------------------------------------------------
# Calculator - supported operations
# ---------------------------------------------------------------------------
def test_calc_yoy_growth():
    res = calculate("yoy_growth", {"current": 105083, "previous": 106555})
    assert res["success"] is True
    assert abs(res["value"] - -1.381446) < 1e-3
    assert res["unit"] == "%"
    assert "105083.0" in res["formula"]


def test_calc_difference():
    res = calculate("difference", {"current": 126.7, "previous": 122.9})
    assert res["success"] is True
    assert abs(res["value"] - 3.8) < 1e-6


def test_calc_margin():
    res = calculate("margin", {"numerator": 42047, "denominator": 105083})
    assert res["success"] is True
    assert abs(res["value"] - 40.013) < 1e-2


def test_calc_ratio():
    res = calculate("ratio", {"numerator": 42047, "denominator": 105083})
    assert res["success"] is True
    assert abs(res["value"] - 0.4001) < 1e-3


def test_calc_average():
    res = calculate("average", {"value1": 10, "value2": 20, "value3": 30})
    assert res["success"] is True
    assert res["value"] == 20.0


def test_calc_cagr():
    res = calculate("cagr", {"begin": 100, "end": 121, "years": 2})
    assert res["success"] is True
    assert abs(res["value"] - 10.0) < 1e-6


def test_calc_sum():
    res = calculate("sum", {"value1": 1, "value2": 2, "value3": 3})
    assert res["success"] is True
    assert res["value"] == 6.0


def test_calc_division_by_zero():
    res = calculate("yoy_growth", {"current": 5, "previous": 0})
    assert res["success"] is False
    assert "division by zero" in res["error"]


def test_calc_missing_operand():
    res = calculate("difference", {"current": 5})
    assert res["success"] is False
    assert "previous" in res["error"]


def test_calc_unsupported_operation():
    res = calculate("import os; os.system('x')", {"a": 1})
    assert res["success"] is False
    assert "unsupported operation" in res["error"]


def test_calc_negative_begin_cagr():
    res = calculate("cagr", {"begin": -5, "end": 10, "years": 2})
    assert res["success"] is False


# ---------------------------------------------------------------------------
# Value binding / extraction
# ---------------------------------------------------------------------------
def test_extract_numeric_pairs_ignores_period_tokens():
    text = "Revenue from Operations Q1FY24: 106,555 Rs mn ; Q1FY25: 105,083 Rs mn"
    values = [v for v, _ in extract_numeric_pairs(text)]
    # '1' from Q1FY24/Q1FY25 or '24'/'25' must not be extracted
    assert 106555 in values and 105083 in values
    assert 1 not in values
    assert 24 not in values
    assert 25 not in values


def test_extract_numeric_pairs_with_units():
    values = [(v, u) for v, u in extract_numeric_pairs("ARPU improved to Rs. 146 vs Rs. 139")]
    assert 146.0 in [v for v, _ in values]
    assert 139.0 in [v for v, _ in values]


# ---------------------------------------------------------------------------
# Engine end-to-end
# ---------------------------------------------------------------------------
def test_engine_yoy_with_reported_verification():
    res = run_numerical_reasoning(
        "What was Vodafone Idea's YoY revenue growth in Q1FY25?",
        explicit_variables={"current": 105083.0, "previous": 106555.0},
        reported_value=-1.4,
    )
    assert res["requires_calculation"] is True
    assert abs(res["result"] - -1.381446) < 1e-3
    assert res["verification"]["status"] == "consistent"
    assert "Calculation:" in res["answer"]


def test_engine_margin_from_chunks():
    chunks = [
        {
            "content": "EBITDA Q1FY25: 42,047 Rs mn ; Revenue from Operations: 11,083 Rs mn",
            "page_no": 11,
            "source_id": "VIL-QR-Q1FY25.pdf",
        }
    ]
    res = run_numerical_reasoning(
        "What percentage of Q1FY25 revenue was EBITDA?",
        chunks=chunks,
    )
    assert res["requires_calculation"] is True
    assert res["result"] is not None


def test_engine_non_numerical_returns_standard_path():
    res = run_numerical_reasoning("What products does the company offer?", chunks=[])
    assert res["requires_calculation"] is False
    assert res["answer"] is None
    assert "standard path" in res["reason"]


def test_engine_missing_values_returns_failure():
    res = run_numerical_reasoning(
        "What was the YoY revenue growth?",
        explicit_variables={"current": 500.0},
    )
    assert res["requires_calculation"] is True
    assert res["answer"] is None
    assert res.get("missing") is not None


def test_engine_provenance():
    res = run_numerical_reasoning(
        "What was the revenue difference between Q1FY25 and Q1FY24?",
        explicit_variables={"current": 105083.0, "previous": 106555.0},
    )
    assert res["requires_calculation"] is True
    assert isinstance(res["proof"], list)
    steps = [s["step"] for s in res["proof"]]
    assert "operation" in steps
    assert "variables" in steps
    assert "result" in steps


def test_engine_result_is_finite():
    res = run_numerical_reasoning(
        "What is the ratio of EBITDA to revenue?",
        explicit_variables={"numerator": 42047.0, "denominator": 105083.0},
    )
    assert res["runtime_ok"] is True
    assert res["result"] is not None
    import math
    assert math.isfinite(res["result"])


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def test_verify_consistent():
    v = verify_result(-1.381446, -1.4)
    assert v["status"] == "consistent"


def test_verify_discrepancy():
    v = verify_result(5.04, 4.5)
    assert v["status"] == "discrepancy"


def test_verify_not_checked():
    v = verify_result(5.04, None)
    assert v["status"] == "not-checked"