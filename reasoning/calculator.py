"""
Numerical calculator (Phase 4).

A small, deterministic calculator that performs ONLY the supported financial
operations using explicit Python arithmetic. No eval(), no AST execution, no
generated-code execution, no sandbox.

Supported operations:
    yoy_growth, percentage_change, difference, margin, ratio,
    average, cagr, percentage_of_total, sum, debt_to_equity

Every operation returns a structured dict::

    {
        "success": bool,
        "value": float | None,
        "formula": str,
        "operands": {...},
        "unit": str | None,
        "error": str | None,
    }
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SUPPORTED_OPERATIONS = {
    "yoy_growth",
    "percentage_change",
    "difference",
    "margin",
    "ratio",
    "average",
    "cagr",
    "percentage_of_total",
    "sum",
    "debt_to_equity",
}


def _numbers(values: Dict[str, float], *keys: str) -> List[Optional[float]]:
    """Fetch numeric operands by key, returning None when missing/invalid."""
    out = []
    for key in keys:
        value = values.get(key)
        if value is None:
            out.append(None)
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _check_not_none(operands: List[Optional[float]], names: List[str]) -> Dict:
    missing = [name for name, value in zip(names, operands) if value is None]
    if missing:
        return {
            "success": False,
            "value": None,
            "formula": "",
            "operands": dict(zip(names, operands)),
            "unit": None,
            "error": f"missing operand(s): {', '.join(missing)}",
        }
    return {}


def calculate(operation: str, values: Dict[str, float], unit: Optional[str] = None) -> Dict:
    """
    Execute one of the supported operations on the given named numeric values.

    ``values`` must be a dict of validated floats keyed by operand name, e.g.::

        {"current": 105083, "previous": 106555}

    Returns the calculation result plus the formula used.
    """
    logger.debug("calculate(%s, %s)", operation, values)
    if operation not in SUPPORTED_OPERATIONS:
        return {
            "success": False,
            "value": None,
            "formula": "",
            "operands": values,
            "unit": unit,
            "error": f"unsupported operation '{operation}'",
        }

    # --- difference: new - old
    if operation == "difference":
        cur, prev = _numbers(values, "current", "previous")
        err = _check_not_none([cur, prev], ["current", "previous"])
        if err:
            return err
        value = cur - prev
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"{cur} - {prev}",
            "operands": {"current": cur, "previous": prev},
            "unit": unit,
            "error": None,
        }

    # --- percentage_change / yoy_growth / qoq_growth:
    #     ((new - old) / old) * 100
    if operation in ("percentage_change", "yoy_growth", "qoq_growth"):
        cur, prev = _numbers(values, "current", "previous")
        err = _check_not_none([cur, prev], ["current", "previous"])
        if err:
            return err
        if prev == 0:
            return {
                "success": False,
                "value": None,
                "formula": f"(({cur} - {prev}) / {prev}) * 100",
                "operands": {"current": cur, "previous": prev},
                "unit": "%",
                "error": "division by zero (previous value is 0)",
            }
        value = ((cur - prev) / prev) * 100
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"(({cur} - {prev}) / {prev}) * 100",
            "operands": {"current": cur, "previous": prev},
            "unit": "%",
            "error": None,
        }

    # --- margin / percentage_of_total: (part / total) * 100
    if operation in ("margin", "percentage_of_total"):
        numerator, denominator = _numbers(values, "numerator", "denominator")
        err = _check_not_none([numerator, denominator], ["numerator", "denominator"])
        if err:
            return err
        if denominator == 0:
            return {
                "success": False,
                "value": None,
                "formula": f"({numerator} / {denominator}) * 100",
                "operands": {"numerator": numerator, "denominator": denominator},
                "unit": "%",
                "error": "division by zero (denominator is 0)",
            }
        value = (numerator / denominator) * 100
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"({numerator} / {denominator}) * 100",
            "operands": {"numerator": numerator, "denominator": denominator},
            "unit": "%",
            "error": None,
        }

    # --- ratio / debt_to_equity: numerator / denominator
    if operation in ("ratio", "debt_to_equity"):
        numerator, denominator = _numbers(values, "numerator", "denominator")
        err = _check_not_none([numerator, denominator], ["numerator", "denominator"])
        if err:
            return err
        if denominator == 0:
            return {
                "success": False,
                "value": None,
                "formula": f"{numerator} / {denominator}",
                "operands": {"numerator": numerator, "denominator": denominator},
                "unit": unit,
                "error": "division by zero (denominator is 0)",
            }
        value = numerator / denominator
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"{numerator} / {denominator}",
            "operands": {"numerator": numerator, "denominator": denominator},
            "unit": unit,
            "error": None,
        }

    # --- average: sum(values) / len(values)
    if operation == "average":
        nums = [v for v in values.values() if v is not None]
        if not nums:
            return {
                "success": False,
                "value": None,
                "formula": "avg()",
                "operands": values,
                "unit": unit,
                "error": "no numeric values to average",
            }
        value = sum(nums) / len(nums)
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"sum({nums}) / {len(nums)}",
            "operands": dict(values),
            "unit": unit,
            "error": None,
        }

    # --- sum
    if operation == "sum":
        nums = [v for v in values.values() if v is not None]
        if not nums:
            return {
                "success": False,
                "value": None,
                "formula": "sum()",
                "operands": values,
                "unit": unit,
                "error": "no numeric values to sum",
            }
        value = sum(nums)
        return {
            "success": True,
            "value": round(value, 6),
            "formula": " + ".join(str(n) for n in nums),
            "operands": dict(values),
            "unit": unit,
            "error": None,
        }

    # --- cagr: ((end / begin) ** (1/years)) - 1 (expressed as %)
    if operation == "cagr":
        end, begin, years = _numbers(values, "end", "begin", "years")
        err = _check_not_none([end, begin, years], ["end", "begin", "years"])
        if err:
            return err
        if begin <= 0:
            return {
                "success": False,
                "value": None,
                "formula": f"(({end} / {begin}) ** (1 / {years})) - 1",
                "operands": {"end": end, "begin": begin, "years": years},
                "unit": "%",
                "error": "beginning value must be positive for CAGR",
            }
        if years <= 0:
            return {
                "success": False,
                "value": None,
                "formula": f"(({end} / {begin}) ** (1 / {years})) - 1",
                "operands": {"end": end, "begin": begin, "years": years},
                "unit": "%",
                "error": "number of years must be positive",
            }
        value = ((end / begin) ** (1 / years) - 1) * 100
        return {
            "success": True,
            "value": round(value, 6),
            "formula": f"(({end} / {begin}) ** (1 / {years})) - 1",
            "operands": {"end": end, "begin": begin, "years": years},
            "unit": "%",
            "error": None,
        }

    return {
        "success": False,
        "value": None,
        "formula": "",
        "operands": values,
        "unit": unit,
        "error": f"unsupported operation '{operation}'",
    }