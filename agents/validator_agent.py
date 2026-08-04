import logging
import os
import re

import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def get_market_data():
    try:
        vix = float(yf.Ticker("^VIX").history(period="5d")["Close"].dropna().iloc[-1])
    except Exception:
        vix = 15.0
    try:
        rate = float(yf.Ticker("^TNX").history(period="5d")["Close"].dropna().iloc[-1])
    except Exception:
        rate = 4.0
    return vix, rate


def _normalize_company_name(name: str | None) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", " ", name).strip().lower()


def _resolve_sec_company(company_name: str | None):
    if not company_name:
        return None

    headers = {
        "User-Agent": os.getenv("SEC_USER_AGENT", "FinDocRiskEngine contact@example.com"),
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }

    try:
        tickers = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers, timeout=10)
        tickers.raise_for_status()
        ticker_rows = tickers.json().values()
        target = _normalize_company_name(company_name)
        for row in ticker_rows:
            title = _normalize_company_name(row.get("title"))
            ticker = str(row.get("ticker", "")).lower()
            if target in title or title in target or ticker in target:
                return row
    except Exception as exc:
        logger.info("SEC company lookup failed: %s", exc)
    return None


def _extract_sec_facts(company_name: str | None):
    company = _resolve_sec_company(company_name)
    if not company:
        return None

    cik = str(company.get("cik_str", "")).zfill(10)
    headers = {
        "User-Agent": os.getenv("SEC_USER_AGENT", "FinDocRiskEngine contact@example.com"),
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }

    try:
        response = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=headers, timeout=10)
        response.raise_for_status()
        facts = response.json().get("facts", {}).get("us-gaap", {})

        def _latest_value(keys):
            for key in keys:
                units = facts.get(key, {}).get("units", {})
                for records in units.values():
                    if records:
                        latest = sorted(records, key=lambda item: item.get("fy", 0))[-1]
                        return float(latest.get("val"))
            return None

        return {
            "company": company.get("title"),
            "ticker": company.get("ticker"),
            "revenue": _latest_value([
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
            ]),
            "net_income": _latest_value(["NetIncomeLoss"]),
        }
    except Exception as exc:
        logger.info("SEC companyfacts lookup failed: %s", exc)
        return None


def validate_analysis(analysis_result, company_ticker=None, user_query: str | None = None):
    """
    Applies deterministic risk rules plus federated cross-checks against PDF, Yahoo Finance, and SEC EDGAR.
    """
    metrics = analysis_result.get("extracted_metrics", {})
    ratios = analysis_result.get("derived_ratios", {})
    missing = analysis_result.get("missing_metrics", [])
    company_name = analysis_result.get("company_name") or user_query

    rule_flags = []
    federated_sources = {"pdf": True, "yahoo_finance": False, "sec_edgar": False}

    de_ratio = ratios.get("debt_to_equity")
    if de_ratio is not None:
        if de_ratio > 2.33:
            rule_flags.append(f"High risk: debt-to-equity is {de_ratio:.2f}")
        elif de_ratio > 1.5:
            rule_flags.append(f"Medium risk: debt-to-equity is {de_ratio:.2f}")
        else:
            rule_flags.append(f"Low risk: debt-to-equity is {de_ratio:.2f}")
    else:
        rule_flags.append("Debt-to-equity data missing")

    ic_ratio = ratios.get("interest_coverage")
    if ic_ratio is not None:
        if ic_ratio < 1.5:
            rule_flags.append(f"High risk: interest coverage is {ic_ratio:.2f}")
        elif ic_ratio < 2.5:
            rule_flags.append(f"Medium risk: interest coverage is {ic_ratio:.2f}")
        else:
            rule_flags.append(f"Low risk: interest coverage is {ic_ratio:.2f}")
    else:
        rule_flags.append("Interest coverage data missing")

    vix, rate = get_market_data()
    federated_sources["yahoo_finance"] = True
    rule_flags.append(f"Market context: VIX={vix:.2f}, 10Y yield={rate:.2f}%")

    temperature_risk = 0.0
    if vix > 25 or rate > 5.0:
        temperature_risk = 0.3
        rule_flags.append("Temperature-aware risk elevated by market volatility.")
    elif vix > 15:
        temperature_risk = 0.15
        rule_flags.append("Temperature-aware risk moderately elevated.")

    sec_snapshot = _extract_sec_facts(company_name)
    if sec_snapshot:
        federated_sources["sec_edgar"] = True
        sec_revenue = sec_snapshot.get("revenue")
        pdf_revenue = metrics.get("revenue")
        if sec_revenue is not None and pdf_revenue is not None:
            delta = abs(pdf_revenue - sec_revenue) / max(abs(sec_revenue), 1.0)
            if delta < 0.2:
                rule_flags.append("SEC cross-check: revenue is consistent with EDGAR records.")
            else:
                rule_flags.append("SEC cross-check: revenue differs materially from EDGAR records.")
        rule_flags.append(f"SEC cross-check loaded for {sec_snapshot.get('company', 'unknown company')}.")
    else:
        rule_flags.append("SEC cross-check unavailable for this document.")

    total_expected = 6
    found_count = total_expected - len(missing)
    base_confidence = max(0.0, min(1.0, found_count / total_expected))
    confidence_score = round(max(0.05, min(0.99, base_confidence - temperature_risk + (0.1 if sec_snapshot else 0.0))), 2)

    assessment = "Automated Python Validation: " + " | ".join(rule_flags)

    return {
        "rule_engine_flags": rule_flags,
        "confidence_score": confidence_score,
        "assessment": assessment,
        "market_context": {"vix": vix, "ten_year_yield": rate},
        "temperature_risk": round(temperature_risk, 2),
        "federated_sources": federated_sources,
        "sec_snapshot": sec_snapshot,
    }
