"""
Financial Analysis Agent
------------------------
Extracts specific financial metrics (Debt, Revenue, etc.) via Regex and Python Logic.
"""

import re
from graph.state import GraphState
from utils.finance_utils import parse_financial_value, calculate_ratio

def extract_metric(text: str, keywords: list) -> float:
    """
    Finds numeric values associated with keywords using regex.
    Prioritizes large currency figures (millions/crores) over small percentages.
    """
    lines = text.split('\n')
    best_val = None
    
    for line in lines:
        lower_line = line.lower()
        if any(k in lower_line for k in keywords):
            # Regex to find numbers like 1,000.00 or 500
            matches = re.findall(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", line)
            
            if matches:
                 vals = []
                 for m in matches:
                     try:
                        v = parse_financial_value(m)
                        vals.append(v)
                     except:
                        pass
                 
                 if vals:
                    # Heuristic: Prefer large absolute numbers (Currency) over small ones (Ratios/Notes/%)
                    large_vals = [v for v in vals if v > 1000]
                    if large_vals:
                         best_val = large_vals[-1] # Usually most recent year is last column
                    else:
                         best_val = vals[-1] # Fallback

    return best_val

def analyze_financials(retrieved_chunks: list, user_query: str) -> dict:
    """
    Core Logic: Extracts raw metrics -> computes ratios -> reports missing data.
    """
    combined_text = "\n".join(chunk["content"] for chunk in retrieved_chunks)

    # 1. Regex Extraction of Core Metrics
    metrics = {
        "total_debt": extract_metric(combined_text, ["total debt", "total borrowings", "long term borrowings"]),
        "total_equity": extract_metric(combined_text, ["total equity", "shareholder's equity", "net worth"]),
        "current_liabilities": extract_metric(combined_text, ["current liabilities", "short term borrowings"]),
        "non_current_liabilities": extract_metric(combined_text, ["non-current liabilities", "long term liabilities"]),
        "EBITDA": extract_metric(combined_text, ["ebitda", "operating profit", "profit before tax"]),
        "interest_expense": extract_metric(combined_text, ["finance costs", "interest expense"]),
    }
    
    # 2. Context-Aware Extraction (Depends on User Query)
    query_lower = user_query.lower()
    if "revenue" in query_lower or "sales" in query_lower:
        metrics["revenue"] = extract_metric(combined_text, ["revenue from operations", "total revenue", "revenue"])
    
    if "profit" in query_lower or "net income" in query_lower:
        metrics["net_profit"] = extract_metric(combined_text, ["net profit", "profit for the period", "net income"])
        
    if "cash flow" in query_lower:
        metrics["cash_flow"] = extract_metric(combined_text, ["cash flow from operating", "net cash from operating"])

    # 3. Deterministic Ratio Calculation (Python Math)
    ratios = {
        "debt_to_equity": calculate_ratio(metrics.get("total_debt"), metrics.get("total_equity")),
        "interest_coverage": calculate_ratio(metrics.get("EBITDA"), metrics.get("interest_expense"))
    }

    # 4. Metadata for Confidence Scoring
    missing = [k for k, v in metrics.items() if v is None]

    return {
        "extracted_metrics": metrics,
        "derived_ratios": ratios,
        "missing_metrics": missing,
        "pages_used": list({chunk["page_no"] for chunk in retrieved_chunks})
    }
