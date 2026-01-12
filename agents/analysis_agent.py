"""
Financial Analysis Agent
------------------------
This module handles the extraction of specific financial metrics from text and calculates
key financial ratios. It uses a combination of keyword search and regex to find values
like Debt, Equity, and EBITDA.
"""

import re
from graph.state import GraphState
from utils.finance_utils import parse_financial_value, calculate_ratio

def extract_metric(text: str, keywords: list) -> float:
    """
    Scans the text for lines containing any of the provided keywords.
    Prioritizes large numbers (likely currency) over small numbers (percentages like 100.0).
    """
    lines = text.split('\n')
    best_val = None
    
    for line in lines:
        lower_line = line.lower()
        if any(k in lower_line for k in keywords):
            # Regex to find numbers like 1,000.00 or 500
            matches = re.findall(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", line)
            if matches:
                 # Filter out clearly non-financial years (e.g. 2024, 2025) if they stand alone, 
                 # but honestly, standard financial tables put values at the end.
                 
                 # Strategy: Look for the largest value in the line that isn't a percentage.
                 # Usually, Revenue columns are absolute numbers (millions/crores).
                 
                 vals = []
                 for m in matches:
                     try:
                        v = parse_financial_value(m)
                        vals.append(v)
                     except:
                        pass
                 
                 if vals:
                    # Heuristic: If we are looking for Revenue/Debt, we usually want the big Number, not "100.0" (%) or "1" (Note 1)
                    # Exclude values that look like percentages (0-100) IF there is a much larger number available
                    large_vals = [v for v in vals if v > 1000]
                    if large_vals:
                         best_val = large_vals[-1] # Usually most recent year is last column
                    else:
                         best_val = vals[-1] # Fallback to last number found

    return best_val

def analyze_financials(retrieved_chunks: list, user_query: str) -> dict:
    """
    Orchestrates the analysis by extracting raw metrics and computing derived ratios.
    
    Args:
        retrieved_chunks (list): The list of text chunks found by the Retriever.
        user_query (str): The original question (used for context if needed).
        
    Returns:
        dict: A structured dictionary containing raw metrics, calculated ratios, and metadata.
    """
    combined_text = "\n".join(
        chunk["content"] for chunk in retrieved_chunks
    )

    # 1. Extract Key Metrics (STANDARD RISK METRICS)
    metrics = {
        "total_debt": extract_metric(combined_text, ["total debt", "total borrowings", "long term borrowings"]),
        "total_equity": extract_metric(combined_text, ["total equity", "shareholder's equity", "net worth"]),
        "current_liabilities": extract_metric(combined_text, ["current liabilities", "short term borrowings"]),
        "non_current_liabilities": extract_metric(combined_text, ["non-current liabilities", "long term liabilities"]),
        "EBITDA": extract_metric(combined_text, ["ebitda", "operating profit", "profit before tax"]),
        "interest_expense": extract_metric(combined_text, ["finance costs", "interest expense"]),
    }
    
    # 2. Extract DYNAMIC Metrics based on User Query
    user_query_lower = user_query.lower()
    if "revenue" in user_query_lower or "sales" in user_query_lower:
        metrics["revenue"] = extract_metric(combined_text, ["revenue from operations", "total revenue", "revenue"])
    
    if "profit" in user_query_lower or "net income" in user_query_lower:
        metrics["net_profit"] = extract_metric(combined_text, ["net profit", "profit for the period", "net income"])
        
    if "cash flow" in user_query_lower:
        metrics["cash_flow"] = extract_metric(combined_text, ["cash flow from operating", "net cash from operating"])

    # 3. Derive Ratios (Python Math - Deterministic)
    ratios = {
        "debt_to_equity": calculate_ratio(metrics.get("total_debt"), metrics.get("total_equity")),
        "interest_coverage": calculate_ratio(metrics.get("EBITDA"), metrics.get("interest_expense"))
    }

    # 3. Identify Missing Data for Confidence Scoring
    missing = [k for k, v in metrics.items() if v is None]

    analysis_result = {
        "extracted_metrics": metrics,
        "derived_ratios": ratios,
        "missing_metrics": missing,
        "pages_used": list({chunk["page_no"] for chunk in retrieved_chunks})
    }

    return analysis_result
