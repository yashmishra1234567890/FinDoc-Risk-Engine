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
    Scans the text for lines containing any of the provided keywords and extracts the last numerical value.
    
    Args:
        text (str): The text content from financial documents (tables/paragraphs).
        keywords (list): A list of strings to search for (e.g., ["total debt", "borrowings"]).
        
    Returns:
        float: The extracted value if found, otherwise None.
    """
    lines = text.split('\n')
    for line in lines:
        lower_line = line.lower()
        if any(k in lower_line for k in keywords):
            # Regex to find numbers like 1,000.00 or 500
            matches = re.findall(r"-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?", line)
            if matches:
                # We assume the last number in the row is the column value we want
                val_str = matches[-1]
                return parse_financial_value(val_str)
    return None

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

    # 1. Extract Key Metrics
    metrics = {
        "total_debt": extract_metric(combined_text, ["total debt", "total borrowings", "long term borrowings"]),
        "total_equity": extract_metric(combined_text, ["total equity", "shareholder's equity", "net worth"]),
        "current_liabilities": extract_metric(combined_text, ["current liabilities", "short term borrowings"]),
        "non_current_liabilities": extract_metric(combined_text, ["non-current liabilities", "long term liabilities"]),
        "EBITDA": extract_metric(combined_text, ["ebitda", "operating profit", "profit before tax"]),
        "interest_expense": extract_metric(combined_text, ["finance costs", "interest expense"])
    }

    # 2. Derive Ratios (Python Math - Deterministic)
    ratios = {
        "debt_to_equity": calculate_ratio(metrics["total_debt"], metrics["total_equity"]),
        "interest_coverage": calculate_ratio(metrics["EBITDA"], metrics["interest_expense"])
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
