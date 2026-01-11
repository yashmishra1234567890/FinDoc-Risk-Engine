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
