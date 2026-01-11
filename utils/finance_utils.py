
import re

def parse_financial_value(text: str) -> float:
    """
    Parses strings like '1,234.56', '(500)', '100 Cr' into floats.
    Returns None if parsing fails.
    """
    if not text:
        return None
    
    # Remove whitespace
    clean_text = text.strip()
    
    # Handle brackets for negative numbers (accounting format)
    if clean_text.startswith('(') and clean_text.endswith(')'):
        clean_text = '-' + clean_text[1:-1]
        
    # Remove commas
    clean_text = clean_text.replace(',', '')
    
    try:
        return float(clean_text)
    except ValueError:
        return None

def calculate_ratio(numerator: float, denominator: float) -> float:
    """Safely calculates a ratio, returning None if denominator is 0 or invalid."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 2)
