import pdfplumber
from typing import List, Dict


def load_text_pdf(path: str) -> List[Dict]:
    """
    Extract text and tables page-wise from a financial PDF.
    Returns a list of dicts with page_no, text, tables.
    """
    pages = []

    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            pages.append({
                "page_no": i + 1,
                "text": text.strip(),
                "tables": tables
            })

    return pages
