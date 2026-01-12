import pdfplumber
from typing import Iterator, Dict


def load_text_pdf(path: str) -> Iterator[Dict]:
    """
    Extract text and tables page-wise from a financial PDF (Generator).
    Yields dicts with page_no, text, tables to save memory.
    """
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            yield {
                "page_no": i + 1,
                "text": text.strip(),
                "tables": tables
            }

