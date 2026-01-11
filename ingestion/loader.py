"""
Data Ingestion Loader
---------------------
Handles loading of financial documents (PDFs).
Auto-detects scanned PDFs and switches to OCR if necessary.

"""
from ingestion.pdf_loader import load_text_pdf
from ingestion.ocr_loader import load_scanned_pdf


def load_pdf(path: str):
    pages = load_text_pdf(path)

    total_text = "".join(p["text"] for p in pages)

    # Heuristic: scanned PDF detection
    if len(total_text) < 100:
        return load_scanned_pdf(path)

    return pages
