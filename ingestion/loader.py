"""
Data Ingestion Loader
---------------------
Handles loading of financial documents (PDFs).
Auto-detects scanned PDFs and switches to OCR if necessary.

"""
from ingestion.pdf_loader import load_text_pdf
from ingestion.ocr_loader import load_scanned_pdf
import logging

logger = logging.getLogger(__name__)

def load_pdf(path: str):
    # 1. Try standard text extraction
    pages = load_text_pdf(path)
    total_text = "".join(p["text"] for p in pages)

    # 2. Heuristic: If extremely little text, it might be a scanned image
    if len(total_text) < 100:
        logger.info(f"Low text detected ({len(total_text)} chars). Attempting OCR fallback...")
        try:
            # 3. Try OCR (Requires Tesseract/Poppler binaries on OS)
            return load_scanned_pdf(path)
        except Exception as e:
            logger.warning(f"OCR Failed (Likely missing system dependencies like tesseract/poppler): {e}")
            logger.warning("Falling back to raw text extraction (content may be empty).")
            # Return the original pages (even if empty) to prevent crashing the whole pipeline
            return pages

    return pages
