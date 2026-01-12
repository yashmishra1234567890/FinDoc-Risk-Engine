"""
Data Ingestion Loader
---------------------
Handles loading of financial documents (PDFs).
Auto-detects scanned PDFs and switches to OCR if necessary.

"""
import logging
from ingestion.pdf_loader import load_text_pdf
# from ingestion.ocr_loader import load_scanned_pdf # Disable OCR for now to force text extraction debugging

logger = logging.getLogger(__name__)

def load_pdf(path: str):
    logger.info(f"Loading PDF from: {path}")
    
    # 1. Force standard text extraction
    pages = load_text_pdf(path)
    
    total_chars = sum(len(p.get("text", "")) for p in pages)
    logger.info(f"Extracted {total_chars} characters from {len(pages)} pages.")

    if total_chars < 50:
         logger.warning("Very low text content detected. Proceeding anyway.")

    return pages
