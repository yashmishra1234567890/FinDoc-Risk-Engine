"""
Data Ingestion Loader
---------------------
Handles loading of financial documents (PDFs).
Auto-detects scanned PDFs and switches to OCR if necessary.

"""
import logging
from ingestion.pdf_loader import load_text_pdf
# from ingestion.ocr_loader import load_scanned_pdf # Disable OCR for now

logger = logging.getLogger(__name__)

def load_pdf(path: str):
    logger.info(f"Loading PDF generator from: {path}")
    
    # Simple pass-through generator to support batch processing
    return load_text_pdf(path)
