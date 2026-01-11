from pdf2image import convert_from_path
import pytesseract
from typing import List, Dict


def load_scanned_pdf(path: str) -> List[Dict]:
    images = convert_from_path(path)
    pages = []

    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)

        pages.append({
            "page_no": i + 1,
            "text": text.strip(),
            "tables": []
        })

    return pages
