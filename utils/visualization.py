import fitz
import base64
import os

def generate_highlighted_image(pdf_path, page_num, text_to_highlight):
    try:
        if not os.path.exists(pdf_path): return None
        doc = fitz.open(pdf_path)
        page_idx = max(0, page_num - 1)
        if page_idx >= len(doc): return None
        page = doc[page_idx]
        text_instances = page.search_for(text_to_highlight[:50])
        for inst in text_instances:
            highlight = page.add_highlight_annot(inst)
            highlight.update()
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_bytes = pix.tobytes("png")
        doc.close()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return "data:image/png;base64," + img_b64
    except Exception as e:
        print(f"Failed to generate highlight image: {e}")
        return None
