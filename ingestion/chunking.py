from typing import List, Dict


def chunk_financial_pages(pages: List[Dict], max_chars: int = 800):
    chunks = []

    for page in pages:
        text = page["text"]

        # Append structured table data if available
        if page.get("tables"):
            for table in page["tables"]:
                # Convert list of lists to simplistic markdown/csv format
                table_str = "\n".join(
                    [" | ".join((str(cell) if cell is not None else "") for cell in row)
                     for row in table if row]
                )
                text += f"\n\n[TABLE START]\n{table_str}\n[TABLE END]\n"

        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars].strip()
            if chunk:
                chunks.append({
                    "content": chunk,
                    "page_no": page["page_no"],
                    "has_table": len(page["tables"]) > 0
                })

    return chunks
