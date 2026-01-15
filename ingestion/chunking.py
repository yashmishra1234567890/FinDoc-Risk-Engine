from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_financial_pages(pages: List[Dict], chunk_size: int = 2000, chunk_overlap: int = 400):
    """
    Splits text using RecursiveCharacterTextSplitter to respect sentence/paragraph boundaries.
    Increased chunk size and overlap to capture full context (Company names, Headers + Data tables).
    """
    chunks = []
    
    # Configure the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

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

        # Split text into smart chunks
        page_chunks = splitter.split_text(text)
        
        for p_chunk in page_chunks:
            if p_chunk.strip():
                chunks.append({
                    "content": p_chunk.strip(),
                    "page_no": page["page_no"],
                    "has_table": len(page["tables"]) > 0
                })

    return chunks
