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


def chunk_pages_excluding_tables(
    pages: List[Dict],
    chunk_size: int = 2000,
    chunk_overlap: int = 400,
    source_id: str = "",
):
    """
    Table-aware text chunking (Phase 3).

    Splits the *paragraph text* of each page but intentionally keeps tables OUT
    of the text chunks -- financial tables are handled by the TableStore as
    row-level units instead of being flattened into generic text chunks.

    Returns chunks with metadata: content, page_no, source_id, has_table=False,
    is_table=False. Existing ``chunk_financial_pages`` is untouched so the
    legacy dense-only pipeline keeps working unchanged.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = []
    for page in pages:
        text = page["text"]
        page_chunks = splitter.split_text(text) if text.strip() else []
        for c_idx, p_chunk in enumerate(page_chunks):
            if not p_chunk.strip():
                continue
            chunks.append(
                {
                    "content": p_chunk.strip(),
                    "page_no": page["page_no"],
                    "source_id": source_id,
                    "has_table": False,
                    "is_table": False,
                }
            )
    return chunks
