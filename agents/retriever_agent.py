from graph.state import GraphState
from retrieval.retriever import retrieve_chunks

def retrieve_content(sub_questions, vectorstore):
    all_chunks = []

    # If sub-questions are empty (shouldn't happen with fallback, but safety first)
    if not sub_questions:
        return []

    for q in sub_questions:
        # Reduced k from 15 to 6 to save LLM context window and increase speed
        chunks = retrieve_chunks(vectorstore, q, k=6)
        all_chunks.extend(chunks)

    # Deduplicate and trim chunks completely
    unique_chunks = []
    seen = set()
    for chunk in all_chunks:
        # Create a hashable and trimmed representation
        clean_text = " ".join(chunk['content'].split())  # Trim excess whitespaces
        identifier = f"{chunk['page_no']}_{clean_text[:50]}"
        
        if identifier not in seen:
            seen.add(identifier)
            chunk['content'] = clean_text # Store trimmed version
            unique_chunks.append(chunk)

    # Limit to top 6 most relevant chunks total to drastically reduce prompt size
    return unique_chunks[:6]
