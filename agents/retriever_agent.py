from graph.state import GraphState
from retrieval.retriever import retrieve_chunks

def retrieve_content(sub_questions, vectorstore):
    all_chunks = []

    # If sub-questions are empty (shouldn't happen with fallback, but safety first)
    if not sub_questions:
        return []

    for q in sub_questions:
        # Increase k to 15 to ensure we capture relevant financial tables which might be far down the list
        chunks = retrieve_chunks(vectorstore, q, k=15)
        all_chunks.extend(chunks)

    # Deduplicate chunks based on content to avoid processing same text twice
    unique_chunks = []
    seen = set()
    for chunk in all_chunks:
        # Create a hashable representation (using page_no and content snippet)
        identifier = f"{chunk['page_no']}_{chunk['content'][:50]}"
        if identifier not in seen:
            seen.add(identifier)
            unique_chunks.append(chunk)

    return unique_chunks
