from graph.state import GraphState
from retrieval.retriever import retrieve_chunks

def retrieve_content(sub_questions, vectorstore):
    all_chunks = []

    for q in sub_questions:
        chunks = retrieve_chunks(vectorstore, q, k=3)
        all_chunks.extend(chunks)

    return all_chunks
