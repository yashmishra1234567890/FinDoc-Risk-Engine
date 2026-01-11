import sys
import os

# Add project root to sys.path to allow imports from ingestion, retrieval, etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.embeddings import get_embedding_model
from ingestion.indexer import build_faiss_index
from retrieval.retriever import retrieve_chunks


PDF_PATH = "data/financial_docs/Zomato_Annual_Report_2022-23.pdf"


def run_phase1_test():
    pages = load_pdf(PDF_PATH)
    print(f"Pages loaded: {len(pages)}")

    chunks = chunk_financial_pages(pages)
    print(f"Chunks created: {len(chunks)}")

    embedder = get_embedding_model()
    vectorstore = build_faiss_index(chunks, embedder)

    query = "total debt and liabilities"
    results = retrieve_chunks(vectorstore, query)

    print("\nTop Results:\n")
    for r in results:
        print(f"[Page {r['page_no']}] {r['content'][:200]}...\n")


if __name__ == "__main__":
    run_phase1_test()
