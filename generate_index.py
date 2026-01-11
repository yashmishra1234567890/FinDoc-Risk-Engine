import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.embeddings import get_embedding_model
from ingestion.indexer import build_faiss_index

PDF_PATH = "data/financial_docs/Zomato_Annual_Report_2022-23.pdf"
INDEX_PATH = "vectorstore/faiss_index"

def generate_index():
    print("Loading PDF...")
    pages = load_pdf(PDF_PATH)
    
    print("Chunking...")
    chunks = chunk_financial_pages(pages)
    
    print("Embedding...")
    embedder = get_embedding_model()
    
    print("Building Index...")
    vectorstore = build_faiss_index(chunks, embedder)
    
    print(f"Saving index to {INDEX_PATH}...")
    vectorstore.save_local(INDEX_PATH)
    print("Done!")

if __name__ == "__main__":
    generate_index()
