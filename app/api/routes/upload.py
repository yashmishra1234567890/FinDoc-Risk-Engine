import os
from fastapi import APIRouter, UploadFile, File

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.indexer import create_documents_from_chunks
from app.api.core.config import DATA_DIR, VECTORSTORE_PATH
from app.api.core.store import vectorstore

router = APIRouter(tags=["Ingestion"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages = load_pdf(file_path)
    chunks = chunk_financial_pages(pages)

    # Create documents and add to the running global vectorstore
    documents = create_documents_from_chunks(chunks)
    
    # This updates the in-memory index used by query.py
    vectorstore.add_documents(documents)

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return {
        "message": "Document ingested successfully",
        "pages": len(pages),
        "chunks": len(chunks)
    }
