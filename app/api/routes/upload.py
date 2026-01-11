import os
from fastapi import APIRouter, UploadFile, File

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.embeddings import get_embedding_model
from ingestion.indexer import build_faiss_index
from app.api.core.config import DATA_DIR, VECTORSTORE_PATH

router = APIRouter(tags=["Ingestion"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    pages = load_pdf(file_path)
    chunks = chunk_financial_pages(pages)

    embedder = get_embedding_model()
    vectorstore = build_faiss_index(chunks, embedder)

    os.makedirs("vectorstore", exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    return {
        "message": "Document ingested successfully",
        "pages": len(pages),
        "chunks": len(chunks)
    }
