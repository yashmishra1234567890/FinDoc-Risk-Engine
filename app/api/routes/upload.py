import os
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import logging

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.indexer import create_documents_from_chunks
from app.api.core.config import DATA_DIR, VECTORSTORE_PATH
from app.api.core.store import vectorstore

router = APIRouter(tags=["Ingestion"])
logger = logging.getLogger(__name__)

def process_file_background(file_path: str):
    """
    Heavy lifting task that runs in the background.
    """
    try:
        logger.info(f"Starting background processing for: {file_path}")
        
        # 1. Load and Chunk
        pages = load_pdf(file_path)
        logger.info(f"Loaded {len(pages)} pages from {file_path}")
        
        chunks = chunk_financial_pages(pages)
        logger.info(f"Created {len(chunks)} chunks from {file_path}")

        # 2. Create Documents
        documents = create_documents_from_chunks(chunks)
        
        # 3. Add to Vector Store (Calls OpenAI Embeddings)
        # We assume vectorstore is thread-safe for simple adds or we accept slight race conditions impacting only concurrent uploads
        vectorstore.add_documents(documents)
        logger.info(f"Added {len(documents)} documents to vectorstore")

        # 4. Save to Disk (Optional backup)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info("Vectorstore saved successfully")

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")


@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Offload processing to background
        background_tasks.add_task(process_file_background, file_path)

        return {
            "message": "File uploaded successfully. Processing started in background.",
            "filename": file.filename,
            "note": "Large files may take 1-2 minutes to be available for query."
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
