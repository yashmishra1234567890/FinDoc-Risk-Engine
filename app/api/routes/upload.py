import os
import uuid
from typing import Dict
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
import logging

from ingestion.loader import load_pdf
from ingestion.chunking import chunk_financial_pages
from ingestion.indexer import create_documents_from_chunks
from app.api.core.config import DATA_DIR, VECTORSTORE_PATH
from app.api.core.store import vectorstore, reset_vectorstore

router = APIRouter(tags=["Ingestion"])
logger = logging.getLogger(__name__)

# Global Status Tracking
upload_status: Dict[str, dict] = {}

def process_file_background(file_path: str, task_id: str):
    """
    Heavy lifting task that runs in the background.
    Optimized for low memory usage (batch processing).
    """
    try:
        logger.info(f"Starting background processing for task {task_id}: {file_path}")
        
        # 0. RESET STORE (Fix for stale data issue)
        reset_vectorstore()
        logger.info("Vector store reset for new document.")
        
        # 1. Load Generator (Lazy loading)
        pages_generator = load_pdf(file_path)
        
        # 2. Process in Batches of 50 pages to save RAM
        BATCH_SIZE = 50
        batch_pages = []
        
        for page in pages_generator:
            batch_pages.append(page)
            if len(batch_pages) >= BATCH_SIZE:
                 _process_batch(batch_pages)
                 batch_pages = [] # Clear memory
        
        # Process remaining
        if batch_pages:
             _process_batch(batch_pages)

        # 4. Save to Disk (Once at the end)
        os.makedirs("vectorstore", exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info("Vectorstore saved successfully")
        
        # Update Status
        upload_status[task_id] = {"status": "completed", "message": "Indexing successful"}

    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        upload_status[task_id] = {"status": "failed", "message": str(e)}

def _process_batch(pages):
    chunks = chunk_financial_pages(pages)
    documents = create_documents_from_chunks(chunks)
    vectorstore.add_documents(documents)
    logger.info(f"Processed batch of {len(pages)} pages.")



@router.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        logger.info(f"Receiving file upload: {file.filename}")
        os.makedirs(DATA_DIR, exist_ok=True)
        file_path = os.path.join(DATA_DIR, file.filename)

        # Stream write to avoid loading entire file into RAM (helps with low-memory environments like Render)
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024) # Read 1MB at a time
                if not chunk:
                    break
                f.write(chunk)
        
        logger.info(f"File saved to disk: {file_path}. Queuing background processing.")

        # Generate ID
        task_id = str(uuid.uuid4())
        upload_status[task_id] = {"status": "processing", "message": "Indexing in progress..."}

        # Offload processing to background
        background_tasks.add_task(process_file_background, file_path, task_id)

        return {
            "message": "File uploaded successfully. Processing started.",
            "task_id": task_id,
            "filename": file.filename
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/upload/status/{task_id}")
async def get_upload_status(task_id: str):
    status = upload_status.get(task_id)
    if not status:
         return {"status": "unknown", "message": "Task not found"}
    return status
