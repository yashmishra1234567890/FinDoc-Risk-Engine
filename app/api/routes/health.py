from fastapi import APIRouter
from app.api.core.store import vectorstore

router = APIRouter(tags=["Health"])

@router.get("/health")
def health():
    # Return basic health plus vector store stats
    doc_count = 0
    try:
        # FAISS implementation details
        doc_count = vectorstore.index.ntotal
    except Exception:
        pass
        
    return {
        "status": "ok", 
        "documents_indexed": doc_count
    }

