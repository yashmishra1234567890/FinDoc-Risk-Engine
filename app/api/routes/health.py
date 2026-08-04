from fastapi import APIRouter
from app.api.core.store import get_vectorstore

router = APIRouter(tags=["Health"])

@router.get("/health")
def health():
    # Return basic health plus vector store stats
    doc_count = 0
    try:
        doc_count = get_vectorstore()._collection.count()
    except Exception:
        pass
        
    return {
        "status": "ok", 
        "documents_indexed": doc_count
    }

