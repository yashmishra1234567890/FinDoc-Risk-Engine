from fastapi import APIRouter
from langchain_community.vectorstores import FAISS
import os

from app.api.schemas.request import QueryRequest
from app.api.schemas.response import QueryResponse
# from app.api.core.config import VECTORSTORE_PATH # Removed as logic moved to store.py
# from ingestion.embeddings import get_embedding_model # Removed
from app.api.core.store import vectorstore
from graph.graph import build_graph

router = APIRouter(tags=["Query"])

# Logic moved to app/api/core/store.py to be shared with upload endpoint
# embedder = get_embedding_model()
# ... 

graph_app = build_graph(vectorstore)


@router.post("/query", response_model=QueryResponse)
def query_financials(req: QueryRequest):
    state = {
        "user_query": req.question
    }

    result = graph_app.invoke(state)

    # Return the final answer
    return QueryResponse(
        answer=result["final_answer"],
        confidence=0.85
    )
