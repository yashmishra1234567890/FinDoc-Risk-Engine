from fastapi import APIRouter
from langchain_community.vectorstores import FAISS

from app.api.schemas.request import QueryRequest
from app.api.schemas.response import QueryResponse
from app.api.core.config import VECTORSTORE_PATH
from ingestion.embeddings import get_embedding_model
from graph.graph import build_graph

router = APIRouter(tags=["Query"])

# load once at startup
embedder = get_embedding_model()
vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedder, allow_dangerous_deserialization=True)
graph_app = build_graph(vectorstore)


@router.post("/query", response_model=QueryResponse)
def query_financials(req: QueryRequest):
    state = {
        "user_query": req.question
    }

    result = graph_app.invoke(state)

    return QueryResponse(
        answer=result.final_answer,
        confidence=0.85
    )
