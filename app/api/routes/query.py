from fastapi import APIRouter
from langchain_community.vectorstores import FAISS
import os

from app.api.schemas.request import QueryRequest
from app.api.schemas.response import QueryResponse
from app.api.core.config import VECTORSTORE_PATH
from ingestion.embeddings import get_embedding_model
from graph.graph import build_graph

router = APIRouter(tags=["Query"])

# load once at startup
embedder = get_embedding_model()

# Handle missing vector store gracefully (e.g., in new environment)
try:
    print(f"Attempting to load vector store from {VECTORSTORE_PATH}...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedder, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Warning: Vector store not found or failed to load ({e}). Creating a new empty index.")
    try:
        import faiss
        from langchain_community.docstore.in_memory import InMemoryDocstore
        
        # Create empty index (1536 dimensions for text-embedding-3-small)
        # Note: Must match the embedding model dimension
        index = faiss.IndexFlatL2(1536) 
        vectorstore = FAISS(
            embedding_function=embedder,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
    except Exception as inner_e:
        print(f"Critical Error creating empty index: {inner_e}")
        raise inner_e

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

