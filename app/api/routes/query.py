from fastapi import APIRouter
from langchain_community.vectorstores import FAISS
import os
import logging
import json

from app.api.schemas.request import QueryRequest
from app.api.schemas.response import QueryResponse, Source
# from app.api.core.config import VECTORSTORE_PATH # Removed as logic moved to store.py
from app.api.core.store import vectorstore
from graph.graph import build_graph

router = APIRouter(tags=["Query"])
logger = logging.getLogger(__name__)

# Logic moved to app/api/core/store.py to be shared with upload endpoint
# embedder = get_embedding_model()
# ... 

graph_app = build_graph(vectorstore)


@router.post("/query", response_model=QueryResponse)
def query_financials(req: QueryRequest):
    try:
        # Check if we have data
        doc_count = 0
        try:
             doc_count = vectorstore.index.ntotal
        except:
             pass

        if doc_count == 0:
             return QueryResponse(
                answer="No documents found in the database. Please wait a moment if you just uploaded a file (processing takes 1-2 mins). If this persists, the PDF might be unreadable/scanned.",
                confidence=0.0,
                sources=[]
            )

        logger.info(f"Received query: {req.question}")
        state = {
            "user_query": req.question
        }

        result = graph_app.invoke(state)
        
        # Deduplicate sources based on page number
        unique_pages = set()
        sources = []
        
        # Robust retrieval of chunks from state
        chunks = result.get("retrieved_chunks", [])
        
        # If chunks is empty, but we have an answer, it might be hiding in a different key or format
        # But assuming flow is correct:
        if chunks:
            for i, chunk in enumerate(chunks):
                try:
                    # Safely extract page number, default to 0 if missing
                    raw_page = chunk.get("page_no")
                    p_no = 0
                    if raw_page is not None:
                        try:
                            p_no = int(raw_page)
                        except:
                            p_no = 0
                        
                    # Safely extract content
                    txt = chunk.get("content", "")
                    if not txt:
                         txt = "No content available"
                    
                    # Create snippet
                    snippet = txt[:100].replace("\n", " ") + "..." if len(txt) > 100 else txt
                    
                    # Add to sources if we haven't seen this page before
                    # (Or if page is 0, we might want to show at least one source)
                    if p_no not in unique_pages:
                        unique_pages.add(p_no)
                        sources.append(Source(page_no=p_no, snippet=snippet))
                        
                except Exception as e:
                    # Log but continue - don't let one bad source fail the request
                    logger.warning(f"Error processing source {i}: {str(e)}")
            
            logger.info(f"Query processed. Found {len(sources)} sources from {len(chunks)} chunks.")
        else:
             logger.warning("Query returned answer but 'retrieved_chunks' was empty in state.")

        # Return the final answer
        return QueryResponse(
            answer=result["final_answer"],
            confidence=0.85,
            sources=sorted(sources, key=lambda x: x.page_no)
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Return a fallback response or re-raise
        return QueryResponse(
            answer=f"I encountered an error analyzing the document. Please try again. ({str(e)})",
            confidence=0.0,
            sources=[]
        )
