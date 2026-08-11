from fastapi import APIRouter
import logging

from app.api.schemas.request import QueryRequest
from app.api.schemas.response import QueryResponse, Source
from app.api.core.store import vectorstore
from graph.graph import build_graph

router = APIRouter(tags=["Query"])
logger = logging.getLogger(__name__)

graph_app = build_graph(vectorstore)


@router.post("/query", response_model=QueryResponse)
def query_financials(req: QueryRequest):
    try:
        # Check if we have data
        doc_count = 0
        try:
            doc_count = vectorstore._collection.count()
        except Exception:
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

        if chunks:
            for i, chunk in enumerate(chunks):
                try:
                    raw_page = chunk.get("page_no")
                    p_no = 0
                    if raw_page is not None:
                        try:
                            p_no = int(raw_page)
                        except Exception:
                            p_no = 0

                    txt = chunk.get("content", "")
                    if not txt:
                        txt = "No content available"

                    snippet = txt[:100].replace("\n", " ") + "..." if len(txt) > 100 else txt

                    if p_no not in unique_pages:
                        unique_pages.add(p_no)
                        sources.append(Source(page_no=p_no, snippet=snippet))

                except Exception as e:
                    logger.warning(f"Error processing source {i}: {str(e)}")

            logger.info(f"Query processed. Found {len(sources)} sources from {len(chunks)} chunks.")
        else:
            logger.warning("Query returned answer but 'retrieved_chunks' was empty in state.")

        # Extract analysis and compliance results from graph state
        analysis = result.get("analysis_result", {})
        compliance = result.get("compliance_result", {})

        metrics = analysis.get("extracted_metrics", {}) or {}
        ratios = analysis.get("derived_ratios", {}) or {}
        confidence = compliance.get("confidence_score", 0.0) or 0.0
        market_context = compliance.get("market_context", {}) or {}
        federated_sources = compliance.get("federated_sources", {}) or {}
        temperature_risk = compliance.get("temperature_risk", 0.0) or 0.0

        return QueryResponse(
            answer=result.get("final_answer", "No answer generated."),
            confidence=confidence,
            sources=sorted(sources, key=lambda x: x.page_no),
            metrics=metrics,
            ratios=ratios,
            compliance=compliance,
            market_context=market_context,
            federated_sources=federated_sources,
            temperature_risk=temperature_risk
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer=f"I encountered an error analyzing the document. Please try again. ({str(e)})",
            confidence=0.0,
            sources=[]
        )
