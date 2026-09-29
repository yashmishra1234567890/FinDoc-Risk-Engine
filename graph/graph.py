"""
LangGraph Orchestrator
----------------------
Defines the workflow graph connecting the agents:
Decomposer -> Retrieve -> [Evidence Check] -> [Numerical (Phase 4) | Analyst] -> Validator -> Summarizer

Phase 5 adds a lightweight evidence-sufficiency check after retrieval. When the
retrieved evidence is insufficient the query is refined and retrieval runs
again (maximum 3 attempts); the graph always terminates.
"""
from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes import (
    decompose_node,
    retrieve_node,
    evidence_check_node,
    numerical_node,
    analysis_node,
    validate_node,
    summarize_node,
)
from graph.edges import (
    route_after_evidence_check,
    route_after_numerical,
    route_after_validation,
)

def build_graph(vectorstore):
    graph = StateGraph(GraphState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, vectorstore))
    graph.add_node("evidence_check", evidence_check_node)
    graph.add_node("numerical", numerical_node)
    graph.add_node("analyze", analysis_node)
    graph.add_node("validate", validate_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "evidence_check")

    # Phase 5: insufficient evidence -> refine and retrieve (bounded); enough
    # evidence (or max attempts) -> proceed to numerical/analysis routing.
    graph.add_conditional_edges(
        "evidence_check",
        route_after_evidence_check,
        {"retrieve": "retrieve", "numerical": "numerical", "analyze": "analyze"}
    )

    # If numerical reasoning produced a final answer, stop; otherwise fall
    # through to the standard analysis pipeline.
    graph.add_conditional_edges(
        "numerical",
        route_after_numerical,
        {"analyze": "analyze", END: END}
    )

    graph.add_edge("analyze", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"summarize": "summarize", END: END}
    )

    graph.add_edge("summarize", END)

    return graph.compile()
