"""
LangGraph Orchestrator
----------------------
Defines the workflow graph connecting the agents:
Decomposer -> Retriever -> Analyst -> Validator -> Summarizer

"""
from langgraph.graph import StateGraph
from graph.state import GraphState
from graph.nodes import (
    decompose_node,
    retrieve_node,
    analysis_node,
    validate_node,
    summarize_node,
)
from graph.edges import route_after_validation

def build_graph(vectorstore):
    graph = StateGraph(GraphState)

    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, vectorstore))
    graph.add_node("analyze", analysis_node)
    graph.add_node("validate", validate_node)
    graph.add_node("summarize", summarize_node)

    graph.set_entry_point("decompose")

    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {"summarize": "summarize"}
    )

    return graph.compile()
