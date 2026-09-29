from langgraph.graph import END
from agents.evidence_agent import MAX_RETRIEVAL_ATTEMPTS
from reasoning.intent import detect_intent


def route_after_retrieval(state):
    """Route a numerical question through the Phase 4 reasoning node.

    Uses the deterministic intent detector; non-numerical questions continue
    through the standard analysis pipeline.
    """
    intent = detect_intent(state.user_query)
    if intent.requires_calculation:
        return "numerical"
    return "analyze"


def route_after_evidence_check(state):
    """
    Phase 5 - decide what to do after the evidence-sufficiency check.

    - Insufficient evidence and attempts left  -> refine and retrieve again
    - Sufficient evidence (or max attempts hit) -> proceed to the numerical
      node for calculation questions, the analysis pipeline otherwise.
    """
    check = state.evidence_check or {}
    attempts = state.retrieval_attempts or 0

    if not check.get("sufficient") and attempts < MAX_RETRIEVAL_ATTEMPTS:
        return "retrieve"

    if not check.get("sufficient"):
        # The deployed graph keeps the existing analyze/validate path as its
        # terminal branch; analysis_node converts this state to a grounded
        # unsupported response without calling the LLM.
        return "analyze"

    intent = detect_intent(state.user_query)
    if intent.requires_calculation:
        return "numerical"
    return "analyze"


def route_after_numerical(state):
    """After the Phase 4 numerical node.

    If the numerical engine produced a final answer we are done; otherwise
    fall through to the standard analysis pipeline.
    """
    if state.final_answer:
        return "summarize"
    return "analyze"


def route_after_validation(state):
    """After the validator (standard pipeline) - always summarize."""
    return "summarize"
