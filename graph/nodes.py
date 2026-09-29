from agents.decomposer_agent import decompose_query
from agents.retriever_agent import retrieve_content
from agents.analysis_agent import analyze_financials
from agents.validator_agent import validate_analysis
from agents.summarizer_agent import summarize_report
from agents.evidence_agent import check_evidence_sufficiency
from reasoning.intent import detect_intent
from reasoning.engine import run_numerical_reasoning

def decompose_node(state):
    print("--- DECOMPOSE ---")
    sub_questions = decompose_query(state.user_query)
    return {"sub_questions": sub_questions}

def retrieve_node(state, vectorstore):
    print("--- RETRIEVE ---")
    # Phase 5: use the refined query on retry passes, the decomposed
    # sub-questions otherwise. Keep the existing retrieval components intact.
    if state.refined_query:
        queries = [state.refined_query]
    else:
        queries = state.sub_questions

    chunks = retrieve_content(queries, vectorstore)
    attempts = (state.retrieval_attempts or 0) + 1
    return {"retrieved_chunks": chunks, "retrieval_attempts": attempts}

def evidence_check_node(state):
    print("--- EVIDENCE CHECK ---")
    decision = check_evidence_sufficiency(state.user_query, state.retrieved_chunks)
    return {
        "evidence_check": decision,
        "refined_query": decision.get("refined_query"),
    }

def numerical_node(state):
    print("--- NUMERICAL (Phase 4) ---")
    result = run_numerical_reasoning(
        question=state.user_query,
        chunks=state.retrieved_chunks,
    )
    if result.get("requires_calculation") and result.get("answer"):
        return {
            "numerical_result": result,
            "final_answer": result["answer"],
            "verification": result.get("verification") or {},
        }
    # Not a supported calculation - fall through to the standard pipeline
    return {"numerical_result": result}


def unsupported_node(state):
    """Return a grounded refusal when retrieval cannot support the claim."""
    return {
        "final_answer": "I cannot verify this claim from the uploaded document.",
        "verification": {
            "status": "INSUFFICIENT EVIDENCE",
            "reason": (state.evidence_check or {}).get("reason", "Evidence was not sufficient."),
        },
    }

def analysis_node(state):
    print("--- ANALYZE ---")
    if (
        state.evidence_check
        and not state.evidence_check.get("sufficient")
        and (state.retrieval_attempts or 0) >= 3
    ):
        return {
            "analysis_result": {},
            "final_answer": "I cannot verify this claim from the uploaded document.",
            "verification": {
                "status": "INSUFFICIENT EVIDENCE",
                "reason": state.evidence_check.get("reason", "Evidence was not sufficient."),
            },
        }
    try:
        result = analyze_financials(state.retrieved_chunks, state.user_query)
        return {"analysis_result": result}
    except Exception as e:
        print(f"Analysis Logic Failed: {e}")
        # Graceful degradation
        return {
            "analysis_result": {
                "extracted_metrics": {},
                "derived_ratios": {},
                "missing_metrics": ["error_during_analysis"],
                "pages_used": []
            }
        }

def validate_node(state):
    print("--- VALIDATE ---")
    compliance = validate_analysis(state.analysis_result, user_query=state.user_query)
    return {"compliance_result": compliance}

def summarize_node(state):
    print("--- SUMMARIZE ---")
    if state.verification and state.verification.get("status") == "INSUFFICIENT EVIDENCE":
        return {"final_answer": state.final_answer}
    # Numerical answers are produced deterministically and must not be
    # replaced by an unconstrained language-model paraphrase.
    if state.numerical_result and state.numerical_result.get("answer"):
        return {"final_answer": state.numerical_result["answer"]}
    final = summarize_report(state.user_query, state.analysis_result, state.compliance_result, state.retrieved_chunks)
    return {"final_answer": final}
