from agents.decomposer_agent import decompose_query
from agents.retriever_agent import retrieve_content
from agents.analysis_agent import analyze_financials
from agents.validator_agent import validate_analysis
from agents.summarizer_agent import summarize_report

def decompose_node(state):
    print("--- DECOMPOSE ---")
    sub_questions = decompose_query(state.user_query)
    return {"sub_questions": sub_questions}

def retrieve_node(state, vectorstore):
    print("--- RETRIEVE ---")
    chunks = retrieve_content(state.sub_questions, vectorstore)
    return {"retrieved_chunks": chunks}

def analysis_node(state):
    print("--- ANALYZE ---")
    result = analyze_financials(state.retrieved_chunks, state.user_query)
    return {"analysis_result": result}

def validate_node(state):
    print("--- VALIDATE ---")
    compliance = validate_analysis(state.analysis_result)
    return {"compliance_result": compliance}

def summarize_node(state):
    print("--- SUMMARIZE ---")
    final = summarize_report(state.user_query, state.analysis_result, state.compliance_result)
    return {"final_answer": final}
