from utils.llm import chat_completion, chat_completion_stream, get_model


def build_summary_prompt(user_query, analysis_result, compliance_result, retrieved_chunks):
    metrics_str = "\n".join([f"{k}: {v}" for k, v in analysis_result.get("extracted_metrics", {}).items() if v is not None])
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks[:6]])

    return f"""
You are a highly intelligent financial analyst. You have access to extracted metrics, source evidence, and validation notes.

User's Question: "{user_query}"

--- RAW DOCUMENT CONTEXT (Most Relevant Segments) ---
{context_text}
--- END CONTEXT ---

--- EXTRACTED FINANCIAL METRICS ---
{metrics_str}
-----------------------------------

--- RISK/COMPLIANCE ANALYSIS ---
{compliance_result}
--------------------------------

INSTRUCTIONS:
1. Answer accurately using the extracted metrics and context.
2. If data is missing, say so plainly.
3. Be direct and concise.

Answer:
"""


def summarize_report(user_query, analysis_result, compliance_result, retrieved_chunks):
    prompt = build_summary_prompt(user_query, analysis_result, compliance_result, retrieved_chunks)
    return chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model=get_model(),
    )


def summarize_report_stream(user_query, analysis_result, compliance_result, retrieved_chunks):
    prompt = build_summary_prompt(user_query, analysis_result, compliance_result, retrieved_chunks)
    yield from chat_completion_stream(
        messages=[{"role": "user", "content": prompt}],
        model=get_model(),
    )