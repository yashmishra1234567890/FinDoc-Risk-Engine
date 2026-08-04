import os
from openrouter import OpenRouter
from graph.state import GraphState

def get_client():
    return OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY", ""))


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
    client = get_client()
    prompt = build_summary_prompt(user_query, analysis_result, compliance_result, retrieved_chunks)

    response = client.chat.send(
        model="mistralai/ministral-8b-2512",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def summarize_report_stream(user_query, analysis_result, compliance_result, retrieved_chunks):
    client = get_client()
    prompt = build_summary_prompt(user_query, analysis_result, compliance_result, retrieved_chunks)
    response = client.chat.send(
        model="mistralai/ministral-8b-2512",
        messages=[{"role": "user", "content": prompt}]
    )
    yield response.choices[0].message.content
