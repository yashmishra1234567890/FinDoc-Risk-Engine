import os
from openai import OpenAI
from graph.state import GraphState

def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def summarize_report(user_query, analysis_result, compliance_result):
    client = get_client()
    prompt = f"""
You are a reliable financial assistant.

User Question:
"{user_query}"

Quantitative Analysis:
{analysis_result["extracted_metrics"]}

Compliance/Risk Flags:
{compliance_result}

INSTRUCTIONS:
1. FIRST, directly answer the User's Question using the data provided.
2. If the answer is "None" or missing in the analysis, state that found data segments but the specific number was unclear.
3. THEN, provide the risk assessment summary.

Do not hallucinate numbers not present in the Analysis.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
