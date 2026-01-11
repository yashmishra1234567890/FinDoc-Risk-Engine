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
User Question:
{user_query}

Analysis:
{analysis_result}

Compliance Check:
{compliance_result}

Write a concise summary (5–6 lines)
for a bank or CA.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
