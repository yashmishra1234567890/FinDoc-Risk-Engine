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
    
    # Format metrics for the LLM to easily read
    metrics_str = "\n".join([f"{k}: {v}" for k, v in analysis_result.get("extracted_metrics", {}).items() if v is not None])
    
    prompt = f"""
You are a helpful financial assistant.

User's Question: "{user_query}"

FOUND DATA (Use this to answer):
{metrics_str}

Risk Report (Context):
{compliance_result}

INSTRUCTIONS:
1. Answer the User's Question DIRECTLY using the "FOUND DATA". 
   - Example: "The Revenue is 626,130."
   - If the specific number is in FOUND DATA, you MUST state it.
2. Only AFTER answering, mention any missing risk data (like Debt/Equity) as a secondary note.
3. If the answer is not in FOUND DATA, say "I found financial data, but not the specific metric you asked for."

Keep it professional and concise.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
