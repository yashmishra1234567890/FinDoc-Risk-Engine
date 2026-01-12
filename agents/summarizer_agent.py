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
You are a concise financial analyst provided with data from a PDF.

User's Question: "{user_query}"

Data Found:
{metrics_str}

Risk Analysis (Internal Use):
{compliance_result}

INSTRUCTIONS:
1. If the user asks for a specific metric (e.g., "What is Revenue?"), ONLY provide that answer.
   - DO NOT mention missing debt, missing equity, or risk flags unless asked.
   - Keep it short: "The Revenue is X."
   
2. If the user asks for "Summary", "Risk", or "Analysis", THEN provide the full Risk Assessment.
   - Use the Risk Analysis provided above to answer these questions.

3. Be direct. No filler words.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
