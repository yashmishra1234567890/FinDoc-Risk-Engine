import os
from openai import OpenAI
from graph.state import GraphState

def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def summarize_report(user_query, analysis_result, compliance_result, retrieved_chunks):
    client = get_client()
    
    # 1. Format Regex Metrics
    metrics_str = "\n".join([f"{k}: {v}" for k, v in analysis_result.get("extracted_metrics", {}).items() if v is not None])
    
    # 2. Format Retrieved Context (Raw Text) - Limit to first 5 most relevant to fit context window
    context_text = "\n\n---\n\n".join([chunk["content"] for chunk in retrieved_chunks[:6]])

    prompt = f"""
You are a highly intelligent financial analyst. You have access to extracted metrics AND raw text segments from a document.

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
1. **Primary Goal**: Answer the user's question accurately using EITHER the extracted metrics OR the "Raw Document Context".
   - If the user asks for "Company Name", "Sector", or other metadata, find it in the Context.
   - If the user asks for "Profit", "Revenue", use the Metrics first, but verify with the Context.

2. **If Data is Missing**:
   - If you cannot find the answer in the Context or Metrics, simply state: "The information regarding [topic] is not provided in the document."
   - Do NOT make up numbers.

3. **Format**:
   - Be direct. "The company name is [Name]." or "The revenue is [Amount]."
   - If asking for "Analysis" or "Summary", synthesize the Risk Analysis and Context.

Answer:
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
