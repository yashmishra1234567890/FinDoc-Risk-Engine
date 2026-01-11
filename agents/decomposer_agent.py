import os
from openai import OpenAI
from graph.state import GraphState

def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

def decompose_query(user_query: str):
    client = get_client()
    prompt = f"""
You are a financial analyst.

Break the following question into 2–4 clear sub-questions
that can be answered using financial documents.

Question:
{user_query}

Return as bullet points.
"""

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    lines = response.choices[0].message.content.split("\n")
    sub_questions = [
        line.strip("-• ").strip()
        for line in lines if line.strip()
    ]

    return sub_questions
