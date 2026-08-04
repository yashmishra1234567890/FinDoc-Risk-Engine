import os
import logging
from openrouter import OpenRouter

logger = logging.getLogger(__name__)

def get_client():
    return OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY", ""))

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

    try:
        response = client.chat.send(
            model="mistralai/ministral-8b-2512",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        sub_questions = [line.strip("- ").strip() for line in content.split("\n") if line.strip().startswith("-")]

        # Fallback: If no bullet points found, or empty, use original query
        if not sub_questions:
             return [user_query]
             
        return sub_questions

    except Exception as e:
        logger.error(f"Decomposer Error: {e}")
        # Critical Fallback: Ensure we never return empty list, or the retriever will do nothing
        return [user_query]
