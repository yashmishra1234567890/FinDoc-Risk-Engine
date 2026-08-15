import logging
from utils.llm import chat_completion, get_model

logger = logging.getLogger(__name__)


def decompose_query(user_query: str):
    prompt = f"""
You are a financial analyst.

Break the following question into 2–4 clear sub-questions
that can be answered using financial documents.

Question:
{user_query}

Return as bullet points.
"""

    try:
        content = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=get_model(),
        )
        sub_questions = [line.strip("- ").strip() for line in content.split("\n") if line.strip().startswith("-")]

        # Fallback: If no bullet points found, or empty, use original query
        if not sub_questions:
            return [user_query]

        return sub_questions

    except Exception as e:
        logger.error(f"Decomposer Error: {e}")
        # Critical Fallback: Ensure we never return empty list, or the retriever will do nothing
        return [user_query]