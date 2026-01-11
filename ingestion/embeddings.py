import os
from langchain_openai import OpenAIEmbeddings


def get_embedding_model():
    """
    Uses OpenAI Embeddings via OpenRouter to save RAM on the server.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
