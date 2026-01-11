import os
from langchain_openai import OpenAIEmbeddings


def get_embedding_model():
    """
    Uses OpenAI Embeddings (API-based) to save RAM on the server.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
