import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings


load_dotenv()


def get_embedding_model():
    """
    Uses OpenAI Embeddings via OpenRouter to save RAM on the server.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to initialize embeddings.")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )
