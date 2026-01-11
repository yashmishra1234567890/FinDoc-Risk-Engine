from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model():
    """
    Local embedding model (free & fast).
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
