from langchain_community.vectorstores import Chroma
from app.api.core.config import VECTORSTORE_PATH
from ingestion.embeddings import get_embedding_model
import os
import logging
import shutil

logger = logging.getLogger(__name__)

def load_or_initialize_vectorstore():
    embedder = get_embedding_model()
    
    try:
        logger.info(f"Attempting to load/initialize Chroma vector store at {VECTORSTORE_PATH}...")
        trusted_path = os.path.abspath(VECTORSTORE_PATH)
        os.makedirs(trusted_path, exist_ok=True)
        
        vectorstore = Chroma(
            persist_directory=trusted_path,
            embedding_function=embedder
        )
        logger.info("Chroma vector store loaded successfully.")
        return vectorstore
             
    except Exception as e:
        logger.error(f"Critical Error initializing Chroma index: {e}")
        raise e

# Global singleton instance
vectorstore = load_or_initialize_vectorstore()


def get_vectorstore():
    return vectorstore

def reset_vectorstore():
    """
    Resets the global vectorstore in-place.
    Crucial for clearing old document data when a new file is uploaded.
    """
    global vectorstore
    logger.info("🧹 Clearing Vector Store...")
    try:
        try:
            vectorstore._collection.delete(where={})
        except Exception:
            trusted_path = os.path.abspath(VECTORSTORE_PATH)
            if os.path.exists(trusted_path):
                vectorstore = load_or_initialize_vectorstore()
            else:
                vectorstore = load_or_initialize_vectorstore()
        logger.info("✅ Vector Store Reset Complete.")
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
