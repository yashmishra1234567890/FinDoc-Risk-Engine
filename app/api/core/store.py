from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from app.api.core.config import VECTORSTORE_PATH
from ingestion.embeddings import get_embedding_model
import faiss
import os

def get_embedding_dimension(embedder):
    """Dynamically determine the embedding dimension."""
    try:
        sample_vector = embedder.embed_query("test")
        return len(sample_vector)
    except Exception as e:
        print(f"Failed to determine embedding dimension dynamically, defaulting to 1536: {e}")
        return 1536

def load_or_initialize_vectorstore():
    embedder = get_embedding_model()
    
    try:
        print(f"Attempting to load vector store from {VECTORSTORE_PATH}...")
        # Security: Ensure we only load from our trusted local directory
        trusted_path = os.path.abspath(VECTORSTORE_PATH)
        
        # Check if index file exists before trying to load
        if os.path.exists(os.path.join(trusted_path, "index.faiss")):
             # We explicitly allow dangerous deserialization here because the 
             # vector store is created internally and not from user uploads.
             vectorstore = FAISS.load_local(
                 trusted_path, 
                 embedder, 
                 allow_dangerous_deserialization=True
             )
             print("Vector store loaded successfully.")
             return vectorstore
        else:
             print("Index file not found. Creating new.")
             raise RuntimeError("Index not found")
             
    except Exception as e:
        print(f"Warning: Vector store not found or failed to load ({e}). Creating a new empty index.")
        try:
            dim = get_embedding_dimension(embedder)
            print(f"Creating new empty index with dimension: {dim}")
            index = faiss.IndexFlatL2(dim) 
            vectorstore = FAISS(
                embedding_function=embedder,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            return vectorstore
        except Exception as inner_e:
            print(f"Critical Error creating empty index: {inner_e}")
            raise inner_e

# Global singleton instance
vectorstore = load_or_initialize_vectorstore()

def reset_vectorstore():
    """
    Resets the global vectorstore in-place.
    Crucial for clearing old document data when a new file is uploaded.
    """
    print("🧹 Clearing Vector Store...")
    try:
        dim = get_embedding_dimension(get_embedding_model())
        # Create fresh components
        new_index = faiss.IndexFlatL2(dim)
        new_docstore = InMemoryDocstore()
        
        # Update the existing singleton provided to other modules
        vectorstore.index = new_index
        vectorstore.docstore = new_docstore
        vectorstore.index_to_docstore_id = {}
        print("✅ Vector Store Reset Complete.")
    except Exception as e:
        print(f"Error resetting vector store: {e}")
