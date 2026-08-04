from ingestion.embeddings import get_embedding_model
from langchain_community.vectorstores import Chroma
from graph.graph import build_graph
from app.api.core.config import VECTORSTORE_PATH

_embedder = get_embedding_model()
_vectorstore = Chroma(persist_directory=VECTORSTORE_PATH, embedding_function=_embedder)
_graph_app = build_graph(_vectorstore)

def get_graph_app():
    return _graph_app
