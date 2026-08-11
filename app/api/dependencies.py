from app.api.core.store import get_vectorstore
from graph.graph import build_graph

def get_graph_app():
    return build_graph(get_vectorstore())