import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from graph.state import GraphState
from graph.graph import build_graph
from ingestion.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS

# Ensure vectorstore exists or mock it
# For now, we assume it needs to be loaded, but if it doesn't exist we might need to create it.
import os
if not os.path.exists("vectorstore/faiss_index"):
    print("Vectorstore not found. Please run ingestion first.")
    exit(1)

embedder = get_embedding_model()
try:
    vectorstore = FAISS.load_local(
        "vectorstore/faiss_index", 
        embedder, 
        allow_dangerous_deserialization=True
    )
except Exception as e:
    print(f"Error loading vectorstore: {e}")
    exit(1)

app = build_graph(vectorstore)

state = GraphState(
    user_query="Analyze debt risk and liabilities for FY 2023"
)

result = app.invoke(state)
print(result["final_answer"])
