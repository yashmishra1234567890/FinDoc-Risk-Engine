
import sys
import os
import asyncio
from dotenv import load_dotenv

# Load env before imports that might instantiate clients
# Explicitly point to the .env file in the root
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY not found in environment variables even after loading .env!")
else:
    print("OPENROUTER_API_KEY loaded successfully.")

# Add project root to sys.path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import FAISS
from app.api.core.config import VECTORSTORE_PATH
from ingestion.embeddings import get_embedding_model
from graph.graph import build_graph

def run_pipeline():
    print("--- STARTING PHASE 2 TEST (Full Pipeline) ---")
    
    # 1. Load Vector Store
    print(f"Loading vector store from {VECTORSTORE_PATH}...")
    try:
        embedder = get_embedding_model()
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedder, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return

    # 2. Build Graph
    print("Building Agent Graph...")
    app = build_graph(vectorstore)
    
    # 3. Define Query
    user_query = "Analyze the debt risk and liabilities for FY 2023. What are the major borrowings?"
    print(f"\nUser Query: {user_query}\n")

    state = {
        "user_query": user_query
    }

    # 4. Invoke Graph
    print("Invoking Graph (Streaming events)...")
    try:
        # Use simple invoke for now
        result = app.invoke(state)
        
        print("\n--- FINAL OUTPUT ---")
        print(f"Final Answer:\n{result.get('final_answer')}")
        print("\n--- AGENT STATES ---")
        if 'sub_questions' in result:
            print(f"Decomposer (Sub-questions): {result['sub_questions']}")
        
        if 'analysis_result' in result:
            ar = result['analysis_result']
            print("\nAnalysis Agent:")
            print(f"  - Metrics Found: {ar.get('extracted_metrics')}")
            print(f"  - Ratios: {ar.get('derived_ratios')}")
            print(f"  - Missing: {ar.get('missing_metrics')}")

        if 'compliance_result' in result:
            cr = result['compliance_result']
            print("\nValidator Agent:")
            print(f"  - Flags: {cr.get('rule_engine_flags')}")
            print(f"  - Confidence: {cr.get('confidence_score')}")
            print(f"  - LLM Explanation: {cr.get('assessment')[:200]}...") # Truncate for display

    except Exception as e:
        print(f"Error running graph: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_pipeline()
