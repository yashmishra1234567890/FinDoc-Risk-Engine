"""
System Evaluator
----------------
Runs a set of test queries against the full FinDoc Agent pipeline and records performance.

How to run:
    python -m evaluation.evaluator
"""

import sys
import os
import time
import asyncio
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.api.core.config import VECTORSTORE_PATH
from langchain_community.vectorstores import FAISS
from ingestion.embeddings import get_embedding_model
from graph.graph import build_graph
from evaluation.metrics import calculate_latency, check_keyword_presence

# Ensure environment variables are loaded for the test
from dotenv import load_dotenv
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(env_path)

RESULTS_FILE = "evaluation/results.md"

TEST_QUERIES = [
    {
        "question": "What is the total debt and equity for FY 2023?",
        "expected_keywords": ["debt", "equity", "liability"]
    },
    {
        "question": "Calculate the interest coverage ratio.",
        "expected_keywords": ["coverage", "ratio", "interest", "EBITDA"]
    }
]

def run_evaluation():
    print("--- 🚀 STARTING EVALUATION ---")
    
    # 1. Initialize System
    print(f"Loading system resources...")
    embedder = get_embedding_model()
    try:
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedder, allow_dangerous_deserialization=True)
        app = build_graph(vectorstore)
    except Exception as e:
        print(f"❌ Failed to load vector store: {e}")
        return

    results_buffer = []
    results_buffer.append(f"# 📊 Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    results_buffer.append("| Question | Latency (s) | Keywords Hit | Confidence | Validator Flag |")
    results_buffer.append("|---|---|---|---|---|")

    # 2. Run Queries
    for test in TEST_QUERIES:
        q = test["question"]
        print(f"Testing: {q}...")
        
        start_time = time.time()
        try:
            # invoke pipeline
            state = {"user_query": q}
            output = app.invoke(state)
            
            end_time = time.time()
            latency = calculate_latency(start_time, end_time)
            
            # Extract Metrics
            final_ans = output.get("final_answer", "")
            kw_score = check_keyword_presence(final_ans, test["expected_keywords"])
            
            # detailed info from agents
            compliance = output.get("compliance_result", {})
            confidence = compliance.get("confidence_score", "N/A")
            flags = compliance.get("rule_engine_flags", [])
            flag_summary = ", ".join(flags) if flags else "None"
            
            # Add to table
            row = f"| {q} | {latency}s | {kw_score} | {confidence} | {flag_summary} |"
            results_buffer.append(row)
            print(f"  ✅ Done in {latency}s")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results_buffer.append(f"| {q} | ERROR | 0.0 | N/A | {e} |")

    # 3. Save Results
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(results_buffer) + "\n\n")
    
    print(f"\n✅ Evaluation complete. Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    run_evaluation()
