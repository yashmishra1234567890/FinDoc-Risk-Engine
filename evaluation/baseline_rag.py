import os
import sys
from openrouter import OpenRouter
from langchain_community.vectorstores import Chroma
from ingestion.embeddings import get_embedding_model
from app.api.core.config import VECTORSTORE_PATH
from dotenv import load_dotenv

# Add project root to path if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

class BaselineRAG:
    """
    A standard Retrieval-Augmented Generation (RAG) pipeline.
    Acts as a control group to benchmark the Agentic system against.
    
    Flow: Retrieve Top-K Chunks -> Single LLM Call -> Answer
    """
    def __init__(self):
        print("Loading Baseline RAG resources...")
        self.embeddings = get_embedding_model()
        
        # Secure the path before loading memory files
        trusted_path = os.path.abspath(VECTORSTORE_PATH)
        # Trust this path since it's an internally generated index
        self.vectorstore = Chroma(
            persist_directory=trusted_path, 
            embedding_function=self.embeddings
        )
        self.client = OpenRouter(api_key=os.getenv("OPENROUTER_API_KEY", ""))
    
    def run(self, query: str) -> str:
        # 1. Retrieve Context
        docs = self.vectorstore.similarity_search(query, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        
        # 2. Generate Answer
        prompt = f"""You are a helper financial assistant. 
Answer the user's question based strictly on the context provided below.
If the answer is not in the context, say "Data not available".

Context:
{context}

Question: {query}
"""
        
        response = self.client.chat.send(
            model="mistralai/ministral-8b-2512",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Simple test
    rag = BaselineRAG()
    test_q = "What is the total debt for FY 2023?"
    print(f"Query: {test_q}")
    print(f"Answer: {rag.run(test_q)}")
