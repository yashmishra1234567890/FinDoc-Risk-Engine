import os
import sys
from openai import OpenAI
from langchain_community.vectorstores import FAISS
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
        # Allow dangerous deserialization since we created the index ourselves
        self.vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    
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
        
        response = self.client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content

if __name__ == "__main__":
    # Simple test
    rag = BaselineRAG()
    test_q = "What is the total debt for FY 2023?"
    print(f"Query: {test_q}")
    print(f"Answer: {rag.run(test_q)}")
