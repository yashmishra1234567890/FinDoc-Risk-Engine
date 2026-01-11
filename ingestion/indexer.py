from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



def create_documents_from_chunks(chunks):
    docs = []
    for chunk in chunks:
        docs.append(
            Document(
                page_content=chunk["content"],
                metadata={
                    "page_no": chunk["page_no"],
                    "has_table": chunk["has_table"]
                }
            )
        )
    return docs

def build_faiss_index(chunks, embedder):
    docs = create_documents_from_chunks(chunks)
    return FAISS.from_documents(docs, embedder)