from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



def build_faiss_index(chunks, embedder):
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

    return FAISS.from_documents(docs, embedder)