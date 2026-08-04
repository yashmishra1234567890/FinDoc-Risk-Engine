from langchain_community.vectorstores import Chroma
from app.api.core.config import VECTORSTORE_PATH
from langchain_core.documents import Document
import os


def build_index(chunks, embedder):
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

    trusted_path = os.path.abspath(VECTORSTORE_PATH)
    os.makedirs(trusted_path, exist_ok=True)
    return Chroma.from_documents(docs, embedder, persist_directory=trusted_path)