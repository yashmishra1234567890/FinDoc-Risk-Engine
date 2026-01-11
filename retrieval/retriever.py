def retrieve_chunks(vectorstore, query: str, k: int = 4):
    """
    Retrieve top-k relevant chunks for a query.
    """
    results = vectorstore.similarity_search(query, k=k)

    return [
        {
            "content": doc.page_content,
            "page_no": doc.metadata["page_no"],
            "has_table": doc.metadata["has_table"]
        }
        for doc in results
    ]
