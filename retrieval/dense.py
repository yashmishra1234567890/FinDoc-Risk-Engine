"""
Dense retrieval leg over a persisted Chroma collection with id alignment.

We query the underlying Chroma collection directly so that returned ids can be
matched to the BM25 leg during RRF fusion. This *is* the existing dense
retrieval (same ANN index / persisted data) -- it is not a replacement.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChromaDenseIndex:
    """Wraps a langchain Chroma vectorstore as an id-aligned dense index."""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def _embed(self, query: str):
        ef = getattr(self.vectorstore, "_embedding_function", None) or \
            getattr(self.vectorstore, "embedding_function", None)
        if ef is None:
            return None
        for meth in ("embed_query",):
            fn = getattr(ef, meth, None)
            if callable(fn):
                try:
                    return fn(query)
                except TypeError:
                    continue
        fn = getattr(ef, "embed_documents", None)
        if callable(fn):
            return fn([query])[0]
        if callable(ef):  # raw callable embedding function
            return ef([query])[0]
        raise RuntimeError("No usable embedding function on the vectorstore")

    def search(self, query: str, k: int = 20) -> List[dict]:
        """Return top-k dense hits as dicts with id/text/metadata/score/rank."""
        vs = self.vectorstore
        collection = getattr(vs, "_collection", None)

        if collection is not None and hasattr(collection, "query"):
            return self._search_collection(query, k)
        # fallback: plain similarity_search (no id alignment but still dense)
        found = vs.similarity_search_with_score(query, k=k)
        return [
            {
                "id": getattr(d, "id", None) or d.page_content[:80],
                "text": d.page_content,
                "content": d.page_content,
                "metadata": dict(d.metadata or {}),
                "score": round(float(1.0 / (1.0 + score)), 4),
                "rank": i + 1,
                "scores": {"dense": round(float(1.0 / (1.0 + score)), 4)},
            }
            for i, (d, score) in enumerate(found)
        ]

    def _search_collection(self, query: str, k: int):
        collection = self.vectorstore._collection
        count = collection.count()
        k = max(0, min(int(k), count))
        if k == 0:
            return []
        embedding = self._embed(query)
        if embedding is None:
            return []
        try:
            res = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Chroma collection query failed (%s); empty dense leg.", exc)
            return []

        ids = res.get("ids", [[]])[0]
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]

        entries = []
        for i, doc_id in enumerate(ids):
            text = documents[i] if i < len(documents) else ""
            meta = dict(metadatas[i] or {}) if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            sim = 1.0 / (1.0 + float(distance or 0.0))
            logical_id = meta.pop("_logical_id", None)
            entries.append(
                {
                    "id": logical_id or doc_id,
                    "text": text,
                    "content": text,
                    "metadata": meta,
                    "score": round(sim, 4),
                    "rank": i + 1,
                    "scores": {"dense": round(sim, 4), "distance": round(float(distance), 4)},
                }
            )
        return entries