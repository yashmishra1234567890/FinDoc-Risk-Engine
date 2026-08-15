"""
Lightweight Okapi BM25 (pure Python + NumPy).

No external IR dependency is required, keeping the project lightweight while
still providing a real lexical retrieval leg for the hybrid pipeline.

Metadata (page_no, source_id, table_id, row_idx, ...) is carried alongside each
document and preserved verbatim on search results, so downstream citation and
page/source tracking keeps working.
"""
import math
import re
from typing import Dict, Iterable, List, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: Optional[str]) -> List[str]:
    """Lower-case tokenization of letters/digits (handles money like 45,200,000 -> 45 200 000)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 index over a list of documents.

    Documents are dicts with keys ``id``, ``text`` and ``metadata``. Search
    results preserve the full ``metadata`` dict so page/source/table info is
    never lost during retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[Dict] = []          # [{id, text, metadata}]
        self._tfs: List[Dict[str, int]] = []  # per-doc term frequencies
        self._dls: List[int] = []             # per-doc token lengths
        self._df: Dict[str, int] = {}         # document frequencies
        self._N: int = 0
        self._avgdl: float = 1.0

    # ------------------------------------------------------------------ build
    def add_documents(self, docs: Iterable[Dict], **__) -> None:
        """Add documents. Each doc = {'id': str, 'text': str, 'metadata': dict}."""
        for doc in docs:
            text = doc.get("text") or ""
            terms = tokenize(text)
            dl = len(terms)
            tf: Dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1
            self._docs.append(doc)
            self._tfs.append(tf)
            self._dls.append(dl)
            self._N += 1
        if self._N:
            self._avgdl = sum(self._dls) / self._N

    def __len__(self) -> int:
        return self._N

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 20) -> List[Dict]:
        """Return top-k matched docs as dicts with id/text/metadata/score/rank."""
        query_terms = tokenize(query)
        if not query_terms or self._N == 0:
            return []

        # idf pre-computed for the query terms present in the corpus
        idf: Dict[str, float] = {}
        for term in query_terms:
            df = self._df.get(term, 0)
            if df:
                idf[term] = math.log1p((self._N - df + 0.5) / (df + 0.5))

        scored: List[Dict] = []
        k1, b, avgdl = self.k1, self.b, self._avgdl
        for i in range(self._N):
            tf = self._tfs[i]
            dl = self._dls[i]
            score = 0.0
            for term in query_terms:
                idf_t = idf.get(term)
                if idf_t is None:
                    continue
                f = tf.get(term, 0)
                if f == 0:
                    continue
                denom = f + k1 * (1 - b + b * dl / avgdl)
                score += idf_t * (f * (k1 + 1)) / denom
            if score > 0.0:
                scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, i) in enumerate(scored[:k]):
            doc = self._docs[i]
            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "content": doc["text"],
                    "metadata": doc.get("metadata") or {},
                    "score": round(score, 4),
                    "rank": rank + 1,
                    "scores": {"bm25": round(score, 4)},
                }
            )
        return results