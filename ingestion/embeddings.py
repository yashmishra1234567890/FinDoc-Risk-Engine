import hashlib
import os
import re
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

import numpy as np


load_dotenv()


def get_embedding_model():
    """
    Uses OpenAI Embeddings via OpenRouter to save RAM on the server.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to initialize embeddings.")

    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1"
    )


class LocalHashEmbeddings(Embeddings):
    """
    Deterministic offline embeddings used for tests and offline evaluation.

    Builds a hashed bag-of-tokens representation, so results are reproducible
    without any network call. This is NOT a replacement for the production
    embedder -- it only exists so the retrieval pipeline can be unit-tested
    and benchmarked deterministically.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 512):
        self.dim = dim

    @staticmethod
    def _stable_hash(token: str, dim: int) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "little") % dim

    def _embed(self, text: str):
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in self._TOKEN_RE.findall((text or "").lower()):
            vec[self._stable_hash(tok, self.dim)] += 1.0
        # character 3-grams add structure for short numeric snippets
        flat = re.sub(r"[^a-z0-9]", "", (text or "").lower())
        for i in range(len(flat) - 2):
            gram = flat[i:i + 3]
            vec[self._stable_hash(gram, self.dim)] += 0.25
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_documents(self, texts):
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text: str):
        return self._embed(text).tolist()

    def __call__(self, input):
        return self.embed_documents(input)
