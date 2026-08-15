"""
Cross-encoder reranker (configurable) with a graceful no-op fallback.

The model name is read from configuration (``RERANKER_MODEL``) and is never
hardcoded here. If the model cannot be loaded (e.g. offline / too large) the
pipeline degrades to the RRF ordering instead of failing.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class Reranker:
    """Interface for a reranker over retrieved candidate dicts."""

    name = "base"

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        raise NotImplementedError


class NoopReranker(Reranker):
    """Pass-through: keeps the RRF ordering, truncates to top_k."""

    name = "none"

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        n = len(candidates) if top_k is None else min(top_k, len(candidates))
        result = list(candidates[:n])
        for entry in result:
            entry.setdefault("scores", {})["rerank"] = None
        return result


class CrossEncoderReranker(Reranker):
    """Reranks candidates using a sentence-transformers CrossEncoder model."""

    name = "cross-encoder"

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[dict], top_k: Optional[int] = None) -> List[dict]:
        if not candidates:
            return []
        pairs = [(query, c.get("text") or c.get("content") or "") for c in candidates]
        scores = self.model.predict(pairs)
        # normalize to a python list of floats
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        ordered = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        result = []
        for i, (entry, score) in enumerate(ordered):
            entry["rank"] = i + 1
            entry.setdefault("scores", {})["rerank"] = round(float(score), 6)
            result.append(entry)
            if top_k is not None and len(result) >= top_k:
                break
        return result


def build_reranker(enabled: bool, model_name: Optional[str] = None) -> Reranker:
    """
    Instantiate the configured reranker.

    Returns a :class:`NoopReranker` when disabled or when the cross-encoder
    model cannot be loaded, so retrieval never crashes because of a model.
    """
    if not enabled:
        return NoopReranker()
    try:
        reranker = CrossEncoderReranker(model_name)
        logger.info("Cross-encoder reranker loaded: %s", model_name)
        return reranker
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.warning("Cross-encoder reranker unavailable (%s); using RRF order.", exc)
        return NoopReranker()