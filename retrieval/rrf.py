"""
Reciprocal Rank Fusion (RRF)
----------------------------
Combines several ranked result lists into a single fused ranking. Each list is
a list of dicts carrying at least an ``id``. Per-source ranks/scores are
recorded on every result so retrieval score tracking is transparent.
"""
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def rrf_fuse(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Fuse ``ranked_lists`` with Reciprocal Rank Fusion.

    Each element of a ranked list is a dict with an ``id`` and optionally
    ``text`` / ``metadata`` / ``score`` / ``scores``. Returned results are
    sorted by descending RRF score and augmented with:
      - ``rrf_score``      : final fusion score
      - ``rrf_contributions`` : per-list {rank, score} used in the fusion
      - ``rank``           : final 1-based rank
    """
    if not ranked_lists:
        return []

    rrf_scores: Dict[str, float] = {}
    contributions: Dict[str, Dict] = {}
    item_by_id: Dict[str, Dict] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for rank, item in enumerate(ranked):
            item_id = item.get("id")
            if item_id is None:
                logger.debug("Skipping RRF item without an id")
                continue
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            contributions.setdefault(item_id, {})[f"list{list_idx}"] = {
                "rank": rank + 1,
                "score": item.get("score"),
            }
            if item_id not in item_by_id:
                item_by_id[item_id] = item

    fused = []
    for item_id, score in rrf_scores.items():
        base = dict(item_by_id[item_id])  # shallow copy of first-seen item
        base["rrf_score"] = round(score, 6)
        base["rrf_contributions"] = contributions[item_id]
        base.setdefault("scores", {})["rrf"] = round(score, 6)
        fused.append(base)

    fused.sort(key=lambda e: e["rrf_score"], reverse=True)
    for i, entry in enumerate(fused):
        entry["rank"] = i + 1
    return fused