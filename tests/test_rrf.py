"""
Phase 2 - Reciprocal Rank Fusion (RRF) unit tests.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.rrf import rrf_fuse


def test_rrf_ranks_high_in_both_lists_first():
    list1 = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    list2 = [{"id": "b"}, {"id": "c"}, {"id": "d"}]
    fused = rrf_fuse([list1, list2], k=60)
    assert [e["id"] for e in fused] == ["b", "c", "a", "d"]


def test_rrf_records_score_and_contributions():
    fused = rrf_fuse([[{"id": "a"}, {"id": "b"}], [{"id": "b"}]], k=60)
    top = fused[0]
    assert "rrf_score" in top
    assert top["rrf_score"] > 0
    assert "rrf_contributions" in top
    assert "list0" in top["rrf_contributions"]
    assert "list1" in top["rrf_contributions"]


def test_rrf_preserves_item_fields():
    fused = rrf_fuse(
        [[{"id": "x", "text": "hello", "metadata": {"page_no": 9}, "scores": {"dense": 0.9}}]],
        k=60,
    )
    assert fused[0]["text"] == "hello"
    assert fused[0]["metadata"]["page_no"] == 9
    assert fused[0]["scores"]["dense"] == 0.9


def test_rrf_empty_inputs():
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


def test_rrf_rank_assignments_are_sequential():
    fused = rrf_fuse(
        [[{"id": "a"}, {"id": "b"}, {"id": "c"}], [{"id": "c"}, {"id": "d"}]],
        k=60,
    )
    for i, e in enumerate(fused):
        assert e["rank"] == i + 1