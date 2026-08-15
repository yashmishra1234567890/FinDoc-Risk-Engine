"""
Phase 2 - Reranker unit tests.

Uses only stub models so tests never download a cross-encoder.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.reranker import CrossEncoderReranker, NoopReranker, build_reranker, Reranker


def test_noop_preserves_order_and_truncates():
    candidates = [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]
    out = NoopReranker().rerank("q", candidates, top_k=1)
    assert len(out) == 1
    assert out[0]["id"] == "a"
    # rerank score is recorded as None for transparency
    assert out[0]["scores"]["rerank"] is None


def test_cross_encoder_reranks_by_score():
    # instantiate without calling the real __init__ (avoids model download)
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "stub"

    class StubModel:
        def predict(self, pairs):
            # longer text scores higher
            return [len(p[1]) for p in pairs]

    reranker.model = StubModel()

    candidates = [{"id": "a", "text": "short"}, {"id": "b", "text": "a much much longer context"}]
    out = reranker.rerank("q", candidates, top_k=2)
    assert out[0]["id"] == "b"  # re-ranked to the top
    assert "rerank" in out[0]["scores"]
    assert out[0]["scores"]["rerank"] is not None


def test_cross_encoder_top_k():
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model_name = "stub"

    class StubModel:
        def predict(self, pairs):
            return list(range(len(pairs)))

    reranker.model = StubModel()
    candidates = [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}, {"id": "c", "text": "z"}]
    out = reranker.rerank("q", candidates, top_k=2)
    assert len(out) == 2


def test_base_reranker_raises():
    try:
        Reranker().rerank("q", [], top_k=1)
        assert False, "should raise"
    except NotImplementedError:
        pass


def test_build_reranker_disabled_returns_noop():
    assert isinstance(build_reranker(enabled=False, model_name="anything"), NoopReranker)


def test_build_reranker_degrades_on_load_failure(monkeypatch):
    import retrieval.reranker as rr

    def _fail_init(self, model_name=None, device=None):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(rr.CrossEncoderReranker, "__init__", _fail_init)
    reranker = build_reranker(enabled=True, model_name="unavailable-model")
    assert isinstance(reranker, NoopReranker)