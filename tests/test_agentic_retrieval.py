"""Phase 5 - Simple Agentic Retrieval tests (offline, deterministic)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.evidence_agent import (
    MAX_RETRIEVAL_ATTEMPTS,
    check_evidence_sufficiency,
    refine_query,
    significant_tokens,
)
from graph.state import GraphState
from graph.edges import route_after_evidence_check


_MOCK_DOC = {
    "content": (
        "Vodafone Idea Revenue from Operations Q1FY25: 105,083 Rs mn . "
        "Revenue from Operations Q1FY24: 106,555 Rs mn."
    ),
    "page_no": 11,
    "has_table": True,
}


def test_sufficient_evidence():
    decision = check_evidence_sufficiency(
        "What was Vodafone Idea's revenue in Q1FY25?", [_MOCK_DOC]
    )
    assert decision["sufficient"] is True
    assert decision["num_chunks"] == 1
    assert decision["refined_query"] is None


def test_insufficient_evidence_empty_chunks():
    decision = check_evidence_sufficiency(
        "What was Vodafone Idea's revenue in Q1FY25?", []
    )
    assert decision["sufficient"] is False
    assert decision["num_chunks"] == 0
    assert decision["refined_query"] is not None


def test_insufficient_when_tokens_not_covered():
    decision = check_evidence_sufficiency(
        "What are Vodafone Idea's Nigerian operations revenue figures?",
        [{"content": "The company reported strong growth in India", "page_no": 3}],
    )
    assert decision["sufficient"] is False


def test_refine_query_keeps_metric_and_period_focus():
    refined = refine_query(
        "How much did the subscriber base grow between Q1FY24 and Q1FY25?"
    )
    assert refined
    assert "subscriber" in refined
    assert len(refined.split()) <= 8


def test_refine_query_on_insufficient_evidence():
    decision = check_evidence_sufficiency("What is 4G subscriber additions?", [])
    assert decision["refined_query"] is not None
    assert "subscriber" in decision["refined_query"]


def test_significant_tokens_drops_stopwords():
    tokens = significant_tokens("What was the total revenue of the company?")
    assert "what" not in tokens
    assert "revenue" in tokens


def test_sufficient_evidence_proceeds_to_analyze():
    state = GraphState(
        user_query="What was Vodafone Idea's revenue in Q1FY25?",
        evidence_check={"sufficient": True},
        retrieval_attempts=1,
    )
    assert route_after_evidence_check(state) == "analyze"


def test_sufficient_numerical_question_goes_to_numerical():
    state = GraphState(
        user_query="What was the YoY revenue growth in Q1FY25?",
        evidence_check={"sufficient": True},
        retrieval_attempts=1,
    )
    assert route_after_evidence_check(state) == "numerical"


def test_insufficient_triggers_refinement_retrieval():
    state = GraphState(
        user_query="What was the company's debt in Q1FY25?",
        evidence_check={"sufficient": False},
        retrieval_attempts=1,
    )
    assert route_after_evidence_check(state) == "retrieve"


def test_max_attempts_respected_no_infinite_loop():
    state = GraphState(
        user_query="What was the company's debt in Q1FY25?",
        evidence_check={"sufficient": False},
        retrieval_attempts=MAX_RETRIEVAL_ATTEMPTS,
    )
    assert route_after_evidence_check(state) != "retrieve"
    assert route_after_evidence_check(state) == "analyze"
# ---------------- graph integration (offline) ----------------
class _MockDocGraph:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _MockCollectionGraph:
    def count(self):
        return 1

    def get(self, include=None):
        return {"ids": [], "documents": [], "metadatas": []}


class _ControlledVector:
    def __init__(self):
        self._collection = _MockCollectionGraph()
        self.calls = 0

    def _docs_for(self, query):
        self.calls += 1
        if self.calls <= 1:
            return []
        return [
            _MockDocGraph(
                "Vodafone Idea revenue from operations Q1FY25 105,083 earnings report",
                {"page_no": 11, "has_table": True},
            )
        ]

    def similarity_search(self, query, k=15):
        return self._docs_for(query)

    def similarity_search_with_score(self, query, k=15):
        return [(d, 0.5) for d in self._docs_for(query)]


class _AlwaysEmptyVector:
    def __init__(self):
        self._collection = _MockCollectionGraph()

    def similarity_search(self, query, k=15):
        return []

    def similarity_search_with_score(self, query, k=15):
        return []


def _build_offline_graph(vs, monkeypatch):
    import graph.graph as gg

    def _fake_decompose(state):
        return {"sub_questions": [state.user_query]}

    def _fake_analyze(state):
        if not (state.evidence_check or {}).get("sufficient") and (
            state.retrieval_attempts or 0
        ) >= MAX_RETRIEVAL_ATTEMPTS:
            return {
                "analysis_result": {},
                "final_answer": "I cannot verify this claim from the uploaded document.",
                "verification": {"status": "INSUFFICIENT EVIDENCE"},
            }
        return {
            "analysis_result": {
                "extracted_metrics": {},
                "derived_ratios": {},
                "missing_metrics": [],
                "pages_used": [c.get("page_no") for c in (state.retrieved_chunks or [])],
            }
        }

    def _fake_validate(state):
        return {"compliance_result": {"confidence_score": 0.5, "rule_engine_flags": []}}

    def _fake_summarize(state):
        if (state.verification or {}).get("status") == "INSUFFICIENT EVIDENCE":
            return {"final_answer": state.final_answer}
        return {"final_answer": "Mock answer for: " + state.user_query}

    monkeypatch.setattr(gg, "decompose_node", _fake_decompose)
    monkeypatch.setattr(gg, "analysis_node", _fake_analyze)
    monkeypatch.setattr(gg, "validate_node", _fake_validate)
    monkeypatch.setattr(gg, "summarize_node", _fake_summarize)
    return gg.build_graph(vs)


def test_graph_terminates_with_refinement(monkeypatch):
    vs = _ControlledVector()
    app = _build_offline_graph(vs, monkeypatch)
    result = app.invoke(
        {"user_query": "What was Vodafone Idea's total revenue in Q1FY25?"}
    )
    assert result["retrieval_attempts"] == 2
    assert result["retrieval_attempts"] <= MAX_RETRIEVAL_ATTEMPTS
    assert result["final_answer"] == (
        "Mock answer for: What was Vodafone Idea's total revenue in Q1FY25?"
    )
    assert vs.calls == 2


def test_graph_terminates_at_max_attempts(monkeypatch):
    vs = _AlwaysEmptyVector()
    app = _build_offline_graph(vs, monkeypatch)
    result = app.invoke(
        {"user_query": "What is the company's total debt and equity position?"}
    )
    assert result["retrieval_attempts"] == MAX_RETRIEVAL_ATTEMPTS
    assert result["final_answer"] == (
        "I cannot verify this claim from the uploaded document."
    )
