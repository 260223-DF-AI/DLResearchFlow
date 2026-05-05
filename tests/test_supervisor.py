"""
Unit Tests — Supervisor Graph

Tests the routing logic and conditional edges using mocked sub-agents.
"""

import os
from unittest.mock import patch, MagicMock

import pytest
from langgraph.errors import NodeInterrupt


class TestSupervisorRouting:
    """Tests for agents.supervisor routing and conditional edges."""

    def test_planner_decomposes_question(self):
        """
        TODO:
        - Mock the LLM call inside planner_node.
        - Assert it populates state["plan"] with a non-empty list.
        """
        pass

    def test_router_selects_retriever(self):
        """
        TODO:
        - Provide a state where the next sub-task requires retrieval.
        - Assert router() returns "retriever".
        """
        from agents.supervisor import router
        assert router({"plan": ["x"]}) == "retriever"

    def test_router_selects_analyst(self):
        """
        TODO:
        - Provide a state where retrieval is complete.
        - Assert router() returns "analyst".
        """
        from agents.supervisor import router
        assert router({
            "plan": ["x"],
            "retrieved_chunks": [{"content": "..."}],
        }) == "analyst"

    def test_router_selects_fact_checker(self):
        from agents.supervisor import router
        assert router({
            "plan": ["x"],
            "retrieved_chunks": [{"content": "..."}],
            "analysis": {"answer": "yes"},
        }) == "fact_checker"

    def test_critique_triggers_retry(self, monkeypatch):
        """
        TODO:
        - Set confidence below threshold, iteration < max.
        - Assert critique_node routes back for refinement.
        """
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "3")
        from agents.supervisor import critique_node
        out = critique_node({"confidence_score": 0.3, "iteration_count": 0,
                             "needs_hitl": False})
        assert out["iteration_count"] == 1
        assert out["retrieved_chunks"] == []      # state was cleared for retry

    def test_critique_triggers_hitl(self, monkeypatch):
        """
        TODO:
        - Set confidence below threshold, iteration >= max.
        - Assert critique_node triggers HITL interrupt.
        """
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "2")
        from agents.supervisor import critique_node
        with pytest.raises(NodeInterrupt):
            critique_node({"confidence_score": 0.3, "iteration_count": 2,
                           "needs_hitl": True})

    def test_critique_accepts_response(self, monkeypatch):
        """
        TODO:
        - Set confidence above threshold.
        - Assert critique_node routes to END.
        """
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "3")
        from agents.supervisor import critique_node, _critique_router
        out = critique_node({"confidence_score": 0.9, "iteration_count": 0,
                             "needs_hitl": False})
        # Should not raise.
        from langgraph.graph import END
        assert _critique_router({**out, "confidence_score": 0.9,
                                  "needs_hitl": False}) == END