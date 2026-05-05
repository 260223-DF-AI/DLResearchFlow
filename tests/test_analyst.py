"""
Unit Tests — Analyst Agent

Tests the analyst node using mocked Bedrock calls.
Validates structured output schema and confidence scoring.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.analyst import AnalysisResult, Citation


class TestAnalystAgent:
    """Tests for agents.analyst.analyst_node."""

    def _stub_result(self):
        return AnalysisResult(
            answer="Apollo 11 landed on July 20, 1969 [1].",
            citations=[Citation(source="apollo.pdf", page_number=12,
                                excerpt="Apollo 11 landed on July 20, 1969.")],
            confidence=0.88,
        )

    def test_returns_valid_analysis_result(self):
        """
        TODO:
        - Mock the Bedrock LLM invocation.
        - Call analyst_node with sample retrieved_chunks.
        - Assert the output parses into a valid AnalysisResult.
        """
        with patch("agents.analyst.ChatBedrock") as MockChat, \
             patch("agents.analyst._PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = self._stub_result()

            instance = MockChat.return_value
            instance.with_structured_output.return_value = MagicMock()

            # Ensure _PROMPT | <structured_llm> returns our chain
            mock_prompt.__or__.return_value = mock_chain

            from agents.analyst import analyst_node
            out = analyst_node({
                "question": "When did Apollo 11 land?",
                "plan": ["When did Apollo 11 land?"],
                "current_subtask_index": 0,
                "retrieved_chunks": [
                    {"content": "Apollo 11 landed on July 20, 1969.",
                     "source": "apollo.pdf", "page_number": 12,
                     "relevance_score": 9.1}
                ],
            })
        assert "analysis" in out
        assert out["confidence_score"] == 0.88
        assert out["analysis"]["citations"][0]["source"] == "apollo.pdf"

    def test_includes_citations(self):
        """
        TODO:
        - Assert the AnalysisResult contains at least one Citation.
        - Assert citation source matches a retrieved chunk source.
        """
        pass

    def test_confidence_within_range(self):
        """
        TODO:
        - Assert confidence_score is between 0.0 and 1.0.
        """
        pass

    def test_short_circuits_when_no_chunks(self):
        from agents.analyst import analyst_node
        out = analyst_node({
            "question": "x",
            "plan": ["x"],
            "current_subtask_index": 0,
            "retrieved_chunks": [],
        })
        assert out["confidence_score"] == 0.0
        assert out["analysis"]["citations"] == []