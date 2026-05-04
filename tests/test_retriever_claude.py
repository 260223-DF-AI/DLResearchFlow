"""
Unit Tests — Retriever Agent

Tests the retriever node using mocked external API calls.
Validates output structure, re-ranking order, context compression,
and graceful handling of empty results.
"""

import pytest
from unittest.mock import patch, MagicMock, call

from agents.retriever import retriever_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockState(dict):
    """
    Simulates ResearchState, which needs BOTH attribute access (state.plan)
    and dict-style item access (state["scratchpad"], state["retrieved_chunks"]).
    Extending dict satisfies the item access; explicit attributes cover the rest.
    """

    def __init__(self, plan=None):
        super().__init__()
        self.plan = plan or ["What are the latest advances in transformer architectures?"]
        self["scratchpad"] = []


def _make_rerank_result(text: str, score: float, source: str = "doc.pdf", page: int = 1):
    """Build a mock object that mirrors a CohereRerank result entry."""
    result = MagicMock()
    result.document = {"text": text, "source": source, "page": page}
    result.relevance_score = score
    return result


# ---------------------------------------------------------------------------
# Shared fixture — patches every external API used by retriever_node
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_env(monkeypatch):
    """Inject all required environment variables."""
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")
    monkeypatch.setenv("PINECONE_REGION", "us-east-1")
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")


@pytest.fixture()
def api_mocks():
    """
    Patch every class/function imported at the top of agents.retriever so that
    no real network calls are made. Yields a dict of the most useful mock handles.

    Patch targets use the 'agents.retriever.<Name>' form — this is what Python's
    import system actually resolves when retriever_node runs, regardless of where
    the original symbol lives.
    """
    with (
        patch("agents.retriever.Pinecone") as MockPinecone,
        patch("agents.retriever.BedrockEmbeddings") as MockEmbeddings,
        patch("agents.retriever.PineconeVectorStore") as MockVectorStore,
        patch("agents.retriever.CohereRerank") as MockCohereRerank,
        patch("agents.retriever.create_deep_agent") as mock_create_agent,
        patch("agents.retriever.StateBackend"),
        patch("agents.retriever.create_summarization_tool_middleware"),
    ):
        # --- Pinecone client ---
        mock_pc = MagicMock()
        MockPinecone.return_value = mock_pc
        mock_pc.has_index.return_value = True          # skip index creation by default
        mock_pc.Index.return_value = MagicMock()

        # --- Vector store ---
        mock_vs = MagicMock()
        MockVectorStore.return_value = mock_vs
        mock_vs.similarity_search.return_value = []    # overridden per test

        # --- Compression agent ---
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        mock_agent.run.return_value = []               # overridden per test

        # --- Reranker ---
        mock_reranker = MagicMock()
        MockCohereRerank.return_value = mock_reranker
        mock_reranker.rerank.return_value = []         # overridden per test

        yield {
            "pinecone": mock_pc,
            "vector_store": mock_vs,
            "agent": mock_agent,
            "reranker": mock_reranker,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrieverAgent:
    """Tests for agents.retriever.retriever_node."""

    def test_returns_structured_chunks(self, mock_env, api_mocks):
        """
        retriever_node must return a dict whose 'retrieved_chunks' value is a
        list of dicts each containing: content, relevance_score, source, page_number.
        """
        reranked = [
            _make_rerank_result("Content A", score=0.9, source="doc1.pdf", page=1),
            _make_rerank_result("Content B", score=0.8, source="doc2.pdf", page=2),
        ]
        # The agent's compressed output feeds into the reranker.
        api_mocks["agent"].run.return_value = [MagicMock(), MagicMock()]
        api_mocks["reranker"].rerank.return_value = reranked

        result = retriever_node(MockState())

        assert "retrieved_chunks" in result
        assert len(result["retrieved_chunks"]) == 2
        for chunk in result["retrieved_chunks"]:
            assert "content" in chunk
            assert "relevance_score" in chunk
            assert "source" in chunk
            assert "page_number" in chunk

    def test_applies_reranking(self, mock_env, api_mocks):
        """
        Chunks must be sorted in descending relevance_score order regardless of
        the order in which the reranker returns them.
        """
        # Reranker returns lower-scored item first — node must sort.
        reranked = [
            _make_rerank_result("Low score content",  score=0.6),
            _make_rerank_result("High score content", score=0.95),
        ]
        api_mocks["agent"].run.return_value = [MagicMock(), MagicMock()]
        api_mocks["reranker"].rerank.return_value = reranked

        result = retriever_node(MockState())

        chunks = result["retrieved_chunks"]
        assert len(chunks) == 2
        assert chunks[0]["relevance_score"] >= chunks[1]["relevance_score"], (
            "Chunks must be sorted highest-score first"
        )

    def test_reranker_receives_compressed_context(self, mock_env, api_mocks):
        """
        The output of agent.run (the compressed context) must be forwarded to
        reranker.rerank — not the raw similarity-search results.
        """
        raw_docs = [MagicMock(), MagicMock(), MagicMock()]
        compressed = [MagicMock(), MagicMock()]          # fewer docs after compression

        api_mocks["vector_store"].similarity_search.return_value = raw_docs
        api_mocks["agent"].run.return_value = compressed
        api_mocks["reranker"].rerank.return_value = []

        state = MockState(plan=["Explain attention mechanisms"])
        retriever_node(state)

        # agent.run must be called with the similarity-search results
        api_mocks["agent"].run.assert_called_once_with(raw_docs)

        # reranker.rerank must be called with the *compressed* output, not raw_docs
        rerank_call_kwargs = api_mocks["reranker"].rerank.call_args
        assert rerank_call_kwargs.kwargs.get("documents") == compressed, (
            "reranker must receive compressed_context, not raw similarity results"
        )

    def test_applies_context_compression(self, mock_env, api_mocks):
        """
        The agent (compression step) must be called with the similarity-search
        results, and its output (compressed context) must flow into the reranker.
        """
        raw_docs = [MagicMock(page_content="Very long verbose document text " * 20)]
        compressed_docs = [MagicMock()]   # agent distils many docs to fewer

        api_mocks["vector_store"].similarity_search.return_value = raw_docs
        api_mocks["agent"].run.return_value = compressed_docs
        api_mocks["reranker"].rerank.return_value = [
            _make_rerank_result("Compressed summary", score=0.85)
        ]

        result = retriever_node(MockState())

        # Compression was applied: agent received raw docs
        api_mocks["agent"].run.assert_called_once_with(raw_docs)
        # Final output reflects the reranked, compressed result
        assert result["retrieved_chunks"][0]["content"] == "Compressed summary"

    def test_handles_empty_results(self, mock_env, api_mocks):
        """
        When Pinecone returns no matches the node must return an empty list
        without raising an exception.
        """
        api_mocks["vector_store"].similarity_search.return_value = []
        api_mocks["agent"].run.return_value = []
        api_mocks["reranker"].rerank.return_value = []

        result = retriever_node(MockState())

        assert "retrieved_chunks" in result
        assert result["retrieved_chunks"] == []

    def test_creates_index_when_missing(self, mock_env, api_mocks):
        """
        If the Pinecone index does not exist yet, the node must create it
        before proceeding.
        """
        api_mocks["pinecone"].has_index.return_value = False

        retriever_node(MockState())

        api_mocks["pinecone"].create_index.assert_called_once()
        create_kwargs = api_mocks["pinecone"].create_index.call_args.kwargs
        assert create_kwargs["name"] == "test-index"
        assert create_kwargs["dimension"] == 1024
        assert create_kwargs["metric"] == "cosine"

    def test_skips_index_creation_when_present(self, mock_env, api_mocks):
        """
        If the index already exists, create_index must NOT be called.
        """
        api_mocks["pinecone"].has_index.return_value = True

        retriever_node(MockState())

        api_mocks["pinecone"].create_index.assert_not_called()

    def test_uses_subtask_as_query(self, mock_env, api_mocks):
        """
        The last element of state.plan must be used as the similarity-search query.
        """
        subtask = "Summarise RLHF techniques from 2023"
        state = MockState(plan=["Step 1: gather papers", subtask])

        retriever_node(state)

        api_mocks["vector_store"].similarity_search.assert_called_once()
        call_kwargs = api_mocks["vector_store"].similarity_search.call_args
        assert call_kwargs.kwargs.get("query") == subtask or call_kwargs.args[0] == subtask

    def test_state_updated_with_retrieved_chunks(self, mock_env, api_mocks):
        """
        retriever_node must also write the chunks back into state["retrieved_chunks"]
        so downstream nodes in the graph can read them.
        """
        reranked = [_make_rerank_result("Chunk content", score=0.7)]
        api_mocks["agent"].run.return_value = [MagicMock()]
        api_mocks["reranker"].rerank.return_value = reranked

        state = MockState()
        retriever_node(state)

        assert "retrieved_chunks" in state
        assert len(state["retrieved_chunks"]) == 1
        assert state["retrieved_chunks"][0]["content"] == "Chunk content"

    def test_scratchpad_updated(self, mock_env, api_mocks):
        """
        The node must append a log entry to state["scratchpad"].
        """
        api_mocks["reranker"].rerank.return_value = [
            _make_rerank_result("X", score=0.5)
        ]
        api_mocks["agent"].run.return_value = [MagicMock()]

        state = MockState()
        retriever_node(state)

        assert len(state["scratchpad"]) == 1
        assert "Retrieved chunks" in state["scratchpad"][0]