"""
Unit Tests — Retriever Agent

Tests the retriever node using mocked Pinecone calls.
Validates re-ranking behavior and output structure.
"""

from unittest.mock import patch, MagicMock

import pytest

from agents.retriever import retriever_node


class TestRetrieverAgent:
    """Tests for agents.retriever.retriever_node."""

    def test_returns_structured_chunks(self):
        """
        TODO:
        - Mock the Pinecone client's query method.
        - Call retriever_node with a sample state.
        - Assert the returned dict contains "retrieved_chunks".
        - Assert each chunk has: content, relevance_score, source, page_number.
        """
        with patch('agents.retriever.pinecone') as mock_pinecone:
            mock_pinecone.query.return_value = [
                {
                    'content': 'Sample content 1',
                    'relevance_score': 0.9,
                    'source': 'doc1.pdf',
                    'page_number': 1
                },
                {
                    'content': 'Sample content 2',
                    'relevance_score': 0.8,
                    'source': 'doc2.pdf',
                    'page_number': 2
                }
            ]
            # Call the retriever node
            result = retriever_node({'query': 'sample query'})

            # Assert the structure of the result
            assert 'retrieved_chunks' in result
            assert len(result['retrieved_chunks']) == 2

            # Assert each chunk has the required fields
            for chunk in result['retrieved_chunks']:
                assert 'content' in chunk
                assert 'relevance_score' in chunk
                assert 'source' in chunk
                assert 'page_number' in chunk

    def test_applies_reranking(self):
        """
        TODO:
        - Provide mock results in non-optimal order.
        - Assert that re-ranking reorders them by relevance.
        """
        with patch('agents.retriever.pinecone') as mock_pinecone:
            mock_pinecone.query.return_value = [
                {
                    'content': 'Sample content 2',
                    'relevance_score': 0.8,
                    'source': 'doc2.pdf',
                    'page_number': 2
                },
                {
                    'content': 'Sample content 1',
                    'relevance_score': 0.9,
                    'source': 'doc1.pdf',
                    'page_number': 1
                }
            ]
            # Call the retriever node
            result = retriever_node({'query': 'sample query'})

            # Assert the structure of the result
            assert 'retrieved_chunks' in result
            assert len(result['retrieved_chunks']) == 2

            # Assert that the chunks are ordered by relevance
            assert result['retrieved_chunks'][0]['relevance_score'] >= result['retrieved_chunks'][1]['relevance_score']

    def test_applies_context_compression(self):
        """
        TODO:
        - Provide a verbose mock chunk.
        - Assert the output chunk content is shorter / compressed.
        """
        with patch('agents.retriever.pinecone') as mock_pinecone:
            mock_pinecone.query.return_value = [
                {
                    'content': 'This is a very long and verbose content that needs to be compressed for better readability and performance.',
                    'relevance_score': 0.9,
                    'source': 'doc1.pdf',
                    'page_number': 1
                }
            ]
            # Call the retriever node
            result = retriever_node({'query': 'sample query'})

            # Assert the structure of the result
            assert 'retrieved_chunks' in result
            assert len(result['retrieved_chunks']) == 1

            # Assert that the content is compressed
            assert len(result['retrieved_chunks'][0]['content']) < len('This is a very long and verbose content that needs to be compressed for better readability and performance.')

    def test_handles_empty_results(self):
        """
        TODO:
        - Mock Pinecone returning zero matches.
        - Assert the node handles it gracefully (empty list, no crash).
        """
        with patch('agents.retriever.pinecone') as mock_pinecone:
            mock_pinecone.query.return_value = []
            # Call the retriever node
            result = retriever_node({'query': 'sample query'})

            # Assert the structure of the result
            assert 'retrieved_chunks' in result
            assert len(result['retrieved_chunks']) == 0
