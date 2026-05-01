"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

import os
from dotenv import load_dotenv

from agents.state import ResearchState

from pinecone import Pinecone, ServerlessSpec

from langchain_aws import BedrockEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    Args:
        state: ResearchState object containing the current plan and context.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.
    """
    retrieved_chunks = []

    # Extract the current sub-task from state["plan"].
    # TODO: Check if plan[-1] is the next or final sub-task, dependent on implementation in state.py/supervisor.py
    subtask = state.plan[-1] if state.plan else "N/A"

    # Query the Pinecone index with semantic search and metadata filters.
    pinecone = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
    )

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="us-east-1"
    )

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name = index_name,
            dimension = 1024,
            metric = "cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pinecone.Index(name=index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        # namespace=namespace
        namespace="primary-corpus"
    )

    results = vector_store.similarity_search(
        query=subtask,
        k=5, #TODO: Tune k based on retrieval quality and token budget.
        filter={} #TODO: Add metadata filters to narrow down results based on sub-task context.
    )

    # Apply context compression to reduce token noise.
    

    # Apply re-ranking to prioritize the most relevant results.
    

    # Return updated state with retrieved_chunks populated.
    state["retrieved_chunks"] = retrieved_chunks

    # Log actions to the scratchpad.
    state["scratchpad"].append(f"Retrieved chunks for subtask '{subtask}': {len(retrieved_chunks)}")


    return retrieved_chunks