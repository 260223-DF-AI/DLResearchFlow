"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

import os
from dotenv import load_dotenv

import os
from dotenv import load_dotenv

from agents.state import ResearchState

from pinecone import Pinecone, ServerlessSpec

from langchain_aws import BedrockEmbeddings
from langchain_cohere import CohereRerank
from langchain_pinecone import PineconeVectorStore
# from langchain_text_splitters.retrievers import ContextualCompressionRetriever # moved to langchain_classic
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware

load_dotenv()


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    Args:
        state: ResearchState object containing the current plan and context.
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
        region_name=os.getenv("AWS_REGION")
    )

    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name = index_name,
            dimension = 1024,
            metric = "cosine",
            spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_REGION"))
        )
    index = pinecone.Index(name=index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        # namespace=namespace
        namespace="primary-corpus"
    )

    # Use Named Entity Recognition (NER) to extract key entities from the sub-task for more focused retrieval.
    # ner = spacy.load("en_core_web_sm")
    # entities = ner(subtask).ents
    # entity_names = [ent.text for ent in entities]


    results = vector_store.similarity_search(
        query=subtask,
        k=5, #TODO: Tune k based on retrieval quality and token budget.
        filter={} #TODO: Add metadata filters to narrow down results based on sub-task context.
    )

    # Contextual compression to reduce the number of retrieved chunks.
    backend = StateBackend()
    model = "anthropic.claude-3-sonnet-20240229-v1:0"
    agent = create_deep_agent(
        model=model,
        middleware=[
            create_summarization_tool_middleware(model, backend)
        ]
    )
    # Use the agent to compress the context.
    compressed_context = agent.run(retrieved_chunks)

    # Apply re-ranking to prioritize the most relevant results.
    reranker = CohereRerank(
        model="rerank-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    reranked_results = reranker.rerank(
        query=subtask,
        documents=compressed_context
    )

    # Process the reranked results.
    for result in reranked_results:
        retrieved_chunks.append({
            "content": result.document.get("text", ""),
            "relevance_score": result.relevance_score,
            "source": result.document.get("source", "Unknown"),
            "page_number": result.document.get("page", 0)
        })


    # for result in results:
    #     retrieved_chunks.append({
    #         "content": result.page_content,
    #         "relevance_score": result.metadata.get("relevance_score", 0),
    #         "source": result.metadata.get("source", "Unknown"),
    #         "page_number": result.metadata.get("page", 0)
    #     })



    # Sort by most relevant.
    retrieved_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)

    # Return updated state with retrieved_chunks populated.
    state["retrieved_chunks"] = retrieved_chunks

    # Log actions to the scratchpad.
    state["scratchpad"].append(f"Retrieved chunks for subtask '{subtask}': {len(retrieved_chunks)}")


    return state