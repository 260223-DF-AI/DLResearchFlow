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


# Module-level singletons, lazily constructed on first use. Lazy init matters:
# both clients read env vars at construction time, so eager init at import time
# would force every caller (tests, scripts, Lambda cold start) to have AWS_REGION
# and PINECONE_API_KEY set *before* `from agents.retriever import ...` runs.
_embedder = None
_pinecone_index = None


def _get_embedder():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _embedder
    if _embedder is None:
        _embedder = BedrockEmbeddings(
            model_id=os.environ["BEDROCK_EMBEDDING_MODEL_ID"],
            region_name=os.environ["AWS_REGION"],
        )
    return _embedder

def _get_index():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
    return _pinecone_index

def _cos_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity for plain Python lists — avoids a numpy import."""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

def _compress(chunk_text: str, query: str, max_sentences: int = 4) -> str:
    """Keep the sentences whose embedding is closest to the query.

    Cheap, deterministic, and good enough to halve token usage on long
    chunks. Replace with an LLM-based compressor if you need higher recall.

    Worked well for our use-case in testing so keeping to avoid unnecessary LLM calls
    """
    sentences = [s.strip() for s in chunk_text.split(". ") if s.strip()]
    if len(sentences) <= max_sentences:
        return chunk_text
    embedder = _get_embedder()
    query_vec = embedder.embed_query(query)
    sent_vecs = embedder.embed_documents(sentences)
    scores = [_cos_sim(query_vec, sv) for sv in sent_vecs]
    top = sorted(range(len(sentences)), key=lambda i: -scores[i])[:max_sentences]
    top.sort()                                      # preserve original order
    return ". ".join(sentences[i] for i in top)

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
    print(state)
    plan = state.get("plan", [])
    print(plan)
    idx = state.get("current_subtask_index", 0)
    subtask = plan[idx] if plan else state["question"]

    # Query the Pinecone index with semantic search and metadata filters.
    pinecone = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
    )

    embeddings = BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"),
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

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from agents.retriever import retriever_node

    state = {
        "question": "Replace this with something your corpus actually contains.",
        "plan": ["Replace this with something your corpus actually contains."],
        "current_subtask_index": 0,
    }
    out = retriever_node(state)

    print("Got", len(out["retrieved_chunks"]), "chunks")
    for c in out["retrieved_chunks"]:
        print(f"  [{c['relevance_score']:.3f}] {c['source']}#p{c['page_number']}: "
              f"{c['content'][:80]}...")