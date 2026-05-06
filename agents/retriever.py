"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_aws import BedrockEmbeddings
import cohere

from agents.state import ResearchState

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

'''
def _get_bedrock_runtime():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ["AWS_REGION"],
        )
    return _bedrock_runtime
'''

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

def _rerank_matches(query: str, matches: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank Pinecone matches using Bedrock Cohere rerank."""
    co = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
    if not matches:
        return []

    documents = [match.get("metadata", {}).get("content", "") for match in matches]

    rankings = co.rerank(
        model=os.environ["COHERE_RERANK_MODEL_ID"],
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
    )

    # rankings is expected to have a "results" list of objects with
    # an "index" into "documents" and a "relevance_score".
    results = rankings.results
    if not results:
        return matches[:top_k]

    reranked = []
    for r in results[:top_k]:
        idx = r.index
        score = r.relevance_score
        if idx is None or idx < 0 or idx >= len(matches):
            continue
        m = matches[idx]
        # attach the reranker score for later use
        try:
            m["rerank_score"] = float(score) if score is not None else None
        except Exception:
            m["rerank_score"] = None
        reranked.append(m)

    return reranked or matches[:top_k]


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
    # print(state)
    plan = state.get("plan", [])
    # print(plan)
    idx = state.get("current_subtask_index", 0)
    sub_task = plan[idx] if plan else state["question"]
    log = [f"[retriever] sub-task: {sub_task!r}"]


    # Query the Pinecone index with semantic search and metadata filters.
    index = _get_index()
    query_vec = _get_embedder().embed_query(sub_task)

    raw = index.query(
        vector=query_vec,
        top_k=10,
        namespace="primary-corpus",
        include_metadata=True,
    )
    matches = raw.get("matches", []) if isinstance(raw, dict) else raw["matches"]
    log.append(f"[retriever] pinecone returned {len(matches)} candidates")

    if not matches:
        return {"retrieved_chunks": [], "scratchpad": log + ["[retriever] no matches"]}


    # Apply re-ranking to prioritize the most relevant results.
    try:
        matches = _rerank_matches(sub_task, matches, top_k=5)
        log.append(f"[retriever] reranked to top {len(matches)} with Cohere")
    except Exception as e:
        log.append(f"[retriever] rerank skipped: {e!r}")


    # Contextual compression to reduce the number of retrieved chunks.
    # Process the reranked results.
    chunks = []
    for match in matches[:5]:
        meta = match["metadata"]
        chunks.append({
            "content": _compress(meta["content"], sub_task),
            "relevance_score": float(match.get("score", 0.0)),
            "source": meta.get("source", "unknown"),
            "page_number": meta.get("page_number"),
        })
    log.append(f"[retriever] kept top {len(chunks)} by Pinecone score")
    return {"retrieved_chunks": chunks, "scratchpad": log}


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from agents.retriever import retriever_node

    state = {
        "question": "What are neural networks?",
        "plan": ["Understand the basics of neural networks."],
        "current_subtask_index": 0,
    }
    out = retriever_node(state)

    print("Got", len(out["retrieved_chunks"]), "chunks")
    for c in out["retrieved_chunks"]:
        print(f"  [{c['relevance_score']:.3f}] {c['source']}#p{c['page_number']}: "
              f"{c['content'][:80]}...")