"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

import os
import json

import boto3
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone

from agents.state import ResearchState

# Module-level singletons, lazily constructed on first use. Lazy init matters:
# both clients read env vars at construction time, so eager init at import time
# would force every caller (tests, scripts, Lambda cold start) to have AWS_REGION
# and PINECONE_API_KEY set *before* `from agents.retriever import ...` runs.
_embedder = None
_pinecone_index = None
_bedrock_runtime = None


def _get_embedder():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _embedder
    if _embedder is None:
        _embedder = BedrockEmbeddings(
            model_id=os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
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


def _get_bedrock_runtime():
    """Lazy-init so unit tests can monkeypatch before first call."""
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=os.environ["AWS_REGION"],
        )
    return _bedrock_runtime


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
    return []
    if not matches:
        return []

    documents = [
        match.get("metadata", {}).get("content", "")
        if isinstance(match, dict) else ""
        for match in matches
    ]

    body = {
        "api_version": 2,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(documents))
    }

    response = _get_bedrock_runtime().invoke_model(
        modelId="cohere.rerank-v3-5:0",
        body=json.dumps(body),
        # accept="application/json",
        # contentType="application/json",
    )
    payload = json.loads(response["body"].read())
    results = payload.get("results", []) if isinstance(payload, dict) else []

    reranked = []
    for result in results:
        idx = result.get("index")
        if isinstance(idx, int) and 0 <= idx < len(matches):
            reranked.append(matches[idx])
    return reranked or matches[:top_k]


def retriever_node(state: ResearchState) -> dict:
    """Retrieve and compress."""
    plan = state.get("plan", [])
    idx = state.get("current_subtask_index", 0)
    sub_task = plan[idx] if plan else state["question"]
    log = [f"[retriever] sub-task: {sub_task!r}"]

    # 1) embed + Pinecone semantic search ------------------------------------
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

    # 1.5) reranker via bedrock ----------------------------------------------
    # Optional, if retrieval quality looks weak use Cohere
    try:
        matches = _rerank_matches(sub_task, matches, top_k=5)
        log.append(f"[retriever] reranked to top {len(matches)} with Cohere")
    except Exception as e:
        log.append(f"[retriever] rerank skipped: {e!r}")


    # 2) compress + structure ------------------------------------------------
    # Pinecone returns matches sorted by cosine score; take top 5.
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
    from dotenv import load_dotenv
    load_dotenv()
    
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