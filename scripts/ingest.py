"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
    python scripts/ingest.py --input-dir ./data/fact_check --namespace fact-check-sources
"""

import argparse
import os
from dotenv import load_dotenv


import datetime
import hashlib
from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone


def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="primary-corpus",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()


def load_documents(input_dir: str) -> list:
    """Walk a directory and load every PDF and .txt file as Document objects.

    PyPDFLoader returns one Document per page (page_number lives in
    metadata['page']). TextLoader returns a single Document for the whole file,
    so we synthesize a page_number = 1 to keep the schema uniform.
    """
    docs = []
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    for path in root.rglob("*"):
        if path.suffix.lower() == ".pdf":
            for page_doc in PyPDFLoader(str(path)).load():
                page_doc.metadata["source"] = path.name
                page_doc.metadata["page_number"] = page_doc.metadata.get("page", 0) + 1
                docs.append(page_doc)
        elif path.suffix.lower() in (".txt", ".md"):
            for d in TextLoader(str(path), encoding="utf-8").load():
                d.metadata["source"] = path.name
                d.metadata["page_number"] = 1
                docs.append(d)

    print(f"Loaded {len(docs)} document pages from {input_dir}")
    return docs


def chunk_documents(documents: list) -> list:
    """Split each Document into ~800-character chunks with 100-char overlap.

    Overlap reduces the chance that a single fact gets split across the chunk
    boundary in a way that hurts retrieval recall.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    timestamp = datetime.datetime.utcnow().isoformat()
    for doc in documents:
        for i, sub in enumerate(splitter.split_documents([doc])):
            # Deterministic ID — re-running ingestion overwrites instead of
            # bloating the index with duplicates.
            raw_id = f"{sub.metadata['source']}::{sub.metadata['page_number']}::{i}"
            sub.metadata["chunk_id"] = hashlib.md5(raw_id.encode()).hexdigest()
            sub.metadata["timestamp"] = timestamp
            chunks.append(sub)

    print(f"Split into {len(chunks)} chunks")
    return chunks


def generate_embeddings(chunks: list) -> list:
    """Embed every chunk's text via Bedrock Titan Embeddings V2.

    Returns a list of (vector_id, embedding, metadata) tuples ready for upsert.
    The embedding model dimension MUST match your Pinecone index dimension —
    Titan Embeddings V2 is 1024-dim by default.
    """
    embedder = BedrockEmbeddings(
        model_id=os.environ.get("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        region_name=os.environ["AWS_REGION"],
    )
    texts = [c.page_content for c in chunks]
    vectors = embedder.embed_documents(texts)

    out = []
    for chunk, vec in zip(chunks, vectors):
        # Pinecone metadata must be JSON-scalar-compatible — coerce.
        metadata = {
            "content": chunk.page_content,           # store the text for retrieval
            "source": chunk.metadata["source"],
            "page_number": int(chunk.metadata["page_number"]),
            "chunk_id": chunk.metadata["chunk_id"],
            "timestamp": chunk.metadata["timestamp"],
        }
        out.append((chunk.metadata["chunk_id"], vec, metadata))
    return out


def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """Upsert in batches of 100 — Pinecone's recommended cap per request."""
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    BATCH = 100
    for start in range(0, len(embeddings), BATCH):
        batch = embeddings[start:start + BATCH]
        vectors = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in batch
        ]
        index.upsert(vectors=vectors, namespace=namespace)
    print(f"Upserted {len(embeddings)} vectors into namespace '{namespace}'")


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()

    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    upsert_to_pinecone(embeddings, args.namespace)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
