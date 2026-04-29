"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


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
    """
    Load and return raw documents from the input directory.
    
    Args
    - input_dir (str): Path to directory containing PDF/text documents.

    Returns
    - List of Document objects with content and metadata (source filename, page number).
    """
    docs = []

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if filename.lower().endswith(".pdf"):
            # Supports PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
            loader = PyPDFLoader(file_path)
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
        elif filename.lower().endswith(".txt"):
            # Supports plain text files.
            with open(file_path, "r", encoding="utf-8") as f:
                # Treat as one page, will be chunked later in chunk_documents()
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": filename, "page": 0, "timestamp": os.path.getmtime(file_path)}))
    return docs


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    Args
    - documents (list): List of Document objects to be chunked.

    Returns
    - List of Document objects representing chunks, with enriched metadata.
    """
    # Use RecursiveCharacterTextSplitter or sentence-level splitting.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    # Attach chunk metadata (chunk_id, source, page_number, timestamp).
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    Args
    - chunks (list): List of Document objects representing chunks to embed.

    Returns
    - List of dicts with 'id', 'embedding', and 'metadata' for each chunk.
    """
    vector_embeddings = []

    # Use Bedrock Titan Embeddings.
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name="us-east-1"
    )

    # Process in batches for efficiency 
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i+100]
        vector_embeddings.extend([
            {"id": chunk.metadata.get("chunk_id"), "embedding": embedding, "metadata": chunk.metadata}
            for chunk, embedding in zip(
                batch,
                embeddings.embed_documents([chunk.page_content for chunk in batch])
            )
        ])

    return vector_embeddings



def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    Args
    - embeddings (list): List of dicts with 'id', 'embedding', and 'metadata' for each chunk.
    - namespace (str): Pinecone namespace to upsert into.
    """
    # Initialize the Pinecone client using env vars.
    # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
    )

    index_name = "dlresearchflow-index"
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name = index_name,
            dimension = 1536,
            metric = "cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pinecone.Index(name=index_name)

    # Upsert vectors with rich metadata into the specified namespace.
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )


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
