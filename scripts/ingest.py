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

    # NOTE: Processing 1990-Elman.pdf gives the following warning due to its formatting:
    # "Ignoring wrong pointing object 0 0 (offset 0)"

    print(f"📂 Loading documents from: {input_dir}")
    docs = []

    for filename in os.listdir(input_dir):
        print(f"📄 Processing file: {filename}")
        file_path = os.path.join(input_dir, filename)
        if filename.lower().endswith(".pdf"):
            # Supports PDF files (using LangChain's PyPDFLoader).
            loader = PyPDFLoader(
                file_path,
                mode="single"
            )
            pdf_docs = loader.load()
            docs.extend(pdf_docs)
        elif filename.lower().endswith(".txt"):
            # Supports plain text files.
            with open(file_path, "r", encoding="utf-8") as f:
                # Treat as one page, will be chunked later in chunk_documents()
                content = f.read()
                # print(f"📄 Loading document: {filename}")
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
    print(f"✂️ Chunking {len(documents)} documents into smaller pieces...")
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
    print(f"🔍 Generating embeddings for {len(chunks)} chunks...")
    vector_embeddings = []

    # Use Bedrock Titan Embeddings.
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name=os.getenv("AWS_REGION")
    )

    # Process in batches for efficiency 
    for i in range(0, len(chunks), 96):
        print(f"⏳ Processing batch {i//96 + 1} of {((len(chunks)-1)//96) + 1}...")
        batch = chunks[i:i+96]
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
    print(f"🚀 Upserting {len(embeddings)} vectors into Pinecone namespace '{namespace}'...")
    # Initialize the Pinecone client using env vars.
    # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    pinecone = Pinecone(
        api_key = os.getenv("PINECONE_API_KEY")
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

    # Upsert vectors with rich metadata into the specified namespace.

    for i in range(0, len(embeddings), 96):
        print(f"⏳ Upserting batch {i//96 + 1} of {((len(embeddings)-1)//96) + 1}...")
        batch = embeddings[i:i+96]
        index.upsert(
            vectors=[(str(item["id"]), item["embedding"], item["metadata"]) for item in batch],
            namespace=namespace
        )

    '''
    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings,
        namespace=namespace
    )

    vector_store.add_vectors(embeddings)
    '''


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
