import hashlib
import json
import time
from pathlib import Path

import chromadb
import requests

from ingestion.document_loader import (
    discover_documents,
    load_document
)
from ingestion.chunker import chunk_text

# configuration
DATA_DIR = Path("data")
CHROMADB_DIR = Path("chromadb")
COLLECTION_NAME = "knowledge_base"

EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 32

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".csv"
}


# Utility functions
def file_hash(file_path):
    """Generate a hash to identify document changes"""
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_chunk_id(source, document_hash, chunk_number):
    """Create a unique chunkId based on source,document_hash and chunk_index"""
    raw_id = f"{source}: {document_hash}: {chunk_number}"
    return hashlib.sha256(
        raw_id.encode("utf-8")).hexdigest()


# Generate embeddings
def get_embeddings(texts):
    if not texts:
        return []

    response = requests.post(
        EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=300

    )
    response.raise_for_status()
    results = response.json()
    embeddings = results.get("embeddings")

    if not embeddings:
        raise RuntimeError(
            "Ollama has not returned any embeddings."
            f"Response: {results}")
    if len(embeddings) != len(texts):
        raise RuntimeError(
            "Ollama embeddings do not match the number of input texts."
        )
    return embeddings


def embed_in_batches(texts, batch_size=EMBEDDING_BATCH_SIZE):
    """Embed a large list of texts into smaller batches"""
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        batch_embeddings = get_embeddings(batch)
        embeddings.extend(batch_embeddings)
        print(
            f"Embedded {min(start + batch_size, len(texts))}"
            f"/{len(texts)} chunks")
    return embeddings


def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMADB_DIR)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
            }
    )
    return collection


# Document processing functions
def process_document(file_path):
    """Load, chunk embed and upsert one chunk"""
    file_path = Path(file_path)
    source = str(file_path)
    document_hash = file_hash(file_path)

    # Load Extracted texts and metadata
    extracted_documents = load_document(file_path)
    chunk_records = []
    for extracted_document in extracted_documents:
        text = extracted_document["text"]
        metadata = extracted_document["metadata"]
        chunks = chunk_text(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )
        for chunk in chunks:
            chunk_records.append({
                "text": chunk,
                "metadata": metadata
            })
    if not chunk_records:
        print(f"No text extracted: {file_path}")
        return 0
    texts = [
        record["text"]
        for record in chunk_records
    ]
    embeddings = embed_in_batches(texts)
    ids = []
    metadatas = []
    for chunk_number, record in enumerate(chunk_records):
        chunk_id = create_chunk_id(
            source,
            document_hash,
            chunk_number
        )

        ids.append(chunk_id)

        # Chroma metadata must contain supported scalar values.
        metadata = dict(record["metadata"])

        metadata["document_hash"] = document_hash
        metadata["chunk_number"] = chunk_number

        metadatas.append(metadata)

    collection = get_collection()

    # Remove old chunks from this source before inserting new ones.
    collection.delete(
        where={
            "source": source
        }
    )

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"  Stored {len(texts)} chunks from {file_path.name}"
    )

    return len(texts)


# Main ingestion
def main():
    start_time = time.time()
    print("Starting multi_document ingestion...")
    print(f"Input directory: {DATA_DIR}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory does not exist: {DATA_DIR}")
    document_paths = [
        path
        for path in discover_documents(DATA_DIR)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    document_paths = sorted(document_paths)
    if not document_paths:
        print("No supported documents found")
        return
    print(f"Discovered {len(document_paths)} documents.")

    successful_documents = 0
    failed_documents = 0
    total_chunks = 0

    for index, file_path in enumerate(document_paths, start=1):
        print(
            f"\n[{index}/{len(document_paths)}] "
            f"Processing {file_path.name}"
        )
        try:
            chunks_created = process_document(file_path)

            total_chunks += chunks_created
            successful_documents += 1

        except Exception as error:
            failed_documents += 1

            print(
                f"  ERROR processing {file_path.name}: {error}"
            )

    elapsed_time = time.time() - start_time

    collection = get_collection()

    print("\n========== INGESTION SUMMARY ==========")
    print(f"Documents discovered: {len(document_paths)}")
    print(f"Documents processed: {successful_documents}")
    print(f"Documents failed: {failed_documents}")
    print(f"Chunks created: {total_chunks}")
    print(f"Total chunks in collection: {collection.count()}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("=====================================")


if __name__ == "__main__":
    main()
