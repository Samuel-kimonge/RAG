
import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import chromadb
import requests

from ingestion.document_loader import discover_documents, load_document
from ingestion.chunker import chunk_document


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIR = Path("data/raw")
CHROMADB_DIR = Path("chromadb")
MANIFEST_PATH = Path("data/processed/ingestion_manifest.sqlite3")

COLLECTION_NAME = "knowledge_base"

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80

EMBEDDING_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 100

MAX_RETRIES = 3
REQUEST_TIMEOUT = (5, 120)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMADB_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


def get_manifest():
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(MANIFEST_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            source TEXT PRIMARY KEY,
            document_hash TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    return connection


def bootstrap_manifest(collection, connection):
    """Import existing ChromaDB sources into the manifest once."""

    existing = connection.execute(
        "SELECT COUNT(*) FROM documents"
    ).fetchone()[0]

    if existing or collection.count() == 0:
        return

    logger.info("Initializing manifest from existing index.")

    records = collection.get(include=["metadatas"])
    grouped = {}

    for metadata in records["metadatas"]:
        source = metadata.get("source")
        document_hash = metadata.get("document_hash")

        if not isinstance(source, str):
            continue

        if not isinstance(document_hash, str):
            continue

        entry = grouped.setdefault(
            source,
            {"hash": document_hash, "count": 0},
        )

        if entry["hash"] == document_hash:
            entry["count"] += 1
        else:
            # Mixed versions: force reprocessing on next run.
            entry["hash"] = ""
            entry["count"] = 0

    connection.executemany(
        """
        INSERT OR REPLACE INTO documents
        (source, document_hash, chunk_count)
        VALUES (?, ?, ?)
        """,
        [
            (source, item["hash"], item["count"])
            for source, item in grouped.items()
        ],
    )

    connection.commit()


# --------------------------------------------------
# DOCUMENT UTILITIES
# --------------------------------------------------

def file_hash(path):
    """Calculate a SHA-256 hash without loading the whole file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def create_chunk_id(source, document_hash, chunk_number):
    """Generate a deterministic ID for every chunk."""

    value = f"{source}:{document_hash}:{chunk_number}"

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def clean_metadata(metadata):
    """Keep only metadata types supported by ChromaDB."""

    return {
        key: value
        for key, value in metadata.items()
        if isinstance(value, (str, int, float, bool))
    }


# --------------------------------------------------
# OLLAMA EMBEDDINGS
# --------------------------------------------------

def get_embeddings(texts, session):
    """Generate embeddings with retries and validation."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.post(
                OLLAMA_URL,
                json={
                    "model": EMBEDDING_MODEL,
                    "input": texts,
                },
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            embeddings = response.json()["embeddings"]

            if len(embeddings) != len(texts):
                raise ValueError(
                    "Embedding count does not match input count."
                )

            if not embeddings or any(
                not embedding
                or not all(
                    isinstance(value, (int, float))
                    for value in embedding
                )
                for embedding in embeddings
            ):
                raise ValueError("Invalid embedding response.")

            dimensions = {len(item) for item in embeddings}

            if len(dimensions) != 1:
                raise ValueError("Inconsistent embedding dimensions.")

            return embeddings

        except (
            requests.RequestException,
            ValueError,
            KeyError,
        ) as error:

            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding failed after {MAX_RETRIES} attempts: {error}"
                ) from error

            delay = 2 ** attempt

            logger.warning(
                "Embedding attempt %s failed: %s. Retrying in %ss.",
                attempt,
                error,
                delay,
            )

            time.sleep(delay)


def embed_in_batches(texts, session):
    """Embed large documents in manageable batches."""

    embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        embeddings.extend(
            get_embeddings(batch, session)
        )

    return embeddings


# --------------------------------------------------
# CHROMADB UTILITIES
# --------------------------------------------------

def get_source_records(collection, source):
    return collection.get(
        where={"source": source},
        include=["metadatas"],
    )


def upsert_in_batches(collection, records):
    """Write vectors in batches to ChromaDB."""

    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[
            start:start + UPSERT_BATCH_SIZE
        ]

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            embeddings=[item["embedding"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )


# --------------------------------------------------
# DOCUMENT PROCESSING
# --------------------------------------------------

def process_document(path, collection, connection, session):
    """Index one document without discarding its previous version prematurely."""

    source = path.relative_to(DATA_DIR).as_posix()

    current_hash = file_hash(path)

    previous = connection.execute(
        """
        SELECT document_hash, chunk_count
        FROM documents
        WHERE source = ?
        """,
        (source,),
    ).fetchone()
    old_records = get_source_records(collection, source)

    if (
        previous
        and previous[0] == current_hash
        and previous[1] > 0
        and len(old_records["ids"]) == previous[1]
        and all(
            metadata.get("document_hash") == current_hash
            for metadata in old_records["metadatas"]
        )
    ):
        logger.info("Unchanged: %s", source)
        return 0, "skipped"

    records = load_document(path)

    # Detect files modified while they were being processed.
    if file_hash(path) != current_hash:
        raise RuntimeError(
            f"Document changed during ingestion: {source}"
        )

    prepared = []
    chunk_number = 0

    for record_index, record in enumerate(records):
        chunks = chunk_document(
            record["text"],
            metadata={
                **record.get("metadata", {}),
                "source": source,
                "filename": path.name,
                "file_type": path.suffix.lower().lstrip("."),
                "document_hash": current_hash,
                "record_index": record_index,
            },
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        for chunk in chunks:
            metadata = {
                **chunk["metadata"],
                "chunk_index": chunk_number,
            }

            prepared.append({
                "id": create_chunk_id(
                    source,
                    current_hash,
                    chunk_number,
                ),
                "text": chunk["text"],
                "metadata": clean_metadata(metadata),
            })

            chunk_number += 1

    old_ids = set(old_records["ids"])

    if prepared:
        texts = [
            item["text"]
            for item in prepared
        ]

        embeddings = embed_in_batches(
            texts,
            session,
        )

        for item, embedding in zip(prepared, embeddings):
            item["embedding"] = embedding

        # Write the new version before removing the old one.
        upsert_in_batches(
            collection,
            prepared,
        )

    new_ids = {
        item["id"]
        for item in prepared
    }

    obsolete_ids = list(old_ids - new_ids)

    if obsolete_ids:
        collection.delete(ids=obsolete_ids)

    connection.execute(
        """
        INSERT OR REPLACE INTO documents
        (source, document_hash, chunk_count, indexed_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (source, current_hash, len(prepared)),
    )

    connection.commit()

    logger.info(
        "Indexed: %s | %s chunks",
        source,
        len(prepared),
    )

    return len(prepared), "processed"


# --------------------------------------------------
# STALE DOCUMENT CLEANUP
# --------------------------------------------------

def remove_deleted_documents(active_sources, collection, connection):
    """Remove indexed sources no longer present in the raw directory."""

    indexed_sources = {
        row[0]
        for row in connection.execute(
            "SELECT source FROM documents"
        ).fetchall()
    }

    stale_sources = indexed_sources - active_sources

    for source in stale_sources:
        existing = get_source_records(
            collection,
            source,
        )

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        connection.execute(
            "DELETE FROM documents WHERE source = ?",
            (source,),
        )

        logger.info("Removed deleted source: %s", source)

    connection.commit()

    return len(stale_sources)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {DATA_DIR}"
        )

    documents = sorted(
        path
        for path in discover_documents(DATA_DIR)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not documents:
        logger.warning(
            "No supported documents found. Existing index preserved."
        )
        return

    collection = get_collection()
    connection = get_manifest()

    try:
        bootstrap_manifest(collection, connection)

        processed = 0
        skipped = 0
        failed = []
        total_chunks = 0

        with requests.Session() as session:
            for path in documents:
                try:
                    chunks, status = process_document(
                        path,
                        collection,
                        connection,
                        session,
                    )

                    total_chunks += chunks

                    if status == "processed":
                        processed += 1
                    else:
                        skipped += 1

                except Exception as error:
                    failed.append((path, str(error)))

                    logger.exception(
                        "Failed to process: %s",
                        path,
                    )

            # Never remove stale sources after a partial ingestion run.
            if not failed:
                active_sources = {
                    path.relative_to(DATA_DIR).as_posix()
                    for path in documents
                }

                removed = remove_deleted_documents(
                    active_sources,
                    collection,
                    connection,
                )
            else:
                removed = 0
                logger.warning(
                    "Stale cleanup skipped because ingestion had failures."
                )

        print("\n========== INGESTION SUMMARY ==========")
        print("Discovered documents:", len(documents))
        print("Processed documents:", processed)
        print("Unchanged documents:", skipped)
        print("Total chunks processed:", total_chunks)
        print("Deleted sources:", removed)
        print("Failures:", len(failed))
        print("ChromaDB collection size:", collection.count())

        for path, error in failed:
            print(f"\nFAILED: {path}\n{error}")

        print("=======================================")

        if failed:
            raise RuntimeError(
                f"Ingestion completed with {len(failed)} failure(s)."
            )

    finally:
        connection.close()


if __name__ == "__main__":
    main()