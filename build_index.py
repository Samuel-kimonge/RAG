
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import requests


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
CHROMADB_DIR = Path("chromadb")
COLLECTION_NAME = "knowledge_base"
MANIFEST_PATH = Path("ingestion_manifest.json")

EMBED_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
EMBEDDING_BATCH_SIZE = 32
VALIDATION_PAGE_SIZE = 1000

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv"}


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from ingestion.document_loader import discover_documents, load_document
from ingestion.chunker import chunk_document


# ============================================================
# DOCUMENT UTILITIES
# ============================================================

def file_hash(path):
    """Calculate a SHA-256 hash without loading the whole file."""

    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def chunk_id(source, document_hash, index):
    """Generate deterministic IDs for document chunks."""

    value = f"{source}:{document_hash}:{index}"

    return hashlib.sha256(value.encode()).hexdigest()


# ============================================================
# CHROMADB
# ============================================================

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMADB_DIR)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def existing_source_records(collection, source):
    return collection.get(
        where={"source": source},
        include=["metadatas"],
    )


def iter_collection_metadata(collection):
    """Read collection metadata in pages."""

    total = collection.count()

    for offset in range(0, total, VALIDATION_PAGE_SIZE):

        result = collection.get(
            include=["metadatas"],
            limit=VALIDATION_PAGE_SIZE,
            offset=offset,
        )

        for record_id, metadata in zip(
            result["ids"],
            result["metadatas"],
        ):
            yield record_id, metadata


# ============================================================
# MANIFEST MANAGEMENT
# ============================================================

def load_manifest():

    if not MANIFEST_PATH.exists():
        return None

    data = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("documents"), dict)
    ):
        raise ValueError(
            "Invalid ingestion manifest. "
            "Back it up and inspect it before proceeding."
        )

    for source, entry in data["documents"].items():

        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("sha256"), str)
            or not isinstance(entry.get("chunks"), int)
            or entry["chunks"] < 0
        ):
            raise ValueError(
                f"Invalid manifest entry: {source}"
            )

    return data


def save_manifest(manifest):
    """Write the manifest atomically."""

    temporary = MANIFEST_PATH.with_suffix(".tmp")

    temporary.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    temporary.replace(MANIFEST_PATH)


def bootstrap_manifest(collection, document_paths):
    """
    Recover manifest information from existing ChromaDB records.
    Existing embeddings are preserved without re-embedding.
    """

    print("\nInitializing manifest from existing ChromaDB index...")

    manifest = {
        "version": 1,
        "documents": {},
    }

    existing = {}

    for _, metadata in iter_collection_metadata(collection):

        if not metadata:
            continue

        source = metadata.get("source")
        digest = metadata.get("document_hash")

        if source and digest:

            entry = existing.setdefault(
                source,
                {
                    "hashes": set(),
                    "count": 0,
                },
            )

            entry["hashes"].add(digest)
            entry["count"] += 1

    for path in document_paths:

        source = path.relative_to(DATA_DIR).as_posix()
        entry = existing.get(source)

        if not entry:
            continue

        digest = file_hash(path)

        if entry["hashes"] == {digest}:

            manifest["documents"][source] = {
                "sha256": digest,
                "chunks": entry["count"],
                "indexed_at": None,
            }

            print(f"Preserved: {source}")

    save_manifest(manifest)

    return manifest


# ============================================================
# EMBEDDING
# ============================================================

def get_embeddings(texts):

    response = requests.post(
        EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts,
        },
        timeout=300,
    )

    response.raise_for_status()

    embeddings = response.json().get("embeddings")

    if (
        not embeddings
        or len(embeddings) != len(texts)
    ):
        raise RuntimeError(
            "Ollama returned an invalid number of embeddings."
        )

    dimensions = len(embeddings[0])

    if not dimensions:
        raise RuntimeError("Empty embedding vector.")

    for vector in embeddings:

        if (
            len(vector) != dimensions
            or not all(math.isfinite(value) for value in vector)
        ):
            raise RuntimeError(
                "Invalid embedding vector returned by Ollama."
            )

    return embeddings


def embed_in_batches(texts):

    embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):

        batch = texts[
            start:start + EMBEDDING_BATCH_SIZE
        ]

        embeddings.extend(
            get_embeddings(batch)
        )

        print(
            f"Embedded {len(embeddings)}/{len(texts)} chunks"
        )

    return embeddings


# ============================================================
# DOCUMENT INDEXING
# ============================================================

def is_unchanged(collection, source, digest, manifest):

    saved = manifest["documents"].get(source)

    if not saved:
        return False

    if saved.get("sha256") != digest:
        return False

    existing = existing_source_records(
        collection,
        source,
    )

    metadatas = existing["metadatas"]

    if len(metadatas) != saved.get("chunks"):
        return False

    return bool(metadatas) and all(
        metadata
        and metadata.get("document_hash") == digest
        for metadata in metadatas
    )


def process_document(collection, path):

    source = path.relative_to(DATA_DIR).as_posix()
    digest = file_hash(path)

    records = []

    for document in load_document(path):

        records.extend(
            chunk_document(
                document["text"],
                metadata=document["metadata"],
                chunk_size=CHUNK_SIZE,
                overlap=CHUNK_OVERLAP,
            )
        )

    if not records:
        raise ValueError(
            f"No text extracted from {path}"
        )

    texts = [
        record["text"]
        for record in records
    ]

    embeddings = embed_in_batches(texts)

    ids = []
    metadatas = []

    for index, record in enumerate(records):

        metadata = {
            **record["metadata"],
            "source": source,
            "filename": path.name,
            "file_type": path.suffix.lower().lstrip("."),
            "document_hash": digest,
            "chunk_index": index,
            "chunk_number": index,
        }

        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
            and isinstance(value, (str, int, float, bool))
        }

        ids.append(
            chunk_id(source, digest, index)
        )

        metadatas.append(metadata)

    # Keep old records until new embeddings have been generated.
    old_ids = existing_source_records(
        collection,
        source,
    )["ids"]

    # Write the new document version.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # Remove obsolete chunks only after successful upsert.
    new_ids = set(ids)

    obsolete_ids = [
        old_id
        for old_id in old_ids
        if old_id not in new_ids
    ]

    if obsolete_ids:
        collection.delete(ids=obsolete_ids)

    return len(records), digest


# ============================================================
# DELETION SYNCHRONIZATION
# ============================================================

def synchronize_deletions(
    collection,
    manifest,
    current_sources,
):
    """
    Remove indexed documents that no longer exist in data/raw.

    Only manifest-tracked documents are automatically deleted.
    """

    deleted_documents = 0
    deleted_chunks = 0

    print("\nChecking for deleted documents...")

    tracked_sources = list(
        manifest["documents"].keys()
    )

    for source in tracked_sources:

        if source in current_sources:
            continue

        print(f"DELETED DOCUMENT DETECTED: {source}")

        existing = existing_source_records(
            collection,
            source,
        )

        ids = existing["ids"]

        if ids:
            collection.delete(ids=ids)

        deleted_chunks += len(ids)

        # Remove the manifest entry after ChromaDB deletion.
        manifest["documents"].pop(source)

        save_manifest(manifest)

        deleted_documents += 1

        print(
            f"Removed {len(ids)} chunks for {source}"
        )

    print(
        f"Deletion synchronization complete: "
        f"{deleted_documents} documents, "
        f"{deleted_chunks} chunks removed."
    )

    return deleted_documents, deleted_chunks


# ============================================================
# INDEX INTEGRITY VALIDATION
# ============================================================

def validate_index(
    collection,
    manifest,
    current_files,
):
    """
    Check consistency between:
    - Current source files
    - Manifest entries
    - ChromaDB records
    """

    print("\n========== INDEX INTEGRITY VALIDATION ==========")

    issues = []

    indexed_by_source = {}

    records_without_source = 0

    # Read all ChromaDB records in pages.
    for _, metadata in iter_collection_metadata(collection):

        if not metadata:
            records_without_source += 1
            continue

        source = metadata.get("source")

        if not source:
            records_without_source += 1
            continue

        indexed_by_source.setdefault(
            source,
            [],
        ).append(metadata)

    manifest_documents = manifest["documents"]

    # Check every manifest entry.
    for source, entry in manifest_documents.items():

        records = indexed_by_source.get(
            source,
            [],
        )

        expected_chunks = entry["chunks"]
        expected_hash = entry["sha256"]

        # Check source file existence.
        path = current_files.get(source)

        if path is None:

            issues.append(
                f"Manifest references missing file: {source}"
            )

        else:

            current_hash = file_hash(path)

            if current_hash != expected_hash:

                issues.append(
                    f"File hash differs from manifest: {source}"
                )

        # Check chunk count.
        if len(records) != expected_chunks:

            issues.append(
                f"{source}: expected {expected_chunks} chunks, "
                f"found {len(records)}"
            )

        # Check document hashes.
        hashes = {
            metadata.get("document_hash")
            for metadata in records
        }

        if hashes != {expected_hash}:

            issues.append(
                f"{source}: inconsistent document hashes"
            )

        # Check chunk indexes.
        indexes = [
            metadata.get("chunk_index")
            for metadata in records
        ]

        expected_indexes = list(
            range(expected_chunks)
        )

        if (
            len(indexes) != expected_chunks
            or any(not isinstance(i, int) for i in indexes)
            or sorted(indexes) != expected_indexes
        ):

            issues.append(
                f"{source}: missing, duplicate or invalid chunk indexes"
            )

    # Find ChromaDB records not tracked by the manifest.
    for source in indexed_by_source:

        if source not in manifest_documents:

            issues.append(
                f"Untracked ChromaDB source: {source}"
            )

    if records_without_source:

        issues.append(
            f"{records_without_source} ChromaDB records "
            f"have missing metadata or source fields"
        )

    # Check current files missing from the manifest.
    for source in current_files:

        if source not in manifest_documents:

            issues.append(
                f"Current document missing from manifest: {source}"
            )

    # Report results.
    if issues:

        print("\nINTEGRITY VALIDATION FAILED")

        for issue in issues:
            print(f"[ERROR] {issue}")

        print(f"\nTotal issues: {len(issues)}")

        return False

    print("\nINTEGRITY VALIDATION PASSED")
    print("All tracked documents have matching hashes.")
    print("All chunk counts match the manifest.")
    print("All chunk indexes are complete and unique.")
    print("No untracked ChromaDB documents were found.")
    print("All current documents are represented in the manifest.")

    return True


# ============================================================
# MAIN INGESTION PIPELINE
# ============================================================

def main():

    started = time.time()

    collection = get_collection()

    if not DATA_DIR.is_dir():

        raise FileNotFoundError(
            f"Input directory does not exist: {DATA_DIR}"
        )

    paths = sorted(
        path
        for path in discover_documents(DATA_DIR)
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    # Safety protection: never automatically delete everything
    # because a scan unexpectedly returned zero documents.
    if not paths:

        raise RuntimeError(
            "No supported documents discovered. "
            "Ingestion and deletion synchronization aborted."
        )

    current_files = {
        path.relative_to(DATA_DIR).as_posix(): path
        for path in paths
    }

    current_sources = set(current_files.keys())

    manifest = load_manifest()

    if manifest is None:

        manifest = bootstrap_manifest(
            collection,
            paths,
        )

    processed = 0
    skipped = 0
    failed = 0
    embedded_chunks = 0

    print(f"\nDiscovered {len(paths)} documents.")

    # --------------------------------------------------------
    # STEP 1: INDEX NEW AND MODIFIED DOCUMENTS
    # --------------------------------------------------------

    for index, path in enumerate(paths, start=1):

        source = path.relative_to(DATA_DIR).as_posix()

        try:

            digest = file_hash(path)

            if is_unchanged(
                collection,
                source,
                digest,
                manifest,
            ):

                skipped += 1

                print(
                    f"[{index}/{len(paths)}] UNCHANGED: {source}"
                )

                continue

            print(
                f"\n[{index}/{len(paths)}] INDEXING: {source}"
            )

            count, digest = process_document(
                collection,
                path,
            )

            manifest["documents"][source] = {
                "sha256": digest,
                "chunks": count,
                "indexed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            save_manifest(manifest)

            processed += 1
            embedded_chunks += count

        except Exception as error:

            failed += 1

            print(
                f"ERROR: {source}: {error}"
            )

    # --------------------------------------------------------
    # STEP 2: SYNCHRONIZE DELETIONS
    # --------------------------------------------------------

    deleted_documents, deleted_chunks = synchronize_deletions(
        collection,
        manifest,
        current_sources,
    )

    # --------------------------------------------------------
    # STEP 3: VALIDATE INDEX INTEGRITY
    # --------------------------------------------------------

    integrity_ok = validate_index(
        collection,
        manifest,
        current_files,
    )

    # --------------------------------------------------------
    # STEP 4: FINAL REPORT
    # --------------------------------------------------------

    print("\n========== INGESTION SUMMARY ==========")

    print(f"Discovered: {len(paths)}")
    print(f"New/modified documents indexed: {processed}")
    print(f"Unchanged documents skipped: {skipped}")
    print(f"Failed documents: {failed}")
    print(f"Chunks embedded this run: {embedded_chunks}")

    print(f"Deleted documents: {deleted_documents}")
    print(f"Deleted chunks: {deleted_chunks}")

    print(
        f"Total documents in manifest: "
        f"{len(manifest['documents'])}"
    )

    print(
        f"Total chunks in ChromaDB: "
        f"{collection.count()}"
    )

    print(
        f"Integrity validation: "
        f"{'PASSED' if integrity_ok else 'FAILED'}"
    )

    print(
        f"Elapsed: {time.time() - started:.2f} seconds"
    )

    print("=======================================")

    if failed or not integrity_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
