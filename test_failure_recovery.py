
import hashlib
import json
from pathlib import Path

import chromadb
import build_index


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = "txt/update_recovery_test.txt"
FILE_PATH = Path("data/raw") / SOURCE
MANIFEST_PATH = Path("ingestion_manifest.json")


# ============================================================
# PRE-TEST VALIDATION
# ============================================================

if not FILE_PATH.is_file():
    raise FileNotFoundError(FILE_PATH)

client = chromadb.PersistentClient(path="chromadb")
collection = client.get_collection("knowledge_base")

with MANIFEST_PATH.open(encoding="utf-8") as file:
    original_manifest = json.load(file)

original_entry = original_manifest["documents"].get(SOURCE)

if not original_entry:
    raise RuntimeError("Test document is missing from the manifest.")

original_records = collection.get(
    where={"source": SOURCE},
    include=["documents", "metadatas"],
)

if not original_records["ids"]:
    raise RuntimeError("No existing ChromaDB records found.")

original_manifest_bytes = MANIFEST_PATH.read_bytes()

original_record_map = {
    record_id: {
        "document": document,
        "metadata": metadata,
    }
    for record_id, document, metadata in zip(
        original_records["ids"],
        original_records["documents"],
        original_records["metadatas"],
    )
}

print("Original indexed version:", original_entry["sha256"])
print("Original chunks:", len(original_records["ids"]))


# ============================================================
# MODIFY THE SOURCE TO VERSION 3
# ============================================================

version_3 = """
NexaTech Document Update and Recovery Test

Document ID: UPDATE-RECOVERY-001

Department: Machine Learning Engineering
System: Production RAG
Version: 3

This is version three of the test document.

It introduces updated information about model evaluation,
vector databases, retrieval optimization, embedding pipelines,
and production monitoring.

The ingestion system must preserve the previously indexed
version if embedding generation fails.
""".strip()

FILE_PATH.write_text(version_3, encoding="utf-8")

new_hash = hashlib.sha256(FILE_PATH.read_bytes()).hexdigest()

assert new_hash != original_entry["sha256"]

print("\nSource changed to Version 3.")


# ============================================================
# SIMULATE AN EMBEDDING FAILURE
# ============================================================

original_embed_function = build_index.embed_in_batches


def simulated_embedding_failure(texts):
    raise RuntimeError(
        "SIMULATED FAILURE: Ollama embedding service unavailable."
    )


build_index.embed_in_batches = simulated_embedding_failure

try:
    build_index.process_document(collection, FILE_PATH)

    raise AssertionError(
        "The simulated embedding failure did not occur."
    )

except RuntimeError as error:
    if "SIMULATED FAILURE" not in str(error):
        raise

    print("\nExpected embedding failure:", error)

finally:
    build_index.embed_in_batches = original_embed_function


# ============================================================
# VERIFY THE OLD INDEX AND MANIFEST ARE PRESERVED
# ============================================================

after_failure = collection.get(
    where={"source": SOURCE},
    include=["documents", "metadatas"],
)

after_failure_map = {
    record_id: {
        "document": document,
        "metadata": metadata,
    }
    for record_id, document, metadata in zip(
        after_failure["ids"],
        after_failure["documents"],
        after_failure["metadatas"],
    )
}

assert after_failure_map == original_record_map

assert MANIFEST_PATH.read_bytes() == original_manifest_bytes

with MANIFEST_PATH.open(encoding="utf-8") as file:
    manifest_after_failure = json.load(file)

assert manifest_after_failure["documents"][SOURCE] == original_entry

print("\nFAILURE PRESERVATION PASSED:")
print("- Previous ChromaDB records remain intact.")
print("- Previous document content remains intact.")
print("- Previous metadata remains intact.")
print("- Manifest remains unchanged.")
print("- Version 3 has not been indexed prematurely.")


# ============================================================
# RECOVER USING THE NORMAL INGESTION PIPELINE
# ============================================================

print("\nStarting normal ingestion recovery...")

build_index.main()


# ============================================================
# VERIFY SUCCESSFUL RECOVERY
# ============================================================

recovered = collection.get(
    where={"source": SOURCE},
    include=["documents", "metadatas"],
)

with MANIFEST_PATH.open(encoding="utf-8") as file:
    recovered_manifest = json.load(file)

recovered_entry = recovered_manifest["documents"][SOURCE]

actual_hash = hashlib.sha256(
    FILE_PATH.read_bytes()
).hexdigest()

assert recovered_entry["sha256"] == actual_hash
assert recovered_entry["sha256"] == new_hash
assert len(recovered["ids"]) == recovered_entry["chunks"]

assert all(
    metadata.get("document_hash") == new_hash
    for metadata in recovered["metadatas"]
)

assert all(
    "Version: 3" in document
    for document in recovered["documents"]
)

assert set(recovered["ids"]).isdisjoint(
    set(original_records["ids"])
)

print("\nRECOVERY VERIFICATION PASSED:")
print("- Version 3 was successfully indexed.")
print("- New document hash saved in manifest.")
print("- ChromaDB contains the new version.")
print("- Previous chunk IDs were replaced.")
print("- No duplicate old chunks remain.")
print("\nFINAL TOTAL CHUNKS:", collection.count())
