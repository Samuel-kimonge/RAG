
#!/usr/bin/env python3

"""
Validate the golden retrieval evaluation dataset against ChromaDB.

Run from the project root:
    python evaluation/validate_dataset.py

Optional:
    python evaluation/validate_dataset.py --dataset evaluation/datasets/retrieval_test.json
    python evaluation/validate_dataset.py --db-path chromadb --collection knowledge_base
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DATASET = "evaluation/datasets/retrieval_test.json"
DEFAULT_DB_PATH = "chromadb"
DEFAULT_COLLECTION = "knowledge_base"

CHUNK_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate retrieval evaluation dataset and ChromaDB references."
    )

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=None)

    return parser.parse_args()


def load_dataset(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Dataset root must be a JSON object.")

    return dataset


def validate_structure(dataset, errors):
    required_fields = {
        "version",
        "name",
        "description",
        "collection",
        "embedding_model",
        "allowed_sources",
        "relevance_scale",
        "test_cases",
    }

    for field in sorted(required_fields):
        if field not in dataset:
            errors.append(f"Missing top-level field: {field}")

    allowed_sources = dataset.get("allowed_sources", [])

    if not isinstance(allowed_sources, list) or not all(
        isinstance(source, str) and source.strip()
        for source in allowed_sources
    ):
        errors.append("'allowed_sources' must contain non-empty strings.")
        allowed_sources = []

    if len(allowed_sources) != len(set(allowed_sources)):
        errors.append("'allowed_sources' contains duplicates.")

    test_cases = dataset.get("test_cases", [])

    if not isinstance(test_cases, list):
        errors.append("'test_cases' must be a list.")
        return [], allowed_sources

    seen_cases = set()

    for index, case in enumerate(test_cases):
        prefix = f"test_cases[{index}]"

        if not isinstance(case, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        case_id = case.get("id")

        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{prefix}: invalid case ID.")
            case_id = f"index-{index}"
        elif case_id in seen_cases:
            errors.append(f"Duplicate test case ID: {case_id}")
        else:
            seen_cases.add(case_id)

        if not isinstance(case.get("query"), str) or not case["query"].strip():
            errors.append(f"{case_id}: query must be non-empty.")

        if not isinstance(case.get("category"), str) or not case["category"].strip():
            errors.append(f"{case_id}: invalid category.")

        answerable = case.get("answerable")

        if not isinstance(answerable, bool):
            errors.append(f"{case_id}: answerable must be boolean.")

        chunks = case.get("relevant_chunks")

        if not isinstance(chunks, list):
            errors.append(f"{case_id}: relevant_chunks must be a list.")
            continue

        if answerable is True and not chunks:
            errors.append(f"{case_id}: answerable case has no relevant chunks.")

        if answerable is False and chunks:
            errors.append(f"{case_id}: unanswerable case must have no relevant chunks.")

        seen_chunk_ids = set()

        for chunk_index, chunk in enumerate(chunks):
            location = f"{case_id}, chunk {chunk_index}"

            if not isinstance(chunk, dict):
                errors.append(f"{location}: expected an object.")
                continue

            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not CHUNK_ID_PATTERN.fullmatch(chunk_id):
                errors.append(f"{location}: invalid 64-character hexadecimal chunk ID.")
                continue

            if chunk_id in seen_chunk_ids:
                errors.append(f"{location}: duplicate chunk ID within case.")

            seen_chunk_ids.add(chunk_id)

            relevance = chunk.get("relevance")

            if (
                isinstance(relevance, bool)
                or not isinstance(relevance, int)
                or relevance not in (1, 2)
            ):
                errors.append(f"{location}: relevance must be integer 1 or 2.")

    return [case for case in test_cases if isinstance(case, dict)], allowed_sources


def validate_chroma(dataset, cases, allowed_sources, db_path, collection_name, errors):
    if not db_path.exists():
        errors.append(f"ChromaDB directory does not exist: {db_path}")
        return 0, 0

    try:
        import chromadb
    except ImportError:
        errors.append("chromadb import failed. Activate your project .venv.")
        return 0, 0

    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(collection_name)
        records = collection.get(include=["metadatas"])
    except Exception as exc:
        errors.append(f"Could not access ChromaDB collection: {exc}")
        return 0, 0

    metadata_by_id = {
        chunk_id: metadata or {}
        for chunk_id, metadata in zip(
            records["ids"], records["metadatas"]
        )
    }

    allowed_source_set = set(allowed_sources)
    referenced_ids = set()

    for case in cases:
        case_id = case.get("id", "<unknown>")
        chunks = case.get("relevant_chunks", [])

        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            chunk_id = chunk.get("chunk_id")

            if not isinstance(chunk_id, str) or not CHUNK_ID_PATTERN.fullmatch(chunk_id):
                continue

            referenced_ids.add(chunk_id)

            metadata = metadata_by_id.get(chunk_id)

            if metadata is None:
                errors.append(
                    f"{case_id}: chunk ID not found in ChromaDB: {chunk_id}"
                )
                continue

            source = metadata.get("source")

            if not isinstance(source, str) or not source.strip():
                errors.append(f"{case_id}: missing source metadata for {chunk_id}")

            elif source not in allowed_source_set:
                errors.append(
                    f"{case_id}: source is not allowed: {source} "
                    f"(chunk ID: {chunk_id})"
                )

    return len(metadata_by_id), len(referenced_ids)


def main():
    args = parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()

    errors = []

    try:
        dataset = load_dataset(dataset_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"DATASET LOAD FAILED: {exc}")
        return 1

    cases, allowed_sources = validate_structure(dataset, errors)

    collection_name = (
        args.collection
        or dataset.get("collection")
        or DEFAULT_COLLECTION
    )

    if not isinstance(collection_name, str) or not collection_name.strip():
        errors.append("Invalid collection name.")
        collection_name = DEFAULT_COLLECTION

    total_chunks, unique_references = validate_chroma(
        dataset,
        cases,
        allowed_sources,
        db_path,
        collection_name,
        errors,
    )

    answerable = sum(case.get("answerable") is True for case in cases)
    unanswerable = sum(case.get("answerable") is False for case in cases)

    categories = Counter(
        case["category"]
        for case in cases
        if isinstance(case.get("category"), str)
    )

    print("\n========== RETRIEVAL DATASET VALIDATION ==========")
    print(f"Dataset:                {dataset_path}")
    print(f"ChromaDB:               {db_path}")
    print(f"Collection:             {collection_name}")
    print(f"Test cases:             {len(cases)}")
    print(f"Answerable:             {answerable}")
    print(f"Unanswerable:           {unanswerable}")
    print(f"Allowed sources:        {len(allowed_sources)}")
    print(f"Unique relevant chunks: {unique_references}")
    print(f"Chunks in ChromaDB:     {total_chunks}")

    print("\nCases by category:")

    for category, count in sorted(categories.items()):
        print(f"  {category:<22} {count}")

    print(f"\nValidation errors: {len(errors)}")

    if errors:
        print("\n========== VALIDATION ERRORS ==========")

        for number, error in enumerate(errors, 1):
            print(f"{number}. {error}")

        print("\nVALIDATION FAILED")
        return 1

    print("\nVALIDATION PASSED")
    print("Dataset structure, chunk IDs, and source references are valid.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
