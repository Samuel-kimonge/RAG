from langchain_core.documents import Document

from retrieval.hybrid_store import (
    HybridKnowledgeBase,
    reciprocal_rank_fusion,
)
from retrieval.vector_store import RetrievalResult


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def count(self):
        return len(self.records)

    def get(self, limit, offset, include):
        batch = self.records[offset:offset + limit]

        return {
            "ids": [item["id"] for item in batch],
            "documents": [item["content"] for item in batch],
            "metadatas": [item["metadata"] for item in batch],
        }


class FakeDenseStore:
    def __init__(self, records, dense_order):
        self.collection = FakeCollection(records)
        self.dense_order = dense_order

    def search(
        self,
        query,
        *,
        k=5,
        file_type=None,
        source=None,
    ):
        records_by_id = {
            item["id"]: item
            for item in self.collection.records
        }

        results = []

        for chunk_id in self.dense_order:
            item = records_by_id[chunk_id]
            metadata = item["metadata"]

            if file_type and metadata["file_type"] != file_type:
                continue

            if source and metadata["source"] != source:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    document=Document(
                        page_content=item["content"],
                        metadata=metadata,
                    ),
                    distance=0.2,
                )
            )

        return results[:k]


def make_record(chunk_id, content, source, file_type="pdf"):
    return {
        "id": chunk_id,
        "content": content,
        "metadata": {
            "source": source,
            "file_type": file_type,
            "filename": source,
        },
    }


def test_rrf_rewards_results_found_by_both_retrievers():
    scores = reciprocal_rank_fusion(
        dense_ids=["a", "b"],
        sparse_ids=["b", "c"],
        rrf_k=60,
    )

    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rrf_rejects_duplicate_ids():
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion(
            dense_ids=["a", "a"],
            sparse_ids=["b"],
        )


def test_rrf_rejects_invalid_configuration():
    import pytest

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(
            dense_ids=["a"],
            sparse_ids=["b"],
            rrf_k=0,
        )


def test_hybrid_search_preserves_sparse_only_results(tmp_path):
    records = [
        make_record(
            "a",
            "Network switch maintenance procedure.",
            "network.pdf",
        ),
        make_record(
            "b",
            "Cloud server backup procedure.",
            "server.pdf",
        ),
        make_record(
            "c",
            "Network switch installation guide.",
            "guide.pdf",
        ),
    ]

    dense_store = FakeDenseStore(
        records=records,
        dense_order=["a", "b"],
    )

    hybrid = HybridKnowledgeBase(
        dense_store=dense_store,
        sparse_db_path=tmp_path / "hybrid.sqlite3",
    )

    results = hybrid.search(
        "network switch",
        k=3,
    )

    by_id = {
        result.chunk_id: result
        for result in results
    }

    # The document found by both retrievers should rank first.
    assert results[0].chunk_id == "a"

    # Dense-only result remains available.
    assert by_id["b"].dense_rank is not None
    assert by_id["b"].sparse_rank is None

    # Sparse-only result must not be discarded.
    assert by_id["c"].dense_rank is None
    assert by_id["c"].sparse_rank is not None

    # Overlapping result records both ranking positions.
    assert by_id["a"].dense_rank is not None
    assert by_id["a"].sparse_rank is not None
    assert by_id["a"].fused_score > by_id["b"].fused_score


def test_hybrid_search_applies_metadata_filters(tmp_path):
    records = [
        make_record(
            "a",
            "Network switch maintenance procedure.",
            "inventory.pdf",
        ),
        make_record(
            "b",
            "Network switch installation guide.",
            "guide.docx",
            "docx",
        ),
    ]

    dense_store = FakeDenseStore(
        records=records,
        dense_order=["a", "b"],
    )

    hybrid = HybridKnowledgeBase(
        dense_store=dense_store,
        sparse_db_path=tmp_path / "filtered.sqlite3",
    )

    results = hybrid.search(
        "network switch",
        k=5,
        file_type="docx",
    )

    assert results
    assert all(
        result.metadata["file_type"] == "docx"
        for result in results
    )
    assert all(result.chunk_id == "b" for result in results)
