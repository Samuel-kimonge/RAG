from retrieval.sparse_store import SQLiteFTS5Index


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


def test_sparse_search_and_filters(tmp_path):
    collection = FakeCollection([
        make_record(
            "chunk-001",
            "Network switches are under maintenance in Nairobi.",
            "inventory.pdf",
        ),
        make_record(
            "chunk-002",
            "The server is running in cloud region B.",
            "server.pdf",
        ),
        make_record(
            "chunk-003",
            "Network switch installation and configuration guide.",
            "guide.docx",
            "docx",
        ),
    ])

    index = SQLiteFTS5Index(
        collection,
        db_path=tmp_path / "test.sqlite3",
    )

    results = index.search(
        "network switches maintenance",
        k=3,
    )

    assert results
    assert results[0].chunk_id == "chunk-001"

    filtered = index.search(
        "network switches maintenance",
        k=3,
        source="inventory.pdf",
    )

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "chunk-001"

    file_filtered = index.search(
        "network switch",
        k=3,
        file_type="docx",
    )

    assert len(file_filtered) == 1
    assert file_filtered[0].chunk_id == "chunk-003"


def test_incremental_sync(tmp_path):
    collection = FakeCollection([
        make_record(
            "chunk-001",
            "Original network configuration.",
            "network.pdf",
        ),
    ])

    index = SQLiteFTS5Index(
        collection,
        db_path=tmp_path / "test.sqlite3",
    )

    assert index.count() == 1

    # Unchanged documents should not be re-indexed.
    stats = index.sync_from_collection()

    assert stats["unchanged"] == 1
    assert stats["inserted"] == 0
    assert stats["updated"] == 0

    # Modify one document and introduce another.
    collection.records = [
        make_record(
            "chunk-001",
            "Updated network configuration and maintenance.",
            "network.pdf",
        ),
        make_record(
            "chunk-002",
            "New database backup procedure.",
            "backup.pdf",
        ),
    ]

    stats = index.sync_from_collection()

    assert stats["updated"] == 1
    assert stats["inserted"] == 1
    assert index.count() == 2

    updated = index.search("maintenance", k=5)

    assert updated
    assert updated[0].chunk_id == "chunk-001"

    # Removed Chroma chunks must disappear from the sparse index.
    collection.records = [
        make_record(
            "chunk-002",
            "New database backup procedure.",
            "backup.pdf",
        ),
    ]

    stats = index.sync_from_collection()

    assert stats["deleted"] == 1
    assert index.count() == 1
    assert index.search("maintenance", k=5) == []


def test_empty_query_terms_return_no_results(tmp_path):
    collection = FakeCollection([
        make_record(
            "chunk-001",
            "Network maintenance procedure.",
            "network.pdf",
        ),
    ])

    index = SQLiteFTS5Index(
        collection,
        db_path=tmp_path / "test.sqlite3",
    )

    assert index.search("--- !!!", k=5) == []
