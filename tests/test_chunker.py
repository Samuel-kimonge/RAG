import pytest
from ingestion.chunker import chunk_document


def test_empty_text():
    assert chunk_document("") == []


def test_chunking():
    text = "\n\n".join(
        f"Section {i}. Production RAG needs reliable retrieval."
        for i in range(100)
    )
    chunks = chunk_document(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(c["text"].strip() for c in chunks)
    assert all(c["metadata"]["token_count"] <= 100 for c in chunks)


def test_metadata():
    chunks = chunk_document(
        "Security Policy\n\nAccess must be controlled.",
        metadata={"source": "security.pdf", "page_number": 2},
    )

    assert chunks[0]["metadata"]["source"] == "security.pdf"
    assert chunks[0]["metadata"]["page_number"] == 2
    assert chunks[0]["metadata"]["chunk_index"] == 0


def test_invalid_settings():
    with pytest.raises(ValueError):
        chunk_document("Some text", chunk_size=100, overlap=100)


def test_invalid_input():
    with pytest.raises(TypeError):
        chunk_document(123)