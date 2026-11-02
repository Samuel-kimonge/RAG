

from ingestion.document_loader import (
    discover_documents,
    load_document
)

DATA_DIRECTORY = "data/raw"


def test_document_discovery():
    """Verify that documents are discovered."""

    documents = list(discover_documents(DATA_DIRECTORY))

    assert len(documents) > 0

    print(f"Discovered {len(documents)} documents")


def test_supported_file_types():
    """Verify that discovered documents have supported extensions."""

    documents = list(discover_documents(DATA_DIRECTORY))

    supported_extensions = {".pdf", ".docx", ".txt", ".csv"}

    for document in documents:
        assert document.suffix.lower() in supported_extensions


def test_document_extraction():
    """Verify that documents can be extracted."""

    documents = list(discover_documents(DATA_DIRECTORY))

    assert len(documents) > 0

    for document in documents:
        extracted_content = load_document(document)

        assert isinstance(extracted_content, list)
        assert len(extracted_content) > 0

        for item in extracted_content:
            assert "text" in item
            assert isinstance(item["text"], str)
            assert item["text"].strip() != ""
