
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_document(text, metadata=None, chunk_size=450, overlap=80):
    """Create token-aware chunks with reliable character offsets."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Invalid chunk_size or overlap")

    text = text.strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=lambda value: len(_ENCODING.encode(value)),
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=False,
    )

    documents = splitter.create_documents([text])
    records, search_from = [], 0

    for index, document in enumerate(documents):
        chunk = document.page_content
        start = text.find(chunk, search_from)

        if start < 0:
            raise ValueError(f"Could not locate chunk {index} in source text")

        records.append({
            "text": chunk,
            "metadata": {
                **(metadata or {}),
                "start_index": start,
                "chunk_index": index,
                "token_count": len(_ENCODING.encode(chunk)),
            },
        })
        search_from = start + 1

    return records


def chunk_text(text, chunk_size=450, overlap=80):
    return [
        chunk["text"]
        for chunk in chunk_document(text, chunk_size, overlap=overlap)
    ]

