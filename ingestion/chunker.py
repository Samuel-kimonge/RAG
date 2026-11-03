import re


def chunk_text(text, chunk_size=1000, overlap=200):
    """_
    Split text into overlapping chunks
    chunk size and overlap are measured in characters
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero!")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >=0 and < chunk_size!")

# Normalize whitespace while retaining paragraph boundaries
    text = re.sub(r"[\t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
    # prefer ending at sentence, paragraph, boundary
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = text.rfind(". ", start, end)
            if boundary == -1:
                boundary = text.rfind(" ", start, end)
            if boundary == start + chunk_size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks
