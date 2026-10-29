
import requests
import chromadb

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"

DATABASE_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"


def get_embeddings(text):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        },
        timeout=30
    )
    response.raise_for_status()
    results = response.json()
    return results["embeddings"][0]


with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

paragraphs = text.split("\n\n")

chunks = []
for paragraph in paragraphs:
    paragraph = paragraph.strip()
    if not paragraph:
        continue

    chunks.append(
        {
            "id": str(len(chunks)),
            "text": paragraph,
            "source": "knowledge.txt"
        }
    )
if not chunks:
    raise ValueError("No text chunks found in the document")

print(f"Loaded {len(chunks)} chunks")

# Connect to the persistent chromadb
client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hsnw:space": "cosine"}
)

print(f"Connected to collection: {COLLECTION_NAME}")

# Remove old chunks from this chunk
collection.delete(
    where={"source": "knowledge.txt"}
)
print("Removed old chunks from this source")

# Generate embeddings and store the chunks
for index, chunk in enumerate(chunks, start=1):
    embedding = get_embeddings(chunk["text"])
    collection.upsert(
        ids=[chunk["id"]],
        documents=[chunk["text"]],
        metadatas=[{
            "source": [chunk["source"]],
            "chunk_index": index - 1
        }],
        embeddings=[embedding]
    )
    print(
        f"stored_chunk{index}/{len(chunks)}",
        f"(ID : {chunk['id']})"
    )
# Verify Database
print(f"Database location:{DATABASE_PATH}")
print(f"Collection name:{collection.name}")
print(f"Total records:{collection.count()}")
