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
    return response.json()["embeddings"][0]


client = chromadb.PersistentClient(
    path=DATABASE_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
)
# Ask user for a question
query = input("Ask a question: ").strip()

if not query:
    raise ValueError("Query cannot be empty!")

query_embedding = get_embeddings(query)

# Retrieve the most relevant chunks
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, collection.count()),
    include=["metadatas", "distances", "documents"]
)

print("Retrieved documents:")
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]
ids = results["ids"][0]


for rank, (chunk_id, document, metadata, distance) in enumerate(
    zip(ids, documents, metadatas, distances),
    start=1
):
    similarity = 1 - distance
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {chunk_id}")
    print(f"Source: {metadata['source']}")
    print(f"Cosine distance: {distance:.4f}")
    print(f"Approx. cosine similarity: {similarity:.4f}")
    print(f"Text: {document}")
