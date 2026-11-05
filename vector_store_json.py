import requests
import numpy as np


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }, timeout=10
    )
    response.raise_for_status()
    results = response.json()
    return results["embeddings"][0]


def cosine_similarity(vec_a, vec_b):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    denominator = (
        np.linalg.norm(vec_a) *
        np.linalg.norm(vec_b)
        )
    if denominator == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denominator)


with open("knowledge.txt", "r") as f:
    text = f.read()

paragraphs = text.split("\n\n")

chunks = []
for paragraph in paragraphs:
    paragraph = paragraph.strip()
    if not paragraph:
        continue
    chunks.append({
        "id": len(chunks),
        "text": paragraph,
        "source": "knowledge.txt"
    })
print(f"Create embeddings for {len(chunks)} chunks")
for chunk in chunks:
    chunk["embedding"] = get_embedding(chunk["text"])
print("Embeddings created successfully")

# ask a question from the user
question = input("Ask a question:").strip()
if not question:
    raise ValueError("Question cannot be null!")
question_embedding = get_embedding(question)

# compare the question embedding with each chunk embedding
for chunk in chunks:
    chunk["score"] = cosine_similarity(
        question_embedding,
        chunk["embedding"]
        )
# rank chunks by similarity score
ranked_chunks = sorted(
    chunks,
    key=lambda chunk: chunk["score"],
    reverse=True)

# display the most relevant chunks
top_k = 3

print(f"\n Top {top_k} results for: {question}")

for chunk in ranked_chunks[:top_k]:
    print("\n" + "=" * 60)
    print(f"Chunk ID: {chunk['id']}")
    print(f"source: {chunk['source']}")
    print(f"similarity_score: {chunk['score']:4f}")
    print(f"text: {chunk['text']}")
