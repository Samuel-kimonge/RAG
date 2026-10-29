import requests
import math

sentences = [
    "Machine Learning is a branch of Artificial Intelligence",
    "Artificial Intelligence systems can learn patterns from data",
    "Chocolate cake is a famous dessert"
]


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": text
        }, timeout=120
    )
    response.raise_for_status()
    result = response.json()
    return result["embeddings"][0]


# Calculate cosine similarity between two vectors
def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude1 = math.sqrt(sum(a * a for a in vec_a))
    magnitude2 = math.sqrt(sum(b * b for b in vec_b))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


# Generate three embeddings
embeddings = [get_embedding(sentence) for sentence in sentences]

# compare every pair of embeddings
pairs = [(0, 1), (0, 2), (1, 2)]

for i, j in pairs:
    score = cosine_similarity(embeddings[i], embeddings[j])
    print(f"Similarity between sentences {i} and {j}: {score:.4f}")
