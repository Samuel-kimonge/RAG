import requests

text = "Machine Learning is a branch of Artificial Intelligence"

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": text
    }
)

result = response.json()
embedding = result["embeddings"][0]

print("Embedding type:", type(embedding))
print("Number of dimensions:", len(embedding))
print("First 10 value:", embedding[:10])
