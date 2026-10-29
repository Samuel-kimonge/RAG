with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunk_size = 300
overlap = 50
chunks = []

start = 0
while start < len(text):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap

print("Number of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"\n --- chunk{i} ---")
    print(chunk)
