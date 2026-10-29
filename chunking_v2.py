with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunk_size = 300

chunks = []

for start in range(0, len(text), chunk_size):
    end = start + chunk_size
    chunk = text[start:end]
    chunks.append(chunk)
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n --- chunk{i} ---")
    print(chunk)
