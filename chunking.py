
with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n")

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n --- Chunk {i} ---")
    print(chunk)
