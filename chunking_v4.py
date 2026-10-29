with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

paragraphs = text.split("\n\n")
chunks = []

for i, paragraph in enumerate(paragraphs):

    paragraph = paragraph.strip()

    if not paragraph:
        continue

    chunk = {
        "id": "i",
        "text": "paragraph",
        "source": "knowledge.txt"
    }
    chunks.append(chunk)

print("Number of chunks:", len(chunks))
for chunk in chunks:
    print("ID:", chunk["id"])
    print("Source:", chunk["source"])
    print("Text:", chunk["text"])
