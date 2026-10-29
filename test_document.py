with open("knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Type:", type(text))
print("Characters:", len(text))
print("First 200 characters:")
print(text[:200])
