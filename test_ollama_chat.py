
import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "qwen2.5:1.5b",
    "messages": [
        {
            "role": "user",
            "content": "What is Artificial Intelligence? Answer in two sentences."
        }
    ],
    "stream": False,
    "keep_alive": "5m"
}

response = requests.post(
    url,
    json=payload,
    timeout=300
)

print("HTTP status:", response.status_code)
response.raise_for_status()

result = response.json()

print("Done reason:", result.get("done_reason"))
print("Response content:", repr(
    result.get("message", {}).get("content", "")
))
