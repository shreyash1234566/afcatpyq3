import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from scripts.generate_questions_groq import GroqClient

client = GroqClient()
key = client.get_current_key()

for model_name in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": 'Reply with JSON: {"status": "ok"}'}],
        "response_format": {"type": "json_object"}
    }
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload
    )
    print(f"Model: {model_name} | Status: {r.status_code}")
    if r.status_code == 200:
        print("Response:", r.json()["choices"][0]["message"]["content"])
        break
    else:
        print("Error:", r.text[:200])
