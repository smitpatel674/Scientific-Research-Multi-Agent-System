import json
import requests
from config import OLLAMA_URL, OLLAMA_MODEL


def chat_with_ollama(system_prompt, user_prompt, temperature=0.2):
    url = f"{OLLAMA_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def extract_json_from_text(text):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from Ollama output:\n{text}")


def json_with_ollama(system_prompt, user_prompt, temperature=0.2):
    content = chat_with_ollama(system_prompt, user_prompt, temperature=temperature)
    return extract_json_from_text(content)
