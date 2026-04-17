import json
from pathlib import Path

CONFIG_DIR = Path("query_output")
CONFIG_DIR.mkdir(exist_ok=True)

RUNTIME_CONFIG_FILE = CONFIG_DIR / "runtime_config.json"


def save_runtime_config(query: str, top_k: int = 5, papers_per_source: int = 5, download_choice: str = "1"):
    data = {
        "query": query,
        "top_k": int(top_k),
        "papers_per_source": int(papers_per_source),
        "download_choice": str(download_choice),
    }
    with open(RUNTIME_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def load_runtime_config():
    if not RUNTIME_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"{RUNTIME_CONFIG_FILE} not found. Run orchestrator.py first."
        )
    with open(RUNTIME_CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
