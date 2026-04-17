from pathlib import Path
from config import PARSED_DIR, RELEVANCE_DIR, CLASSIFICATION_DIR
from file_utils import load_json, save_json
from ollama_utils import json_with_ollama

SYSTEM_PROMPT = """
You are a scientific research paper classifier.

Allowed categories:
- highly_relevant
- partially_relevant
- background
- not_useful

Return only valid JSON.
"""

def classify_paper(query, paper, relevance_info):
    # Reduced to 10k for faster local LLM processing
    paper_text = paper.get("full_text", "")[:10000]

    user_prompt = f"""
User Query:
{query}

Paper Title:
{paper.get("title", "")}

Relevance Info:
{relevance_info}

Paper Text:
{paper_text}

Return JSON exactly:
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "title": "{paper.get('title', '')}",
  "category": "highly_relevant",
  "reason": ""
}}
"""
    return json_with_ollama(SYSTEM_PROMPT, user_prompt)


def run_paper_classification(query):
    parsed_files = list(Path(PARSED_DIR).glob("*.json"))
    if not parsed_files:
        print("No parsed papers found.")
        return []

    results = []

    for file in parsed_files:
        paper = load_json(file)
        relevance_file = RELEVANCE_DIR / f"{paper['paper_id']}.json"

        if not relevance_file.exists():
            print(f"Missing relevance file for {paper['paper_id']}")
            continue

        relevance_info = load_json(relevance_file)

        print(f"[CLASSIFY] {paper['file_name']}")
        result = classify_paper(query, paper, relevance_info)
        save_json(CLASSIFICATION_DIR / f"{paper['paper_id']}.json", result)
        results.append(result)

    save_json(CLASSIFICATION_DIR / "all_classifications.json", results)

    print("\nDone. Classification completed.")
    return results


if __name__ == "__main__":
    try:
        from runtime_config import load_runtime_config
        runtime = load_runtime_config()
        query = str(runtime.get("query", "")).strip()
    except Exception:
        query = input("Enter user query: ").strip()

    run_paper_classification(query)
