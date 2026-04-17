from pathlib import Path
from config import PARSED_DIR, RELEVANCE_DIR
from file_utils import load_json, save_json
from ollama_utils import json_with_ollama

SYSTEM_PROMPT = """
You are a scientific paper relevance evaluator.

Return only valid JSON.
Scores must be from 0 to 10.
"""

def score_paper(query, paper):
    # Reduced from 15k to 10k for faster local LLM processing
    paper_text = paper.get("full_text", "")[:10000]

    user_prompt = f"""
User Query:
{query}

Paper Title:
{paper.get("title", "")}

Paper Text:
{paper_text}

Return JSON exactly in this format:
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "title": "{paper.get('title', '')}",
  "query_relevance": 0,
  "technical_relevance": 0,
  "implementation_value": 0,
  "novelty_relevance": 0,
  "overall_score": 0,
  "reason": ""
}}
"""
    return json_with_ollama(SYSTEM_PROMPT, user_prompt)


def run_relevance_scoring(query):
    parsed_files = list(Path(PARSED_DIR).glob("*.json"))
    if not parsed_files:
        print("No parsed papers found. Run pdf_extract.py first.")
        return []

    results = []

    for file in parsed_files:
        paper = load_json(file)
        print(f"[RELEVANCE] {paper['file_name']} (Analysing with Ollama, please wait...)")
        result = score_paper(query, paper)
        save_json(RELEVANCE_DIR / f"{paper['paper_id']}.json", result)
        results.append(result)

    results = sorted(results, key=lambda x: x.get("overall_score", 0), reverse=True)
    save_json(RELEVANCE_DIR / "all_relevance_ranked.json", results)

    print("\nDone. Relevance scoring completed.")
    return results


if __name__ == "__main__":
    try:
        from runtime_config import load_runtime_config
        runtime = load_runtime_config()
        query = str(runtime.get("query", "")).strip()
    except Exception:
        query = input("Enter user query: ").strip()

    run_relevance_scoring(query)
