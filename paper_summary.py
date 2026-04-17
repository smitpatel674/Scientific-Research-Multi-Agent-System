from pathlib import Path
from config import PARSED_DIR, SUMMARY_DIR
from file_utils import load_json, save_json
from ollama_utils import json_with_ollama

SYSTEM_PROMPT = """
You are a scientific paper summarization agent.

Return only valid JSON.
Be concise and precise.
"""

def summarize_paper(query, paper):
    # Reduced from 18k to 10k for faster local LLM processing
    paper_text = paper.get("full_text", "")[:10000]

    user_prompt = f"""
User Query:
{query}

Paper Title:
{paper.get("title", "")}

Paper Text:
{paper_text}

Return JSON exactly:
{{
  "paper_id": "{paper.get('paper_id', '')}",
  "title": "{paper.get('title', '')}",
  "problem": "",
  "core_idea": "",
  "method": "",
  "results": "",
  "limitations": "",
  "why_relevant_to_query": ""
}}
"""
    return json_with_ollama(SYSTEM_PROMPT, user_prompt)


def run_paper_summaries(query):
    parsed_files = list(Path(PARSED_DIR).glob("*.json"))
    if not parsed_files:
        print("No parsed papers found.")
        return []

    summaries = []

    for file in parsed_files:
        paper = load_json(file)
        summary = summarize_paper(query, paper)
        save_json(SUMMARY_DIR / f"{paper['paper_id']}.json", summary)
        summaries.append(summary)

    save_json(SUMMARY_DIR / "all_summaries.json", summaries)

    print("\nDone. Summaries created.")
    return summaries


if __name__ == "__main__":
    try:
        from runtime_config import load_runtime_config
        runtime = load_runtime_config()
        query = str(runtime.get("query", "")).strip()
    except Exception:
        query = input("Enter user query: ").strip()

    run_paper_summaries(query)
