from config import SUMMARY_DIR, RELEVANCE_DIR, CLASSIFICATION_DIR, COMPARISON_DIR
from file_utils import load_json, save_json
from ollama_utils import json_with_ollama

SYSTEM_PROMPT = """
You are a cross-paper scientific comparison agent.

Return only valid JSON.
"""

def run_cross_paper_comparison(query):
    summary_file = SUMMARY_DIR / "all_summaries.json"
    relevance_file = RELEVANCE_DIR / "all_relevance_ranked.json"
    classification_file = CLASSIFICATION_DIR / "all_classifications.json"

    if not summary_file.exists():
        print("Missing summaries.")
        return None
    if not relevance_file.exists():
        print("Missing relevance ranked file.")
        return None
    if not classification_file.exists():
        print("Missing classifications.")
        return None

    summaries = load_json(summary_file)
    relevance = load_json(relevance_file)
    classifications = load_json(classification_file)

    user_prompt = f"""
User Query:
{query}

Summaries:
{summaries}

Relevance Ranking:
{relevance}

Classifications:
{classifications}

Return JSON exactly:
{{
  "best_paper_id": "",
  "best_paper_title": "",
  "comparison_table": [
    {{
      "paper_id": "",
      "title": "",
      "strengths": "",
      "weaknesses": "",
      "main_focus": "",
      "use_case": ""
    }}
  ],
  "overall_insights": "",
  "research_gaps": "",
  "next_search_direction": ""
}}
"""

    result = json_with_ollama(SYSTEM_PROMPT, user_prompt)
    save_json(COMPARISON_DIR / "cross_paper_comparison.json", result)

    print("\nDone. Cross-paper comparison completed.")
    return result


if __name__ == "__main__":
    try:
        from runtime_config import load_runtime_config
        runtime = load_runtime_config()
        query = str(runtime.get("query", "")).strip()
    except Exception:
        query = input("Enter user query: ").strip()

    run_cross_paper_comparison(query)
