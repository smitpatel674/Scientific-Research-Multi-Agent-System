import json
import os
from pathlib import Path
from typing import List, Optional
import requests
from config import RELEVANCE_DIR, CLASSIFICATION_DIR, SUMMARY_DIR, COMPARISON_DIR, REPORT_DIR, BASE_DIR, OLLAMA_URL, OLLAMA_MODEL
from file_utils import load_json, save_json
from ollama_utils import json_with_ollama

REASONING_DIR = BASE_DIR / "research_outputs" / "reasoning"
REASONING_DIR.mkdir(parents=True, exist_ok=True)

FINAL_REASONING_FILE = REASONING_DIR / "final_reasoning.json"

SYSTEM_PROMPT = """
You are an Expert Multi-Domain Final Reasoning Engine.

Your task is to generate the FINAL ANSWER from selected research evidence.

-----------------------------------
🎯 CORE OBJECTIVE
Convert evidence → correct, domain-aligned final answer.

-----------------------------------
🧠 STEP 1: IDENTIFY QUERY TARGET TYPE (CRITICAL)
Determine WHAT the user is asking for:
- model / architecture (e.g., MobileNet, LightGBM, EfficientNet, DistilBERT)
- method / algorithm
- strategy / approach
- treatment / intervention
- system / design

🚨🚨🚨 CRITICAL DISTINCTION — READ CAREFULLY:
"architecture" in AI/ML means a MODEL DESIGN (neural network, algorithm, framework).
It does NOT mean hardware (CPU, GPU, TPU).

If the query says "which architecture uses less GPU":
- "architecture" = model/algorithm design
- "GPU" = the constraint (compute resource)
- The answer must be a MODEL that runs efficiently on limited GPU

CORRECT answers for "which architecture uses less GPU and data":
✅ LightGBM, XGBoost (tree-based, no GPU needed)
✅ MobileNet, EfficientNet (lightweight CNNs)
✅ DistilBERT, TinyBERT (efficient transformers)
✅ Dynamic Compositional Architecture (DCA)
✅ Logistic Regression, SVM (classical ML)

WRONG answers:
❌ CPU (this is hardware, not a model)
❌ GPU (this is hardware)
❌ TPU, NPU (hardware accelerators)

-----------------------------------
🧩 STEP 2: EXTRACT CONSTRAINTS
- low cost → cost-effective
- less data → data-efficient / sample-efficient
- less compute/GPU → compute-efficient / low-resource
- fast → low latency

-----------------------------------
📚 STEP 3: FILTER EVIDENCE (STRICT)
Keep ONLY evidence about models, algorithms, or methods.
Reject evidence that ONLY compares hardware without discussing model efficiency.

-----------------------------------
⚖️ STEP 4: NORMALIZE CANDIDATES
Extract for each valid candidate:
- name (must be a model/method name, NOT hardware)
- type
- resource efficiency
- data requirement
- performance quality
- evidence strength

-----------------------------------
🚨 STEP 5: HARD SAFETY RULES
RULE 1: If query asks for model/architecture → answer MUST be a model/algorithm name. NEVER output CPU, GPU, TPU, NPU.
RULE 2: If evidence is weak → say "Current evidence does not directly answer the query".
RULE 3: If papers discuss hardware comparisons, extract the MODEL insights, not the hardware conclusion.

Example: If a paper says "CPUs use less GPU than GPUs" → this is a hardware fact, NOT an architecture recommendation.
But if a paper says "DCA achieves 40% faster inference and 30% memory reduction" → THIS is a valid architecture answer.

-----------------------------------
🧠 STEP 6: DECISION LOGIC
Rank candidates by: relevance, constraint match, evidence strength, applicability.

-----------------------------------
📊 STEP 7: FINAL ANSWER
Produce: best overall answer (MUST be a model/method), best by scenario, clear reasoning.

-----------------------------------
📦 STEP 8: OUTPUT FORMAT (STRICT JSON)
{
  "query_target": "str",
  "constraints": ["str"],
  "valid_evidence_count": int,
  "discarded_evidence_count": int,
  "discard_reasons": ["str"],
  "best_overall_answer": "str (MUST be a model/method name, NOT hardware)",
  "best_by_scenario": [
    {"scenario": "str", "best_choice": "str", "why": "str"}
  ],
  "reasoning_summary": "str",
  "confidence": "high / medium / low",
  "warning": "str"
}

-----------------------------------
📌 STEP 9: DOMAIN-AWARE LANGUAGE
AI → model/architecture, Finance → risk/strategy, Medical → treatment.

-----------------------------------
📌 STEP 10: FINAL VALIDATION
Before outputting, verify:
- Is best_overall_answer a MODEL/METHOD name? (not CPU/GPU/TPU)
- Does it match the query target type?
- Are constraints addressed?
If validation fails, revise the answer.
"""


def load_all_evidence():
    """Load and aggregate all available research evidence."""
    query_output_dir = Path("query_output")
    
    def safe_load(path, default=None):
        if Path(path).exists():
            return load_json(path)
        return default if default is not None else []

    evidence = {
        "relevance": safe_load(RELEVANCE_DIR / "all_relevance_ranked.json", []),
        "classifications": safe_load(CLASSIFICATION_DIR / "all_classifications.json", []),
        "summaries": safe_load(SUMMARY_DIR / "all_summaries.json", []),
        "comparison": safe_load(COMPARISON_DIR / "cross_paper_comparison.json", {}),
        "query_data": safe_load(query_output_dir / "query_data.json", {}),
    }
    return evidence

def format_evidence_for_llm(evidence_dict):
    """Format aggregated evidence into a readable string for the reasoning engine."""
    formatted = []
    summaries = {s.get("paper_id"): s for s in evidence_dict.get("summaries", [])}
    relevance = evidence_dict.get("relevance", [])
    
    # Use summaries as primary source (they have the richest data)
    # Cross-reference with relevance scores
    relevance_map = {r.get("paper_id"): r for r in relevance}
    
    for s in summaries.values():
        pid = s.get("paper_id", "")
        r = relevance_map.get(pid, {})
        
        entry = (
            f"PAPER: {s.get('title', 'Unknown')}\n"
            f"- Problem: {s.get('problem', 'N/A')}\n"
            f"- Core Idea: {s.get('core_idea', 'N/A')}\n"
            f"- Method: {s.get('method', 'N/A')}\n"
            f"- Results: {s.get('results', 'N/A')}\n"
            f"- Limitations: {s.get('limitations', 'N/A')}\n"
            f"- Why Relevant: {s.get('why_relevant_to_query', 'N/A')}\n"
            f"- Relevance Score: {r.get('overall_score', 'N/A')}/10\n"
        )
        formatted.append(entry)
    
    return "\n".join(formatted)


def run_final_reasoning(query: str, evidence: dict = None) -> dict:
    if not evidence:
        evidence = load_all_evidence()
    
    formatted_evidence = format_evidence_for_llm(evidence)
    
    user_prompt = f"User Query: {query}\n\nEVIDENCE:\n{formatted_evidence}\n\nReturn ONLY JSON."
    
    try:
        reasoning_data = json_with_ollama(SYSTEM_PROMPT, user_prompt, temperature=0.2)
    except Exception as e:
        print(f"Error in reasoning engine: {e}")
        reasoning_data = {"error": str(e)}

    # Save final reasoning
    save_json(FINAL_REASONING_FILE, reasoning_data)
    print(f"✅ Final reasoning saved to {FINAL_REASONING_FILE}")
    
    return reasoning_data

if __name__ == "__main__":
    test_query = "which architecture less use gpu and data"
    run_final_reasoning(test_query)
