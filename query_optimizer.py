import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Any

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:8b"

OUTPUT_DIR = Path("query_output")
OUTPUT_DIR.mkdir(exist_ok=True)

JSON_OUTPUT_FILE = OUTPUT_DIR / "query_data.json"
TXT_OUTPUT_FILE = OUTPUT_DIR / "query_data.txt"


@dataclass
class OptimizedQuery:
    user_topic: str
    detected_domain: str
    interpreted_meaning: str
    optimized_queries: List[str]
    broad_query: str
    core_keywords: List[str]
    supporting_keywords: List[str]
    synonyms: List[str]
    negative_keywords: List[str]
    must_have_terms: List[str]
    # Virtual fields for backward compatibility
    optimized_query: str = ""
    keywords: List[str] = None

    def __post_init__(self):
        if not self.optimized_query and self.optimized_queries:
            # Join multiple queries for legacy search fields
            self.optimized_query = self.optimized_queries[0]
        
        if self.keywords is None:
            # Flatten all positive keyword categories into a single list
            self.keywords = list(dict.fromkeys(
                self.core_keywords + self.supporting_keywords + self.synonyms
            ))


def clean_text(text: Any) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def unique_list(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = clean_text(item)
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def safe_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return unique_list([clean_text(x) for x in value if clean_text(x)])


def fallback_optimizer(user_topic: str) -> OptimizedQuery:
    topic = clean_text(user_topic)

    words = [w for w in re.split(r"\W+", topic.lower()) if w]
    must_have_terms = words[: min(5, len(words))] if words else [topic.lower()]

    core_keywords = [topic]
    supporting_keywords = [f"{topic} method", f"{topic} model", f"{topic} framework"]
    synonyms = [f"{topic} approach", f"{topic} architecture"]

    negative_keywords = unique_list([
        "tutorial", "blog", "course", "job", "hiring", "news", "patent"
    ])

    optimized_queries = [
        f'"{topic}" AND (method OR architecture)',
        f'"{topic}" AND (efficient OR low-cost)',
        f'"{topic}" AND ("low-resource" OR "sample-efficient")',
        f'"{topic}" alternative approach development'
    ]
    broad_query = f'"{topic}"'

    return OptimizedQuery(
        user_topic=user_topic,
        detected_domain="General Science",
        interpreted_meaning=f"General scientific research into {topic}",
        optimized_queries=optimized_queries,
        broad_query=broad_query,
        core_keywords=core_keywords,
        supporting_keywords=supporting_keywords,
        synonyms=synonyms,
        negative_keywords=negative_keywords,
        must_have_terms=must_have_terms,
    )


def optimize_query_with_ollama(user_topic: str) -> OptimizedQuery:
    schema = {
        "type": "object",
        "properties": {
            "detected_domain": {"type": "string"},
            "interpreted_meaning": {"type": "string"},
            "optimized_queries": {"type": "array", "items": {"type": "string"}},
            "broad_query": {"type": "string"},
            "core_keywords": {"type": "array", "items": {"type": "string"}},
            "supporting_keywords": {"type": "array", "items": {"type": "string"}},
            "synonyms": {"type": "array", "items": {"type": "string"}},
            "negative_keywords": {"type": "array", "items": {"type": "string"}},
            "must_have_terms": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "detected_domain", "interpreted_meaning", "optimized_queries", 
            "broad_query", "core_keywords", "supporting_keywords", 
            "synonyms", "negative_keywords", "must_have_terms"
        ],
    }

    system_prompt = """
You are an Expert Multi-Domain Research Query Optimization Engine.

Your task is to convert ANY user topic into accurate, high-quality academic search queries for research paper retrieval across ALL domains.

You must generate output that matches how research papers are written.

-----------------------------------
CORE OBJECTIVE
Transform user queries into:
- precise academic search queries
- domain-relevant keywords
- safe filtering rules
Ensure high recall (find relevant papers) without losing precision.

-----------------------------------
STEP 1: DOMAIN DETECTION
Identify the most relevant domain:
- AI / ML: model, training, data, neural, prediction, learning, GPU
- Finance / Economics: risk, market, stock, credit, portfolio, pricing, forecasting
- Medical / Healthcare: disease, diagnosis, treatment, patient, clinical, drug
- Physics / Engineering: energy, materials, systems, mechanics, electronics
- Business / Marketing: sales, customers, growth, strategy, revenue
- Social Science / Education: behavior, learning outcomes, policy, education
If ambiguous → choose the most scientific/technical interpretation.

-----------------------------------
STEP 2: INTENT UNDERSTANDING
Extract:
- main problem (e.g., risk prediction, disease detection)
- constraints: low cost, limited data, low compute, high efficiency

-----------------------------------
STEP 3: LANGUAGE TRANSFORMATION
Convert user language → research terminology:
- "less / low / minimal" → "efficient", "low-resource", "cost-effective"
- "less data" → "data-efficient", "sample-efficient", "sparse data"
- "low compute / GPU" → "compute-efficient", "low computational cost", "reduced complexity"
- "fast" → "low latency", "computationally efficient"
IMPORTANT: Avoid engineering phrases like "GPU utilization". Use academic terms used in papers.

-----------------------------------
STEP 4: DOMAIN KNOWLEDGE MAPPING
Use domain-specific terminology:
- AI / ML: neural architecture, model, training, inference, sample-efficient learning, parameter-efficient models, memory-efficient networks, efficient attention (sparse, linear)
- Finance: financial risk, risk management, econometric modeling, statistical forecasting, portfolio optimization, credit risk assessment
- Medical: diagnosis, screening, treatment, early detection, clinical models, epidemiology
- Engineering: system design, energy efficiency, materials optimization, low-power systems
- Business: cost optimization, revenue growth strategies, customer analytics

-----------------------------------
STEP 5: DOMAIN CONSISTENCY (CRITICAL)
- Keep all outputs within detected domain. DO NOT mix unrelated domains.
- Finance → DO NOT include neural attention unless clearly ML finance.
- Medical → DO NOT include stock market.
- AI → DO NOT include legal/policy topics.

-----------------------------------
STEP 6: GENERATE OPTIMIZED QUERIES
You MUST generate EXACTLY 4 optimized_queries.
- Each query must be non-empty and use academic language.
- Include domain-specific terms and problem + constraint.

-----------------------------------
STEP 7: GENERATE KEYWORDS
- core_keywords (4–6): main domain concepts.
- supporting_keywords (6–10): methods, techniques, approaches used in domain.
- synonyms (3–6): alternative research phrases.

-----------------------------------
STEP 8: NEGATIVE KEYWORDS (SAFE FILTERING)
- ONLY remove clearly unrelated domains.
- DO NOT remove domain methods (ML, statistics, analysis) or core topic terms.
- Finance → DO NOT remove "model", "analysis", "machine learning".
- AI → DO NOT remove "learning", "data".

-----------------------------------
STEP 9: MUST-HAVE TERMS
- Select 1–2 broad, flexible words ONLY (e.g., "risk", "model", "disease").
- AVOID long phrases or rare technical words.

-----------------------------------
STEP 10: OUTPUT FORMAT
Return STRICT JSON.

-----------------------------------
STEP 11: VALIDATION
- keywords must match the detected domain.
- no cross-domain noise allowed.
- negative keywords must be safe.
- must_have_terms must have 1–2 words only.

-----------------------------------
INPUT:
User Topic: {user_topic}

Return ONLY JSON.
"""

    user_prompt = f"User Topic: {user_topic}"

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": schema,
        "stream": False,
        "options": {
            "temperature": 0.2
        }                           
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    raw_response = data.get("response", "").strip()
    if not raw_response:
        raise ValueError("Empty response from Ollama.")

    parsed = json.loads(raw_response)

    return OptimizedQuery(
        user_topic=user_topic,
        detected_domain=clean_text(parsed.get("detected_domain", "General Science")),
        interpreted_meaning=clean_text(parsed.get("interpreted_meaning", "")),
        optimized_queries=safe_list(parsed.get("optimized_queries", [])),
        broad_query=clean_text(parsed.get("broad_query", "")),
        core_keywords=safe_list(parsed.get("core_keywords", [])),
        supporting_keywords=safe_list(parsed.get("supporting_keywords", [])),
        synonyms=safe_list(parsed.get("synonyms", [])),
        negative_keywords=safe_list(parsed.get("negative_keywords", [])),
        must_have_terms=safe_list(parsed.get("must_have_terms", []))
    )


def optimize_query(user_topic: str) -> OptimizedQuery:
    try:
        return optimize_query_with_ollama(user_topic)
    except Exception as e:
        print(f"⚠ Ollama failed, using fallback optimizer. Error: {e}")
        return fallback_optimizer(user_topic)


def save_query_files(result: OptimizedQuery):
    data = asdict(result)

    with open(JSON_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    with open(TXT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(f"User Topic: {result.user_topic}\n")
        f.write(f"Detected Domain: {result.detected_domain}\n")
        f.write(f"Interpreted Meaning: {result.interpreted_meaning}\n\n")
        
        f.write("Optimized Queries:\n")
        for i, q in enumerate(result.optimized_queries, 1):
            f.write(f"{i}. {q}\n")
        
        f.write(f"\nBroad Query: {result.broad_query}\n\n")

        f.write("Core Keywords:\n")
        for k in result.core_keywords:
            f.write(f"- {k}\n")

        f.write("\nSupporting Keywords:\n")
        for k in result.supporting_keywords:
            f.write(f"- {k}\n")

        f.write("\nSynonyms:\n")
        for k in result.synonyms:
            f.write(f"- {k}\n")

        f.write("\nNegative Keywords:\n")
        for k in result.negative_keywords:
            f.write(f"- {k}\n")

        f.write("\nMust-Have Terms:\n")
        for k in result.must_have_terms:
            f.write(f"- {k}\n")

    print(f"\n✅ JSON saved: {JSON_OUTPUT_FILE}")
    print(f"✅ TXT saved : {TXT_OUTPUT_FILE}")


def run_query_optimization(query: str = None) -> dict:
    # Load topic from arg, then config, then user input
    if not query:
        try:
            from runtime_config import load_runtime_config
            runtime = load_runtime_config()
            query = str(runtime.get("query", "")).strip()
        except Exception:
            pass
    
    if not query:
        query = input("Enter research query: ").strip() or "efficient transformer attention"

    result = optimize_query(query)
    save_query_files(result)

    print("\nUser Topic:")
    print(result.user_topic)

    print("\nDetected Domain:")
    print(result.detected_domain)

    print("\nInterpreted Meaning:")
    print(result.interpreted_meaning)

    print("\nOptimized Queries:")
    for i, q in enumerate(result.optimized_queries, 1):
        print(f"{i}. {q}")

    print("\nBroad Query:")
    print(result.broad_query)

    print("\nCore Keywords:")
    for k in result.core_keywords:
        print("-", k)

    print("\nSupporting Keywords:")
    for k in result.supporting_keywords:
        print("-", k)

    print("\nSynonyms:")
    for k in result.synonyms:
        print("-", k)

    print("\nNegative Keywords:")
    for k in result.negative_keywords:
        print("-", k)

    print("\nMust-Have Terms:")
    for k in result.must_have_terms:
        print("-", k)
    
    return asdict(result)


if __name__ == "__main__":
    import sys
    test_query = sys.argv[1] if len(sys.argv) > 1 else None
    run_query_optimization(test_query)
