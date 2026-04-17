import os
import sys
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from pathlib import Path

# Import our refactored modules
from query_optimizer import run_query_optimization
from paper_fetcher import run_paper_fetching
from search_scrape import run_search_scraping
from top_k_search import run_ranking
from pdf_extract import run_pdf_extraction
from relevance_score import run_relevance_scoring
from paper_classification import run_paper_classification
from paper_summary import run_paper_summaries
from cross_paper_comparison import run_cross_paper_comparison
from reasoning_engine import run_final_reasoning
from final_report import run_final_report
from runtime_config import save_runtime_config

# ==========================================
# 1. Define State
# ==========================================
class GraphState(TypedDict):
    """
    Represents the state of our multi-agent research graph.
    """
    query: str
    optimized_query_data: Optional[dict]
    papers: Optional[List[dict]]
    web_results: Optional[List[dict]]
    ranked_pdfs: Optional[List[dict]]
    parsed_papers: Optional[List[dict]]
    relevance_scores: Optional[List[dict]]
    classifications: Optional[List[dict]]
    summaries: Optional[List[dict]]
    comparison: Optional[dict]
    reasoning: Optional[dict]
    final_report: Optional[str]

# ==========================================
# 2. Define Nodes
# ==========================================

def optimizer_node(state: GraphState):
    print("\n--- NODE: QUERY OPTIMIZATION ---")
    res = run_query_optimization(state["query"])
    return {"optimized_query_data": res}

def fetcher_node(state: GraphState):
    print("\n--- NODE: PAPER FETCHING ---")
    # Convert dict back to OptimizedQueryData if needed, 
    # but run_paper_fetching handles None by reading from file
    res = run_paper_fetching()
    return {"papers": res}

def scraper_node(state: GraphState):
    print("\n--- NODE: WEB SCRAPING ---")
    res = run_search_scraping(query=state["query"])
    return {"web_results": res}

def ranking_node(state: GraphState):
    print("\n--- NODE: PDF RANKING ---")
    res = run_ranking(query=state["query"])
    return {"ranked_pdfs": res}

def extraction_node(state: GraphState):
    print("\n--- NODE: PDF EXTRACTION ---")
    res = run_pdf_extraction()
    return {"parsed_papers": res}

def scoring_node(state: GraphState):
    print("\n--- NODE: RELEVANCE SCORING ---")
    res = run_relevance_scoring(state["query"])
    return {"relevance_scores": res}

def classification_node(state: GraphState):
    print("\n--- NODE: PAPER CLASSIFICATION ---")
    res = run_paper_classification(state["query"])
    return {"classifications": res}

def summary_node(state: GraphState):
    print("\n--- NODE: PAPER SUMMARIZATION ---")
    res = run_paper_summaries(state["query"])
    return {"summaries": res}

def comparison_node(state: GraphState):
    print("\n--- NODE: CROSS-PAPER COMPARISON ---")
    res = run_cross_paper_comparison(state["query"])
    return {"comparison": res}

def reasoning_node(state: GraphState):
    print("\n--- NODE: EXPERT FINAL REASONING ---")
    res = run_final_reasoning(state["query"])
    return {"reasoning": res}

def report_node(state: GraphState):
    print("\n--- NODE: FINAL REPORT GENERATION ---")
    res = run_final_report(state["query"])
    return {"final_report": res}

# ==========================================
# 3. Build Graph
# ==========================================

def build_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("optimize", optimizer_node)
    workflow.add_node("fetch", fetcher_node)
    workflow.add_node("scrape", scraper_node)
    workflow.add_node("rank", ranking_node)
    workflow.add_node("extract", extraction_node)
    workflow.add_node("score", scoring_node)
    workflow.add_node("classify", classification_node)
    workflow.add_node("summarize", summary_node)
    workflow.add_node("compare", comparison_node)
    workflow.add_node("reason", reasoning_node)
    workflow.add_node("report", report_node)

    # Set Entry Point
    workflow.set_entry_point("optimize")

    # Define Edges (Sequential for now)
    workflow.add_edge("optimize", "fetch")
    workflow.add_edge("fetch", "scrape")
    workflow.add_edge("scrape", "rank")
    workflow.add_edge("rank", "extract")
    workflow.add_edge("extract", "score")
    workflow.add_edge("score", "classify")
    workflow.add_edge("classify", "summarize")
    workflow.add_edge("summarize", "compare")
    workflow.add_edge("compare", "reason")
    workflow.add_edge("reason", "report")
    workflow.add_edge("report", END)

    return workflow.compile()

# ==========================================
# 4. Main execution
# ==========================================

def main():
    query = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    if not query:
        query = input("Enter research query: ").strip()
    if not query:
        print("Query is required.")
        return

    # Initialize runtime config (still useful for some internal loads)
    save_runtime_config(query=query)

    print(f"\n🚀 Starting LangGraph Orchestrator for query: {query}")
    
    app = build_graph()
    
    initial_state = {
        "query": query,
        "optimized_query_data": None,
        "papers": None,
        "web_results": None,
        "ranked_pdfs": None,
        "parsed_papers": None,
        "relevance_scores": None,
        "classifications": None,
        "summaries": None,
        "comparison": None,
        "reasoning": None,
        "final_report": None
    }

    # Run the graph
    for output in app.stream(initial_state):
        # Output is a dict mapping node name to the returned state update
        for node_name, state_update in output.items():
            print(f"Finished node: {node_name}")

    print("\n" + "=" * 90)
    print("LANGGRAPH PIPELINE COMPLETED")
    print("Final report saved in research_outputs/report/")
    print("=" * 90)

if __name__ == "__main__":
    main()
