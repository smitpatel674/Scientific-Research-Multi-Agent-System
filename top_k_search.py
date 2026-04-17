import os
import re
import shutil
from pathlib import Path

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================
# CONFIG
# =============================
try:
    from runtime_config import load_runtime_config
    _runtime = load_runtime_config()
    QUERY = _runtime.get("query", "efficient transformer attention")
    TOP_K = int(_runtime.get("top_k", 5))
except Exception:
    QUERY = "efficient transformer attention"
    TOP_K = 5

SOURCE_FOLDERS = [
    "downloaded_papers",
    "firecrawl_local_output/pdfs",
]

OUTPUT_FOLDER = "top_k_paper"


# =============================
# HELPERS
# =============================
def _infer_source_type(folder: str) -> str:
    if folder.replace("\\", "/").rstrip("/") == "downloaded_papers":
        return "academic_pdf"
    if folder.replace("\\", "/").rstrip("/") == "firecrawl_local_output/pdfs":
        return "web_pdf"
    return "unknown"


def get_all_pdf_files(folders):
    pdf_files: list[dict] = []
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        source_type = _infer_source_type(folder)
        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(folder, file)
                pdf_files.append({"pdf_path": full_path, "source_type": source_type})

    return pdf_files


def extract_text_from_pdf(pdf_path, max_pages=20):
    text_parts = []

    try:
        doc = fitz.open(pdf_path)
        pages_to_read = min(len(doc), max_pages)

        for page_num in range(pages_to_read):
            page = doc[page_num]
            text = page.get_text("text")
            if text:
                text_parts.append(text)

        doc.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

    return "\n".join(text_parts).strip()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_likely_blog_or_explainer_pdf(pdf_path: str) -> bool:
    # Heuristic: if these dominate Top-K, we end up with hardware explainers
    # instead of research papers, which breaks "paper-based" reports.
    name = Path(pdf_path).name.lower()
    return any(
        s in name
        for s in [
            "medium.com",
            "ibm.com",
            "marktechpost",
            "all_top",
            "all_top5",
            "combined",
            "top5",
        ]
    )


def rank_pdfs_by_query(pdf_items: list[dict], query: str):
    documents = []
    valid_items: list[dict] = []

    for item in pdf_items:
        pdf_path = item["pdf_path"]
        print(f"Reading: {pdf_path}")
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print("  Skipped: no text extracted")
            continue

        # include filename also for better matching
        combined_text = f"{Path(pdf_path).stem} {text[:5000]}"
        documents.append(clean_text(combined_text))
        valid_items.append(item)

    if not documents:
        return []

    corpus = [clean_text(query)] + documents

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    query_vector = tfidf_matrix[0:1]
    doc_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(query_vector, doc_vectors)[0]

    ranked = []
    for item, score in zip(valid_items, scores):
        pdf_path = item["pdf_path"]
        source_type = item.get("source_type", "unknown")

        # Prefer academic PDFs over web-scraped explainers by applying a weight.
        # This is crucial for "accurate paper-based" reporting.
        source_weight = 1.0
        if source_type == "web_pdf":
            source_weight = 0.25

        adjusted_score = float(score) * source_weight
        ranked.append({
            "pdf_path": pdf_path,
            "score": adjusted_score,
            "raw_score": float(score),
            "source_type": source_type,
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def safe_filename(name):
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return name[:180]


def copy_top_k_pdfs(ranked_results, output_folder, top_k):
    os.makedirs(output_folder, exist_ok=True)

    # clean old PDFs in output folder
    for f in os.listdir(output_folder):
        if f.lower().endswith(".pdf"):
            os.remove(os.path.join(output_folder, f))

    selected = ranked_results[:top_k]

    for idx, item in enumerate(selected, start=1):
        src = item["pdf_path"]
        score = item["score"]
        original_name = os.path.basename(src)
        new_name = f"{idx:02d}_{safe_filename(Path(original_name).stem)}.pdf"
        dst = os.path.join(output_folder, new_name)

        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst} | score={score:.4f}")

    return selected


# =============================
# MAIN
# =============================
def run_ranking(query: str = None, top_k: int = None) -> list[dict]:
    global QUERY, TOP_K
    if query is not None:
        QUERY = query
    if top_k is not None:
        TOP_K = top_k

    print("Collecting PDFs...")
    all_pdfs = get_all_pdf_files(SOURCE_FOLDERS)

    if not all_pdfs:
        print("No PDFs found in source folders.")
        return []

    # If we have any downloaded academic PDFs, rank ONLY those.
    # Web PDFs are a fallback; otherwise they often dominate due to keyword density
    # (e.g., hardware explainers) and degrade report accuracy.
    academic = [p for p in all_pdfs if p.get("source_type") == "academic_pdf"]
    web = [p for p in all_pdfs if p.get("source_type") == "web_pdf"]

    if academic:
        pdf_pool = academic
        print(f"Total PDFs found: {len(all_pdfs)} (academic={len(academic)}, web={len(web)})")
        print("Using academic PDFs only for ranking (web PDFs are fallback).")
    else:
        # If no academic PDFs exist, keep web PDFs, but drop likely blog/explainer
        # combined files which tend to be low-quality evidence.
        filtered_web = [p for p in web if not _is_likely_blog_or_explainer_pdf(p["pdf_path"])]
        pdf_pool = filtered_web if filtered_web else web
        print(f"Total PDFs found: {len(all_pdfs)} (academic=0, web={len(web)})")
        if filtered_web and len(filtered_web) != len(web):
            print(f"Filtered web PDFs: {len(web)} -> {len(filtered_web)}")

    print(f"Total PDFs ranked: {len(pdf_pool)}")
    print(f"Ranking PDFs for query: {QUERY}\n")

    ranked_results = rank_pdfs_by_query(pdf_pool, QUERY)

    if not ranked_results:
        print("No ranked results found.")
        return []

    print("\nTop ranked PDFs:")
    for i, item in enumerate(ranked_results[:TOP_K], start=1):
        extra = ""
        if "source_type" in item:
            extra = f" | source={item.get('source_type')}"
        print(f"{i}. {item['pdf_path']} | score={item['score']:.4f}{extra}")

    print(f"\nCopying top {TOP_K} PDFs to '{OUTPUT_FOLDER}'...")
    selected = copy_top_k_pdfs(ranked_results, OUTPUT_FOLDER, TOP_K)

    print("\nDone.")
    print(f"Top {len(selected)} PDFs saved in: {OUTPUT_FOLDER}")
    return selected


def main():
    run_ranking()


if __name__ == "__main__":
    main()
