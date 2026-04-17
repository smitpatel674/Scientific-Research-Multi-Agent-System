import os
import re
import json
from urllib.parse import urlparse

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT


# =========================================================
# CONFIG
# =========================================================
try:
    from runtime_config import load_runtime_config
    _runtime = load_runtime_config()
    QUERY = _runtime.get("query", "efficient transformer attention")
    TOP_K = int(_runtime.get("top_k", 5))
except Exception:
    QUERY = "efficient transformer attention"
    TOP_K = 5

FIRECRAWL_API_URL = os.getenv("FIRECRAWL_API_URL", "http://localhost:3002")
OUTPUT_DIR = "firecrawl_local_output"
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")


# =========================================================
# HELPERS
# =========================================================
def safe_filename(text: str) -> str:
    text = re.sub(r"[^\w\-. ]+", "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:180] if text else "untitled"


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    if FIRECRAWL_API_KEY:
        h["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
    return h


def firecrawl_healthcheck():
    print(f"Checking Firecrawl at: {FIRECRAWL_API_URL}")
    r = requests.get(FIRECRAWL_API_URL, timeout=10)
    print(f"Health check status: {r.status_code}")
    print("Health response:", r.text[:300])


def firecrawl_search(query: str, limit: int = 5) -> dict:
    url = f"{FIRECRAWL_API_URL.rstrip('/')}/v1/search"

    payload = {
        "query": query,
        "limit": limit,
        "scrapeOptions": {
            "formats": ["markdown", "html", "links"]
        }
    }

    r = requests.post(url, headers=headers(), json=payload, timeout=120)

    if not r.ok:
        print("Search request failed")
        print("Status:", r.status_code)
        print("Response:", r.text)
        r.raise_for_status()

    return r.json()


def firecrawl_scrape(url_to_scrape: str) -> dict:
    url = f"{FIRECRAWL_API_URL.rstrip('/')}/v1/scrape"

    payload = {
        "url": url_to_scrape,
        "formats": ["markdown", "html", "links"]
    }

    r = requests.post(url, headers=headers(), json=payload, timeout=120)

    if not r.ok:
        print("Scrape request failed")
        print("Status:", r.status_code)
        print("Response:", r.text)
        r.raise_for_status()

    return r.json()


def extract_result_url(item: dict) -> str:
    return item.get("url", "") or item.get("href", "") or item.get("link", "")


def extract_result_title(item: dict) -> str:
    return item.get("title", "").strip()


def extract_result_markdown(item: dict) -> str:
    return item.get("markdown", "") or item.get("content", "")


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_search_results(search_data: dict) -> list[dict]:
    data = search_data.get("data", [])

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return data.get("web", [])

    return []


# =========================================================
# PDF HELPERS
# =========================================================
def clean_text_for_pdf(text: str) -> str:
    if not text:
        return ""

    # basic markdown cleanup
    text = text.replace("\r", "\n")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "• ", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # reportlab paragraph-safe escaping
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    return text.strip()


def save_text_to_pdf(pdf_path: str, title: str, url: str, text: str):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        "MetaCustom",
        parent=styles["MetaCustom"] if "MetaCustom" in styles else styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=10,
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    story = []

    story.append(Paragraph(clean_text_for_pdf(title or "Untitled"), title_style))
    story.append(Paragraph(f"<b>Source URL:</b> {clean_text_for_pdf(url)}", meta_style))
    story.append(Spacer(1, 0.15 * inch))

    cleaned = clean_text_for_pdf(text)
    paragraphs = cleaned.split("\n\n")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = para.split("\n")
        para_html = "<br/>".join(line.strip() for line in lines if line.strip())
        if para_html:
            story.append(Paragraph(para_html, body_style))
            story.append(Spacer(1, 0.08 * inch))

    if not story:
        story.append(Paragraph("No content extracted.", body_style))

    doc.build(story)


def save_combined_pdf(pdf_path: str, records: list[dict]):
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=50,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontSize=18,
        leading=22,
        alignment=TA_LEFT,
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "SectionCustom",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceAfter=8,
    )

    meta_style = ParagraphStyle(
        "MetaCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    story = []
    story.append(Paragraph(f"Firecrawl Search Results - {clean_text_for_pdf(QUERY)}", title_style))
    story.append(Spacer(1, 0.15 * inch))

    for i, record in enumerate(records, start=1):
        story.append(Paragraph(f"{i}. {clean_text_for_pdf(record['title'] or 'Untitled')}", section_style))
        story.append(Paragraph(f"<b>URL:</b> {clean_text_for_pdf(record['url'])}", meta_style))

        cleaned = clean_text_for_pdf(record["final_markdown"])
        paragraphs = cleaned.split("\n\n")[:80]  # keep PDF manageable

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_html = "<br/>".join(line.strip() for line in para.split("\n") if line.strip())
            if para_html:
                story.append(Paragraph(para_html, body_style))

        if i != len(records):
            story.append(PageBreak())

    doc.build(story)


# =========================================================
# MAIN
# =========================================================
def run_search_scraping(query: str = None, top_k: int = None) -> list[dict]:
    global QUERY, TOP_K
    if query is not None:
        QUERY = query
    if top_k is not None:
        TOP_K = top_k

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_dir = os.path.join(OUTPUT_DIR, "pdfs")
    os.makedirs(pdf_dir, exist_ok=True)

    firecrawl_healthcheck()

    print(f"\nSearching with Firecrawl local: {QUERY}")
    search_data = firecrawl_search(QUERY, TOP_K)

    search_path = os.path.join(OUTPUT_DIR, "search_results_raw.json")
    save_json(search_path, search_data)
    print(f"Saved raw search response: {search_path}")

    results = normalize_search_results(search_data)
    if not results:
        print("No results found.")
        print("Raw response:", json.dumps(search_data, indent=2)[:2000])
        return []

    top_results_summary = []
    full_outputs = []

    for idx, item in enumerate(results[:TOP_K], start=1):
        page_url = extract_result_url(item)
        title = extract_result_title(item)
        markdown = extract_result_markdown(item)

        print(f"\n[{idx}/{min(TOP_K, len(results))}] {title or 'Untitled'}")
        print(f"URL: {page_url}")

        scrape_data = None
        if page_url and not markdown:
            try:
                print("No markdown in search result, scraping page...")
                scrape_data = firecrawl_scrape(page_url)
                scrape_payload = scrape_data.get("data", scrape_data)
                markdown = scrape_payload.get("markdown", "")
            except Exception as e:
                print(f"Scrape failed for {page_url}: {e}")

        domain = urlparse(page_url).netloc.replace("www.", "") if page_url else f"result_{idx}"

        record = {
            "rank": idx,
            "query": QUERY,
            "title": title,
            "url": page_url,
            "search_result": item,
            "scrape_result": scrape_data,
            "final_markdown": markdown,
        }

        # save json
        json_filename = f"{idx:02d}_{safe_filename(domain)}.json"
        json_filepath = os.path.join(OUTPUT_DIR, json_filename)
        save_json(json_filepath, record)
        print(f"Saved JSON: {json_filepath}")

        # save pdf
        pdf_filename = f"{idx:02d}_{safe_filename(domain)}.pdf"
        pdf_filepath = os.path.join(pdf_dir, pdf_filename)
        try:
            save_text_to_pdf(pdf_filepath, title, page_url, markdown)
            print(f"Saved PDF : {pdf_filepath}")
        except Exception as e:
            print(f"PDF save failed for {page_url}: {e}")

        top_results_summary.append({
            "rank": idx,
            "title": title,
            "url": page_url,
            "markdown_length": len(markdown or ""),
            "pdf_file": pdf_filepath,
        })

        full_outputs.append(record)

    summary_path = os.path.join(OUTPUT_DIR, "top5_summary.json")
    save_json(summary_path, top_results_summary)

    combined_path = os.path.join(OUTPUT_DIR, "all_top5_full.json")
    save_json(combined_path, full_outputs)

    combined_pdf_path = os.path.join(pdf_dir, "all_top5_combined.pdf")
    try:
        save_combined_pdf(combined_pdf_path, full_outputs)
        print(f"Saved Combined PDF: {combined_pdf_path}")
    except Exception as e:
        print(f"Combined PDF failed: {e}")

    print("\nDone.")
    print(f"Summary file : {summary_path}")
    print(f"Combined file: {combined_path}")
    print(f"PDF folder   : {pdf_dir}")
    return full_outputs


if __name__ == "__main__":
    run_search_scraping()
