import fitz
from pathlib import Path
from config import PDF_INPUT_DIR, PARSED_DIR
from file_utils import save_json


def extract_pdf_data(pdf_path):
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    pages = []
    full_text_parts = []

    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_number": i + 1,
            "text": text.strip()
        })
        full_text_parts.append(text.strip())

    full_text = "\n\n".join(full_text_parts).strip()
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    title = lines[0] if lines else pdf_path.stem

    return {
        "paper_id": pdf_path.stem,
        "file_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "title": title,
        "total_pages": len(doc),
        "pages": pages,
        "full_text": full_text
    }


def run_pdf_extraction():
    pdf_files = list(PDF_INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in top_k_paper/")
        return []

    # Ensure downstream steps only see the current Top-K set.
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    for old in PARSED_DIR.glob("*.json"):
        try:
            old.unlink()
        except Exception:
            pass

    parsed_papers = []

    for pdf_file in pdf_files:
        print(f"[PDF EXTRACT] {pdf_file.name}")
        data = extract_pdf_data(pdf_file)
        save_json(PARSED_DIR / f"{data['paper_id']}.json", data)
        parsed_papers.append(data)

    print(f"\nDone. Parsed {len(parsed_papers)} papers.")
    return parsed_papers


if __name__ == "__main__":
    run_pdf_extraction()
