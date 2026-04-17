from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

PDF_INPUT_DIR = BASE_DIR / "top_k_paper"

OUTPUT_DIR = BASE_DIR / "research_outputs"
PARSED_DIR = OUTPUT_DIR / "parsed_json"
RELEVANCE_DIR = OUTPUT_DIR / "relevance"
CLASSIFICATION_DIR = OUTPUT_DIR / "classification"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
REPORT_DIR = OUTPUT_DIR / "report"

MAX_PAPERS_PER_SOURCE = 12
USE_LLM_VERIFIER = True
DOWNLOAD_TARGET_COUNT = 8

OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434"

for folder in [
    OUTPUT_DIR,
    PARSED_DIR,
    RELEVANCE_DIR,
    CLASSIFICATION_DIR,
    SUMMARY_DIR,
    COMPARISON_DIR,
    REPORT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
