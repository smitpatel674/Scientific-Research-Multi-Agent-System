"""
Research Paper Fetcher & Downloader
Flow:
1. query_optimizer.py saves query_output/query_data.json
2. this file reads that JSON
3. fetches only related papers
"""

import os
import re
import time
import json
import asyncio
import requests
import urllib.parse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from config import MAX_PAPERS_PER_SOURCE, USE_LLM_VERIFIER, DOWNLOAD_TARGET_COUNT
from ollama_utils import json_with_ollama


# ── Auto-install playwright if missing ────────────────────────────────────────
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    print("⚙ Installing playwright…")
    os.system("pip install playwright --quiet")
    os.system("python -m playwright install chromium --quiet")
    try:
        from playwright.async_api import async_playwright
        PLAYWRIGHT_AVAILABLE = True
    except ImportError:
        PLAYWRIGHT_AVAILABLE = False
        print("⚠ Playwright unavailable – scraping fallback disabled.")


# ── Config ─────────────────────────────────────────────────────────────────────
DOWNLOAD_DIR = Path("downloaded_papers")
DOWNLOAD_DIR.mkdir(exist_ok=True)

QUERY_DIR = Path("query_output")
QUERY_JSON_FILE = QUERY_DIR / "query_data.json"

YOUR_EMAIL = "smitpatidar6704@gmail.com"

# Semantic Scholar API key from environment
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}

PDF_LINK_SELECTORS = [
    'a[href$=".pdf"]',
    'a[href*="/pdf/"]',
    'a[href*="pdf"]',
    'a[title*="PDF" i]',
    'a[aria-label*="PDF" i]',
    'a:has-text("PDF")',
    'a:has-text("Download PDF")',
    'a:has-text("Full Text PDF")',
    'a:has-text("View PDF")',
    'a:has-text("Open PDF")',
    'a:has-text("Open PDF in Browser")',
    'button:has-text("PDF")',
    'button:has-text("Open PDF")',
    'button:has-text("Open PDF in Browser")',
    'button:has-text("Download This Paper")',
    '[data-track-action*="pdf" i]',
    '.pdf-download a',
    '.article-tools a[href*="pdf"]',
    '#full-text-links-list a',
    '.c-pdf-download__link',
    '.article__tool--pdf a',
    'a.show-pdf',
    'a[data-article-url*="pdf"]',
]


@dataclass
class OptimizedQueryData:
    user_topic: str
    optimized_queries: list[str]
    broad_query: str
    keywords: list[str]
    negative_keywords: list[str]
    must_have_terms: list[str]
    # For backward compatibility
    optimized_query: str = ""


@dataclass
class Paper:
    title: str
    authors: list[str]
    abstract: str
    year: Optional[int]
    doi: Optional[str]
    pdf_url: Optional[str]
    source: str
    paper_id: str
    extra: dict = field(default_factory=dict)

    def __str__(self):
        authors_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            authors_str += " et al."
        return (
            f"[{self.source}] {self.title}\n"
            f"  Authors : {authors_str}\n"
            f"  Year    : {self.year}\n"
            f"  DOI     : {self.doi}\n"
            f"  PDF     : {self.pdf_url}\n"
        )


def load_query_data() -> OptimizedQueryData:
    if not QUERY_JSON_FILE.exists():
        raise FileNotFoundError(
            f"{QUERY_JSON_FILE} not found.\n"
            f"Run query_optimizer.py first."
        )

    with open(QUERY_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load optimized_queries, fallback to singular optimized_query if missing
    optimized_queries = data.get("optimized_queries", [])
    if not optimized_queries and data.get("optimized_query"):
        optimized_queries = [data["optimized_query"]]

    return OptimizedQueryData(
        user_topic=data.get("user_topic", ""),
        optimized_queries=optimized_queries,
        optimized_query=data.get("optimized_query", ""),
        broad_query=data.get("broad_query", ""),
        keywords=data.get("keywords", []),
        negative_keywords=data.get("negative_keywords", []),
        must_have_terms=data.get("must_have_terms", []),
    )


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_related_paper(
    paper: Paper,
    positive_keywords: list[str],
    negative_keywords: list[str],
    must_have_terms: list[str],
) -> tuple[bool, str]:
    text = normalize_text(f"{paper.title} {paper.abstract}")

    def matches_term(term: str, target_text: str) -> bool:
        term = normalize_text(term)
        # bag-of-words for multi-word terms
        words = term.split()
        if not words:
            return False
        return all(w in target_text for w in words)

    positive_hits = [kw for kw in positive_keywords if matches_term(kw, text)]
    negative_hits = [kw for kw in negative_keywords if matches_term(kw, text)]
    must_have_hits = [kw for kw in must_have_terms if matches_term(kw, text)]

    if len(negative_hits) >= 2:
        return False, f"Negative keywords hit: {', '.join(negative_hits[:2])}"

    if must_have_terms and not must_have_hits:
        return False, f"Missing Must-Have: {', '.join(must_have_terms[:1])}"

    if len(positive_hits) >= 2:
        return True, ""

    if len(positive_hits) >= 1 and len(must_have_hits) >= 1:
        return True, ""

    return False, f"Too few keyword hits ({len(positive_hits)})"


def llm_verify_relevance(papers: list[Paper], query_data: OptimizedQueryData) -> list[Paper]:
    """Uses Ollama to accurately verify the relevance of multiple papers at once."""
    if not USE_LLM_VERIFIER or not papers:
        return papers

    print(f"🧠 Verifying {len(papers)} papers with Ollama...")

    # We'll batch them to avoid hitting context limits, though 10-15 titles/abstracts should fit.
    papers_context = []
    for i, p in enumerate(papers):
        papers_context.append({
            "id": i,
            "title": p.title,
            "abstract": (p.abstract[:300] + "...") if len(p.abstract) > 300 else p.abstract
        })

    system_prompt = """
You are a Research Relevance Expert. Your job is to strictly filter a list of academic papers based on a user's research topic.
Return ONLY a JSON list of indices ([0, 2, ...]) for papers that are HIGHLY relevant.
If a paper is only tangentially related or missing key terms from the 'Must-Have' list, exclude it.
    """

    user_prompt = f"""
Research Topic: {query_data.user_topic}
Optimized Query: {query_data.optimized_query}
Must-Have Terms: {query_data.must_have_terms}

Papers to Evaluate:
{json.dumps(papers_context, indent=2)}

Return ONLY a JSON list of indices of the most relevant papers.
    """

    try:
        relevant_indices = json_with_ollama(system_prompt, user_prompt)
        if not isinstance(relevant_indices, list):
            print("⚠ Ollama returned invalid format. Falling back to keyword filter.")
            return papers

        verified_papers = [papers[i] for i in relevant_indices if 0 <= i < len(papers)]
        print(f"✅ LLM verified {len(verified_papers)}/{len(papers)} papers as highly relevant.")
        return verified_papers
    except Exception as e:
        print(f"⚠ LLM Verification failed: {e}. Falling back to default list.")
        return papers


def deduplicate_papers(papers: list[Paper]) -> list[Paper]:
    seen = set()
    unique = []

    for p in papers:
        title_key = re.sub(r"\s+", " ", (p.title or "").strip().lower())
        doi_key = (p.doi or "").strip().lower()

        key = ("doi", doi_key) if doi_key else ("title", title_key)

        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


def fetch_arxiv(query: str, max_results: int = 5) -> list[Paper]:
    url = "https://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance"
    }
    
    # ArXiv is strict: 1 request per 3 seconds. Using 4s for stability.
    time.sleep(4)
    print(f"  🕒 Polite wait for arXiv...")
    
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if resp.status_code == 429:
                # arXiv can rate-limit aggressively; long sleeps make the full pipeline feel hung.
                # We skip arXiv for this run and rely on the other academic sources.
                print("  ⚠ arXiv rate limited (429). Skipping arXiv source for this run.")
                return []
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)
                print(f"  ⚠ arXiv error. Retrying in {wait}s…")
                time.sleep(wait)
            else:
                print(f"  ✗ arXiv API failed after 3 attempts: {e}")
                return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom"
    }
    try:
        root = ET.fromstring(resp.text)
    except Exception as e:
        print(f"  ✗ Error parsing arXiv XML: {e}")
        return []

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", namespaces=ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
        published = entry.findtext("atom:published", namespaces=ns, default="")
        year = int(published[:4]) if published else None
        authors = [a.findtext("atom:name", namespaces=ns) for a in entry.findall("atom:author", ns)]
        arxiv_id = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1]
        doi_elem = entry.find("arxiv:doi", ns)
        doi = doi_elem.text if doi_elem is not None else None

        papers.append(Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            source="arXiv",
            paper_id=arxiv_id,
        ))
    return papers


def requests_get_semantic_scholar(url: str, params: dict = None, max_retries: int = 5) -> Optional[requests.Response]:
    """Helper to perform Semantic Scholar requests with polite waits and backoff."""
    ss_headers = HEADERS.copy()
    if SEMANTIC_SCHOLAR_API_KEY:
        ss_headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    for attempt in range(max_retries):
        # Polite wait before EVERY request to be safe
        # (Using a slightly higher delay to be safe for free tier)
        wait_time = 1.5 if not SEMANTIC_SCHOLAR_API_KEY else 0.2
        time.sleep(wait_time)

        try:
            resp = requests.get(url, params=params, headers=ss_headers, timeout=30)
            if resp.status_code == 429:
                # Exponential backoff with jitter
                backoff = (2 ** attempt) + (random.uniform(0, 1))
                print(f"  ⚠ Semantic Scholar rate limited. Retrying in {backoff:.1f}s… (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff)
                continue
            
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  ✗ Semantic Scholar error: {e}")
                return None
            time.sleep(1)
    return None


def fetch_semantic_scholar(query: str, max_results: int = 5) -> list[Paper]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,year,externalIds,openAccessPdf"
    }

    print(f"  🕒 Polite wait for Semantic Scholar...")
    resp = requests_get_semantic_scholar(url, params=params)
    if not resp:
        print("  ✗ Semantic Scholar skipping source due to errors.")
        return []

    papers = []
    for item in resp.json().get("data", []):
        ext = item.get("externalIds") or {}
        doi = ext.get("DOI")
        arxiv_id = ext.get("ArXiv")
        oa = item.get("openAccessPdf") or {}
        pdf_url = oa.get("url") or (f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None)
        authors = [a.get("name", "") for a in item.get("authors", [])]

        papers.append(Paper(
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstract") or "",
            year=item.get("year"),
            doi=doi,
            pdf_url=pdf_url,
            source="Semantic Scholar",
            paper_id=item.get("paperId", "")
        ))
    return papers


def fetch_crossref(query: str, max_results: int = 5) -> list[Paper]:
    resp = requests.get(
        "https://api.crossref.org/works",
        params={
            "query": query,
            "rows": max_results,
            "select": "title,author,abstract,published,DOI,link",
            "mailto": YOUR_EMAIL
        },
        headers=HEADERS,
        timeout=30
    )
    resp.raise_for_status()

    papers = []
    for item in resp.json().get("message", {}).get("items", []):
        title = (item.get("title") or ["No Title"])[0]
        authors = [f"{a.get('given','')} {a.get('family','')}".strip() for a in item.get("author", [])]
        date_parts = item.get("published", {}).get("date-parts", [[None]])[0]
        year = date_parts[0] if date_parts else None
        doi = item.get("DOI")
        pdf_url = next((l["URL"] for l in item.get("link", []) if l.get("content-type") == "application/pdf"), None)

        papers.append(Paper(
            title=title,
            authors=authors,
            abstract=item.get("abstract", ""),
            year=year,
            doi=doi,
            pdf_url=pdf_url,
            source="Crossref",
            paper_id=doi or title[:40]
        ))
    return papers


def fetch_pubmed(query: str, max_results: int = 5) -> list[Paper]:
    api_key = os.getenv("NCBI_API_KEY", "")
    base = {"api_key": api_key} if api_key else {}

    search = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
        params={**base, "db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
        headers=HEADERS,
        timeout=30
    )
    search.raise_for_status()
    pmids = search.json()["esearchresult"].get("idlist", [])
    if not pmids:
        return []

    time.sleep(0.4)

    fetch = requests.get(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        params={**base, "db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
        headers=HEADERS,
        timeout=30
    )
    fetch.raise_for_status()
    root = ET.fromstring(fetch.text)

    papers = []
    for article in root.findall(".//PubmedArticle"):
        medline = article.find(".//MedlineCitation")
        art = medline.find("Article") if medline is not None else None
        if art is None:
            continue

        title = art.findtext("ArticleTitle") or ""
        authors = []
        for a in art.findall(".//Author"):
            name = f"{a.findtext('ForeName') or ''} {a.findtext('LastName') or ''}".strip()
            if name:
                authors.append(name)

        abstract = " ".join(t.text or "" for t in art.findall(".//AbstractText"))
        pub_date = art.find(".//PubDate")
        year_text = pub_date.findtext("Year") if pub_date is not None else None
        year = int(year_text) if year_text and year_text.isdigit() else None
        pmid = medline.findtext("PMID")

        doi = None
        pmc_id = None
        for eid in article.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text
            if eid.get("IdType") == "pmc":
                pmc_id = eid.text

        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else None

        papers.append(Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            doi=doi,
            pdf_url=pdf_url,
            source="PubMed",
            paper_id=pmid or "",
            extra={"pmc_id": pmc_id}
        ))
    return papers


def fetch_europe_pmc(query: str, max_results: int = 5) -> list[Paper]:
    resp = requests.get(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        params={"query": query, "pageSize": max_results, "format": "json", "resultType": "core"},
        headers=HEADERS,
        timeout=30
    )
    resp.raise_for_status()

    papers = []
    for item in resp.json().get("resultList", {}).get("result", []):
        pmcid = item.get("pmcid")
        doi = item.get("doi")
        pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render" if pmcid else None
        authors = [a.strip() for a in (item.get("authorString") or "").split(",") if a.strip()]

        papers.append(Paper(
            title=item.get("title", ""),
            authors=authors,
            abstract=item.get("abstractText") or "",
            year=item.get("pubYear"),
            doi=doi,
            pdf_url=pdf_url,
            source="Europe PMC",
            paper_id=item.get("id", ""),
            extra={"pmcid": pmcid, "pmid": item.get("pmid")}
        ))
    return papers


def sanitize_filename(name: str, max_len: int = 120) -> str:
    safe = re.sub(r"[^\w\s\-]", "_", name)
    safe = re.sub(r"\s+", " ", safe).strip()
    return safe[:max_len]


def is_real_pdf(path: Path, min_kb: int = 5) -> bool:
    if not path.exists():
        return False
    if path.stat().st_size < min_kb * 1024:
        return False
    with open(path, "rb") as f:
        return f.read(4) == b"%PDF"


def try_download_url(url: str, dest: Path, label: str, extra_headers: dict = None) -> bool:
    hdrs = {**HEADERS, **(extra_headers or {})}
    try:
        print(f"  → [{label}] {url[:90]}…")
        resp = requests.get(url, headers=hdrs, timeout=60, stream=True, allow_redirects=True)
        first = next(resp.iter_content(512), b"")
        ct = resp.headers.get("Content-Type", "")

        if resp.status_code == 200 and ("pdf" in ct.lower() or first[:4] == b"%PDF"):
            with open(dest, "wb") as f:
                f.write(first)
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

            if is_real_pdf(dest):
                size_kb = dest.stat().st_size // 1024
                print(f"  ✓ Saved ({size_kb} KB): {dest.name}")
                return True

            if dest.exists():
                dest.unlink()

        else:
            print(f"  ✗ Not a PDF ({resp.status_code}, {ct[:60]})")
    except requests.RequestException as e:
        print(f"  ✗ Request error: {e}")

    return False


def try_unpaywall(doi: str) -> Optional[str]:
    try:
        data = requests.get(
            f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='')}?email={YOUR_EMAIL}",
            timeout=15
        ).json()
        return (data.get("best_oa_location") or {}).get("url_for_pdf")
    except Exception:
        return None


def try_semantic_scholar_doi(doi: str) -> Optional[str]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi, safe='')}"
    params = {"fields": "openAccessPdf,externalIds"}
    
    resp = requests_get_semantic_scholar(url, params=params, max_retries=3)
    if resp and resp.status_code == 200:
        try:
            data = resp.json()
            oa = data.get("openAccessPdf") or {}
            if oa.get("url"):
                return oa["url"]
            arxiv_id = (data.get("externalIds") or {}).get("ArXiv")
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        except Exception:
            pass
    return None


def try_open_access_button(doi: str) -> Optional[str]:
    try:
        resp = requests.get(
            f"https://api.openaccessbutton.org/find?id={urllib.parse.quote(doi, safe='')}",
            headers=HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            urls = resp.json().get("url") or []
            if isinstance(urls, str):
                urls = [urls]
            for u in urls:
                if u and u.lower().endswith(".pdf"):
                    return u
            return urls[0] if urls else None
    except Exception:
        return None


def try_pmc_via_doi(doi: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/",
            params={"ids": doi, "format": "json"},
            headers=HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            for rec in resp.json().get("records", []):
                pmcid = rec.get("pmcid")
                if pmcid:
                    return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    except Exception:
        pass
    return None


def try_arxiv_title_search(title: str) -> Optional[str]:
    try:
        resp = requests.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{title}"', "max_results": 1},
            headers=HEADERS,
            timeout=15
        )
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        entry = root.find("atom:entry", ns)
        if entry is not None:
            arxiv_id = (entry.findtext("atom:id", namespaces=ns) or "").split("/abs/")[-1]
            if arxiv_id:
                return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    except Exception:
        pass
    return None


def resolve_api_candidates(paper: Paper) -> list[tuple[str, str]]:
    candidates = []
    if paper.pdf_url:
        candidates.append(("direct", paper.pdf_url))

    if paper.doi:
        for label, func in [
            ("Unpaywall", try_unpaywall),
            ("Semantic Scholar", try_semantic_scholar_doi),
            ("OA Button", try_open_access_button),
            ("PMC", try_pmc_via_doi),
        ]:
            u = func(paper.doi)
            if u:
                candidates.append((label, u))

    if paper.title:
        u = try_arxiv_title_search(paper.title)
        if u:
            candidates.append(("arXiv title", u))

    dedup = []
    seen = set()
    for label, url in candidates:
        if url not in seen:
            seen.add(url)
            dedup.append((label, url))

    return dedup


async def playwright_scrape_pdf(paper: Paper, dest: Path) -> bool:
    if not PLAYWRIGHT_AVAILABLE:
        return False

    landing_url = None
    if paper.doi:
        landing_url = f"https://doi.org/{urllib.parse.quote(paper.doi, safe='/')}"
    elif paper.source == "arXiv" and paper.paper_id:
        landing_url = f"https://arxiv.org/abs/{paper.paper_id}"
    elif paper.source == "PubMed" and paper.paper_id:
        landing_url = f"https://pubmed.ncbi.nlm.nih.gov/{paper.paper_id}/"
    elif paper.source == "Europe PMC":
        pmcid = paper.extra.get("pmcid")
        pmid = paper.extra.get("pmid")
        if pmcid:
            landing_url = f"https://europepmc.org/article/PMC/{pmcid}"
        elif pmid:
            landing_url = f"https://europepmc.org/article/MED/{pmid}"

    if not landing_url:
        return False

    print(f"  🌐 Playwright → {landing_url[:90]}")

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=HEADERS["User-Agent"],
                accept_downloads=True,
                viewport={"width": 1280, "height": 800},
            )
            page = await context.new_page()

            await page.goto(landing_url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)

            pdf_href = None
            for selector in PDF_LINK_SELECTORS:
                try:
                    el = page.locator(selector).first
                    if await el.count() > 0:
                        href = await el.get_attribute("href")
                        if href:
                            if href.startswith("//"):
                                href = "https:" + href
                            elif href.startswith("/"):
                                from urllib.parse import urlparse
                                base = urlparse(page.url)
                                href = f"{base.scheme}://{base.netloc}{href}"
                            pdf_href = href
                            print(f"  🔗 Found PDF via selector: {selector}")
                            break
                except Exception:
                    continue

            if not pdf_href:
                try:
                    html = await page.content()
                    match = re.search(r'https?://[^\s"\']+\.pdf', html, re.IGNORECASE)
                    if match:
                        pdf_href = match.group(0)
                        print("  🔎 Found PDF URL in HTML")
                except Exception:
                    pass

            if pdf_href:
                try:
                    r = requests.get(
                        pdf_href,
                        headers={**HEADERS, "Referer": page.url},
                        timeout=60,
                        stream=True,
                        allow_redirects=True
                    )
                    first = next(r.iter_content(512), b"")
                    ct = r.headers.get("Content-Type", "")
                    if r.status_code == 200 and ("pdf" in ct.lower() or first[:4] == b"%PDF"):
                        with open(dest, "wb") as f:
                            f.write(first)
                            for chunk in r.iter_content(8192):
                                f.write(chunk)
                        await browser.close()
                        return is_real_pdf(dest)
                except Exception:
                    pass

            await browser.close()
    except Exception:
        return False

    return False


async def download_paper_async(paper: Paper) -> bool:
    filename = sanitize_filename(paper.title) + ".pdf"
    dest = DOWNLOAD_DIR / filename

    if is_real_pdf(dest):
        print(f"  ✓ Already exists: {dest.name}")
        return True

    candidates = resolve_api_candidates(paper)
    for label, url in candidates:
        if try_download_url(url, dest, label):
            return True
        time.sleep(0.3)

    print("  ⚙ API sources exhausted — trying Playwright scraper…")
    success = await playwright_scrape_pdf(paper, dest)
    if success and is_real_pdf(dest):
        return True

    print(f"  ✗ All methods failed: {paper.title[:80]}")
    return False


def download_paper(paper: Paper) -> bool:
    return asyncio.run(download_paper_async(paper))


def save_metadata(papers: list[Paper], filename: str = "related_papers_metadata.json"):
    path = DOWNLOAD_DIR / filename
    records = [{
        "title": p.title,
        "authors": p.authors,
        "abstract": p.abstract[:1000],
        "year": p.year,
        "doi": p.doi,
        "pdf_url": p.pdf_url,
        "source": p.source,
        "paper_id": p.paper_id,
    } for p in papers]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Metadata saved → {path}")


def run_paper_fetching(optimized: OptimizedQueryData = None, max_results: int = None, choice: str = None) -> list:
    if optimized is None:
        try:
            optimized = load_query_data()
        except FileNotFoundError as e:
            print(f"\n❌ {e}\n")
            return []

    if max_results is None:
        max_results = MAX_PAPERS_PER_SOURCE
        print(f"Papers per source starting limit: {max_results}")

    print("\n" + "=" * 80)
    print("USER TOPIC      :", optimized.user_topic)
    print("OPTIMIZED QUERY :", optimized.optimized_query)
    print("BROAD QUERY     :", optimized.broad_query)
    print("KEYWORDS        :", ", ".join(optimized.keywords[:12]))
    print("NEGATIVE        :", ", ".join(optimized.negative_keywords[:12]))
    print("MUST-HAVE       :", ", ".join(optimized.must_have_terms))
    print("=" * 80)

    all_papers: list[Paper] = []

    sources = [
        ("arXiv", fetch_arxiv),
        ("Semantic Scholar", fetch_semantic_scholar),
        ("Crossref", fetch_crossref),
        ("PubMed", fetch_pubmed),
        ("Europe PMC", fetch_europe_pmc),
    ]

    for name, fetcher in sources:
        print(f"\n▶ {name}...")
        try:
            # Collect papers from ALL optimized queries + the broad query
            all_source_candidates = []
            
            # 1. Broad query search
            all_source_candidates.extend(fetcher(optimized.broad_query, max_results))
            
            # 2. Optimized queries search
            for i, q in enumerate(optimized.optimized_queries, 1):
                # Small pause between sub-queries to avoid rapid-fire API hits
                time.sleep(0.5)
                all_source_candidates.extend(fetcher(q, max_results))

            source_papers = deduplicate_papers(all_source_candidates)
            print(f"  Found {len(source_papers)} candidate paper(s)")

            related = []
            for p in source_papers:
                is_rel, reason = is_related_paper(
                    p,
                    optimized.keywords,
                    optimized.negative_keywords,
                    optimized.must_have_terms,
                )
                if is_rel:
                    related.append(p)
                    print(f"  ✅ RELATED   : {p.title[:90]}")
                else:
                    print(f"  ❌ UNRELATED : {p.title[:90]} ({reason})")

            all_papers.extend(related)
            time.sleep(1.2)

        except Exception as e:
            print(f"  ✗ Error: {e}")

    all_papers = deduplicate_papers(all_papers)

    # --- ADVANCED OPTIMIZATION: LLM VERIFICATION ---
    if USE_LLM_VERIFIER and all_papers:
        all_papers = llm_verify_relevance(all_papers, optimized)

    print("\n" + "-" * 80)
    print(f"Total related papers found: {len(all_papers)}")

    # Limit to download target if we have too many
    if len(all_papers) > DOWNLOAD_TARGET_COUNT:
        print(f"💡 Capping downloads at {DOWNLOAD_TARGET_COUNT} most relevant papers.")
        all_papers = all_papers[:DOWNLOAD_TARGET_COUNT]

    if not all_papers:
        print("No related papers found.")
        return []

    if choice is None:
        try:
            from runtime_config import load_runtime_config
            runtime = load_runtime_config()
            choice = str(runtime.get("download_choice", "1")).strip()
            print(f"Choice from runtime config: {choice}")
        except Exception:
            print("\nOptions:")
            print("  1 – Download PDFs")
            print("  2 – Save metadata only")
            choice = input("Your choice [1/2]: ").strip()

    if choice == "1":
        downloaded = 0
        failed = 0

        print(f"\n📥 Downloading related papers → ./{DOWNLOAD_DIR}/\n")

        for i, paper in enumerate(all_papers, 1):
            print(f"\n[{i}/{len(all_papers)}] {paper.title[:100]}")
            print(f"  Source: {paper.source} | DOI: {paper.doi or 'N/A'}")

            ok = download_paper(paper)
            if ok:
                downloaded += 1
            else:
                failed += 1

            time.sleep(1)

        print("\n" + "-" * 80)
        print(f"✅ Downloaded : {downloaded}")
        print(f"❌ Failed     : {failed}")

    save_metadata(all_papers)
    print("\nDone! 🎉")
    return [asdict(p) for p in all_papers]


def main():
    run_paper_fetching()


if __name__ == "__main__":
    main()
