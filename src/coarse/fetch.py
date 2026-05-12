"""Paper acquisition — download PDFs by DOI or title.

Resolution order: Unpaywall → Semantic Scholar → OpenAlex → direct DOI fetch.
Never raises on network failure — returns path=None with fetch_source='not_found'.
"""
from __future__ import annotations

import difflib
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import requests

from coarse.config import load_config

try:
    import trafilatura as _trafilatura
except ImportError:
    _trafilatura = None

logger = logging.getLogger(__name__)

_UNPAYWALL = "https://api.unpaywall.org/v2/{doi}?email={email}"
_S2_DOI = (
    "https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    "?fields=title,authors,year,openAccessPdf,externalIds"
)
_S2_SEARCH = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
    "?query={query}&fields=title,authors,year,openAccessPdf,externalIds&limit=1"
)
_OPENALEX = (
    "https://api.openalex.org/works"
    "?search={query}&per-page=1"
    "&select=title,publication_year,primary_location,authorships,ids"
    "&mailto={email}"
)
_TIMEOUT = 15
_DOWNLOAD_TIMEOUT = 60
_HEADERS = {"User-Agent": "coarse-ink/1.0 (mailto:coarse@example.com)"}
# Minimum normalized title similarity to accept a resolver result.
# Prevents wrong-paper matches when keyword search returns unrelated papers.
_MIN_TITLE_SIMILARITY = 0.6
# Max sibling pages fetched when crawling multi-page reports (e.g. Pew chapters).
_MAX_SIBLINGS = 8
# Patterns for locating PDF links in landing page HTML. Checked in order.
_PDF_PATTERNS = [
    # Standard academic citation meta tag (journals, Pew, NORC, institutional sites)
    re.compile(
        r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    ),
    re.compile(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']citation_pdf_url["\']',
        re.IGNORECASE,
    ),
    # Direct href to a .pdf file
    re.compile(r'href=["\']([^"\']*\.pdf(?:[?#][^"\']*)?)["\']', re.IGNORECASE),
]


def _slug(text: str) -> str:
    """Filesystem-safe slug, max 60 chars."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:60].strip("_")


def _extract_pdf_from_landing_page(page_url: str, html: str) -> str | None:
    """Return the first PDF URL found in landing page HTML, resolved to absolute."""
    for pattern in _PDF_PATTERNS:
        m = pattern.search(html)
        if m:
            return urllib.parse.urljoin(page_url, m.group(1))
    return None


def _discover_sibling_pages(page_url: str, html: str) -> list[str]:
    """Return sibling page URLs linked from HTML that share the same parent path.

    Detects multi-page reports where each chapter is its own URL (e.g. a Pew
    methodology report whose sections live at the same date-directory level).
    Only returns pages at the same path depth as page_url — not sub-pages and
    not shallower category pages. Capped at _MAX_SIBLINGS.
    """
    parsed = urllib.parse.urlparse(page_url)
    current_path = parsed.path.rstrip("/")
    if "/" not in current_path.lstrip("/"):
        return []
    parent_path = current_path.rsplit("/", 1)[0] + "/"
    target_depth = current_path.count("/")

    siblings: list[str] = []
    seen: set[str] = {current_path, current_path + "/"}

    for m in re.finditer(r'href=["\']([^"\'#?][^"\']*)["\']', html, re.IGNORECASE):
        href = m.group(1)
        if href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        full = urllib.parse.urljoin(page_url, href)
        fp = urllib.parse.urlparse(full)
        fp_path = fp.path.rstrip("/")
        if (
            fp.netloc == parsed.netloc
            and fp_path.startswith(parent_path)
            and fp_path != current_path
            and fp_path.count("/") == target_depth
            and fp_path not in seen
        ):
            siblings.append(full)
            seen.add(fp_path)
            seen.add(fp_path + "/")
            if len(siblings) >= _MAX_SIBLINGS:
                break

    return siblings


def _resolve_to_pdf(url: str) -> str | None:
    """Confirm url is a PDF or extract a direct PDF link from a landing page.

    Streams the response so large PDFs are not read into memory twice.
    Returns the URL to pass to _download, or None if no PDF is reachable.
    """
    try:
        with requests.get(
            url, timeout=_TIMEOUT, headers=_HEADERS, stream=True
        ) as resp:
            if resp.status_code != 200:
                return None
            ctype = resp.headers.get("Content-Type", "")
            if "pdf" in ctype.lower():
                return url
            if "html" not in ctype.lower():
                return None
            html = resp.text
        return _extract_pdf_from_landing_page(url, html)
    except Exception:
        logger.debug("Landing page resolution failed for %s", url, exc_info=True)
        return None


def _fetch_html_as_markdown(
    url: str,
    out_dir: Path,
    slug: str,
    crawl_siblings: bool = True,
) -> Path | None:
    """Fetch URL, extract article text via trafilatura, save as .md.

    When crawl_siblings is True (default), also discovers and fetches sibling
    pages that share the same parent path (e.g. chapters of a multi-page Pew
    report) and concatenates their text into a single .md file, separated by
    horizontal rules.

    Returns the Path of the saved .md file, or None if extraction failed
    or trafilatura is not installed.
    """
    if _trafilatura is None:
        logger.debug("trafilatura not installed; skipping HTML extraction for %s", url)
        return None
    try:
        html = _trafilatura.fetch_url(url)
        if not html:
            return None
        text = _trafilatura.extract(html, output_format="markdown", include_links=False)
        if not text:
            return None
        sections = [text]
        if crawl_siblings:
            for sibling_url in _discover_sibling_pages(url, html):
                try:
                    sibling_html = _trafilatura.fetch_url(sibling_url)
                    if sibling_html:
                        sibling_text = _trafilatura.extract(
                            sibling_html, output_format="markdown", include_links=False
                        )
                        if sibling_text:
                            sections.append(f"\n\n---\n\n{sibling_text}")
                except Exception:
                    logger.debug(
                        "Sibling page extraction failed for %s", sibling_url, exc_info=True
                    )
        dest = out_dir / f"{slug}.md"
        dest.write_text("".join(sections), encoding="utf-8")
        return dest
    except Exception:
        logger.debug("HTML extraction failed for %s", url, exc_info=True)
        return None


def _title_similar(query: str, result_title: str) -> bool:
    """Return True if result_title is similar enough to query to be trusted."""
    q = re.sub(r"[^a-z0-9 ]", "", query.lower()).strip()
    r = re.sub(r"[^a-z0-9 ]", "", result_title.lower()).strip()
    return difflib.SequenceMatcher(None, q, r).ratio() >= _MIN_TITLE_SIMILARITY


def _s2_meta(data: dict) -> dict:
    authors = [a.get("name", "") for a in (data.get("authors") or [])]
    doi = ((data.get("externalIds") or {}).get("DOI") or "").lower()
    return {
        "title": data.get("title", ""),
        "authors": authors,
        "year": data.get("year"),
        "doi": doi,
    }


def _try_unpaywall(doi: str, email: str) -> str | None:
    try:
        resp = requests.get(
            _UNPAYWALL.format(doi=doi, email=email), timeout=_TIMEOUT
        )
        if resp.status_code != 200:
            return None
        loc = resp.json().get("best_oa_location") or {}
        return loc.get("url_for_pdf") or loc.get("url")
    except Exception:
        logger.debug("Unpaywall failed for %s", doi, exc_info=True)
        return None


def _try_semantic_scholar_doi(doi: str) -> tuple[str | None, dict]:
    try:
        resp = requests.get(_S2_DOI.format(doi=doi), timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None, {}
        data = resp.json()
        url = (data.get("openAccessPdf") or {}).get("url")
        return url, data
    except Exception:
        logger.debug("S2 DOI lookup failed for %s", doi, exc_info=True)
        return None, {}


def _try_semantic_scholar_title(title: str) -> tuple[str | None, dict]:
    try:
        url = _S2_SEARCH.format(query=urllib.parse.quote(title))
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None, {}
        items = resp.json().get("data") or []
        if not items:
            return None, {}
        item = items[0]
        if not _title_similar(title, item.get("title") or ""):
            logger.debug(
                "S2 title mismatch: wanted %r, got %r", title, item.get("title")
            )
            return None, {}
        pdf = (item.get("openAccessPdf") or {}).get("url")
        return pdf, item
    except Exception:
        logger.debug("S2 title search failed for %s", title, exc_info=True)
        return None, {}


def _try_openalex_title(title: str, email: str) -> str | None:
    try:
        url = _OPENALEX.format(query=urllib.parse.quote(title), email=email)
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        results = resp.json().get("results") or []
        if not results:
            return None
        result = results[0]
        result_title = result.get("title") or result.get("display_name") or ""
        if not _title_similar(title, result_title):
            logger.debug(
                "OpenAlex title mismatch: wanted %r, got %r", title, result_title
            )
            return None
        loc = result.get("primary_location") or {}
        return loc.get("pdf_url") or loc.get("landing_page_url")
    except Exception:
        logger.debug("OpenAlex title search failed for %s", title, exc_info=True)
        return None


def _try_direct_doi(doi: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"https://doi.org/{doi}", headers=_HEADERS
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            ctype = resp.headers.get("Content-Type", "")
            final_url = resp.url
        if "pdf" in ctype.lower():
            return final_url
        return final_url  # return landing page; caller will attempt download
    except Exception:
        logger.debug("Direct DOI fetch failed for %s", doi, exc_info=True)
        return None


def _download(url: str, dest: Path) -> bool:
    try:
        resp = requests.get(
            url, timeout=_DOWNLOAD_TIMEOUT, stream=True, headers=_HEADERS
        )
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "html" in ctype.lower():
            logger.debug("Expected PDF but got HTML from %s", url)
            return False
        dest.write_bytes(resp.content)
        return True
    except Exception:
        logger.debug("Download failed from %s", url, exc_info=True)
        return False


def fetch_paper(
    doi: str | None = None,
    title: str | None = None,
    url: str | None = None,
    output_dir: str = ".",
) -> dict:
    """Download a paper PDF by DOI or title.

    Resolution order: Unpaywall → Semantic Scholar → OpenAlex → direct DOI fetch → url.
    At least one of doi, title, or url must be provided.

    Returns a dict with keys: path, title, authors, year, doi, fetch_source.
    path is None and fetch_source is 'not_found' when no PDF could be retrieved.
    """
    if not doi and not title and not url:
        raise ValueError("At least one of doi, title, or url must be provided")

    config = load_config()
    email = config.unpaywall_email or "coarse@example.com"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "title": title or "",
        "authors": [],
        "year": None,
        "doi": doi or "",
    }
    pdf_url: str | None = None
    fetch_source = "not_found"

    if doi:
        pdf_url = _try_unpaywall(doi, email)
        if pdf_url:
            fetch_source = "unpaywall"

    if not pdf_url and doi:
        pdf_url, s2_data = _try_semantic_scholar_doi(doi)
        if pdf_url:
            fetch_source = "semantic_scholar"
            meta.update(_s2_meta(s2_data))

    if not pdf_url and title:
        pdf_url, s2_data = _try_semantic_scholar_title(title)
        if pdf_url:
            fetch_source = "semantic_scholar"
            meta.update(_s2_meta(s2_data))

    if not pdf_url and title:
        pdf_url = _try_openalex_title(title, email)
        if pdf_url:
            fetch_source = "openalex"

    if not pdf_url and doi:
        pdf_url = _try_direct_doi(doi)
        if pdf_url:
            fetch_source = "direct"

    if not pdf_url and url:
        resolved = _resolve_to_pdf(url)
        if resolved:
            # URL resolved to a confirmed PDF link — download directly.
            _name = _slug(meta["title"] or title or _slug(url.rstrip("/").split("/")[-1]) or "web_clip")
            if meta.get("year"):
                _name = f"{_name}_{meta['year']}"
            _dest = out_dir / f"{_name}.pdf"
            if not _download(resolved, _dest):
                return {**meta, "path": None, "fetch_source": "not_found"}
            return {**meta, "path": str(_dest.resolve()), "fetch_source": "url"}
        # URL is HTML-only — extract article text directly to markdown.
        _name = _slug(meta["title"] or title or _slug(url.rstrip("/").split("/")[-1]) or "web_clip")
        _md = _fetch_html_as_markdown(url, out_dir, _name)
        if _md:
            return {**meta, "path": str(_md.resolve()), "fetch_source": "url"}
        return {**meta, "path": None, "fetch_source": "not_found"}

    if not pdf_url:
        return {**meta, "path": None, "fetch_source": "not_found"}

    # Confirm the URL serves a PDF — follow landing pages if needed.
    pdf_url = _resolve_to_pdf(pdf_url)
    if not pdf_url:
        return {**meta, "path": None, "fetch_source": "not_found"}

    name = _slug(meta["title"] or title or doi or "paper")
    if meta.get("year"):
        name = f"{name}_{meta['year']}"
    dest = out_dir / f"{name}.pdf"

    if not _download(pdf_url, dest):
        return {**meta, "path": None, "fetch_source": "not_found"}

    return {**meta, "path": str(dest.resolve()), "fetch_source": fetch_source}
