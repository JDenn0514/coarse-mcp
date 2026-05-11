"""Paper acquisition — download PDFs by DOI or title.

Resolution order: Unpaywall → Semantic Scholar → OpenAlex → direct DOI fetch.
Never raises on network failure — returns path=None with fetch_source='not_found'.
"""
from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import requests

from coarse.config import load_config

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


def _slug(text: str) -> str:
    """Filesystem-safe slug, max 60 chars."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:60].strip("_")


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
        loc = results[0].get("primary_location") or {}
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
        dest.write_bytes(resp.content)
        return True
    except Exception:
        logger.debug("Download failed from %s", url, exc_info=True)
        return False


def fetch_paper(
    doi: str | None = None,
    title: str | None = None,
    output_dir: str = ".",
) -> dict:
    """Download a paper PDF by DOI or title.

    Resolution order: Unpaywall → Semantic Scholar (DOI) →
    Semantic Scholar (title) → OpenAlex (title) → direct DOI fetch.

    Returns a dict with keys: path, title, authors, year, doi, fetch_source.
    path is None and fetch_source is 'not_found' when no PDF could be retrieved.
    """
    if not doi and not title:
        raise ValueError("At least one of doi or title must be provided")

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

    if not pdf_url:
        return {**meta, "path": None, "fetch_source": "not_found"}

    name = _slug(meta["title"] or title or doi or "paper")
    if meta.get("year"):
        name = f"{name}_{meta['year']}"
    dest = out_dir / f"{name}.pdf"

    if not _download(pdf_url, dest):
        return {**meta, "path": None, "fetch_source": "not_found"}

    return {**meta, "path": str(dest.resolve()), "fetch_source": fetch_source}
