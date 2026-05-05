"""CrossRef DOI enrichment for bibliography formatting.

Extracts DOIs from citation URLs and fetches structured metadata from the
CrossRef API to produce formatted bibliography entries.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_DOI_ORG_RE = re.compile(r"(?:dx\.)?doi\.org/(10\.\d{4,}/\S+)")
_BARE_DOI_RE = re.compile(r"(10\.\d{4,}/\S+)")
_CROSSREF_API = "https://api.crossref.org/works/{doi}"
_TIMEOUT = 5


def extract_doi(url: str) -> str | None:
    """Return bare DOI string from a URL, or None if not found."""
    m = _DOI_ORG_RE.search(url)
    if m:
        return m.group(1).rstrip("/")
    m = _BARE_DOI_RE.search(url)
    if m:
        return m.group(1).rstrip("/")
    return None


def enrich_citations(urls: list[str]) -> list[str]:
    """Return formatted bibliography strings; falls back to raw URL on any failure."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_enrich_one, url) for url in urls]
    results = []
    for i, future in enumerate(futures):
        try:
            results.append(future.result())
        except Exception:
            results.append(urls[i])
    return results


def _enrich_one(url: str) -> str:
    doi = extract_doi(url)
    if doi is None:
        return url
    enriched = _fetch_crossref(doi)
    return enriched if enriched is not None else url


def _fetch_crossref(doi: str) -> str | None:
    api_url = _CROSSREF_API.format(doi=doi)
    try:
        with urllib.request.urlopen(api_url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except Exception:
        logger.debug("CrossRef lookup failed for doi %s", doi, exc_info=True)
        return None
    return _format_reference(data.get("message") or {}, doi)


def _format_reference(message: dict, doi: str) -> str:
    authors = message.get("author") or []
    year: int | None = None
    date_parts = (message.get("published") or {}).get("date-parts") or [[]]
    if date_parts and date_parts[0]:
        year = date_parts[0][0]

    titles = message.get("title") or []
    title = titles[0] if titles else None

    journals = message.get("container-title") or []
    journal = journals[0] if journals else None

    doi_str = message.get("DOI") or doi

    author_str = ""
    if authors:
        first = authors[0]
        family = (first.get("family") or "").strip()
        given = (first.get("given") or "").strip()
        if family and given:
            initials = "".join(f"{n[0]}." for n in given.split())
            author_str = f"{family}, {initials}"
        elif family:
            author_str = family

    parts: list[str] = []
    if author_str and year:
        parts.append(f"{author_str} ({year}).")
    elif author_str:
        parts.append(f"{author_str}.")
    elif year:
        parts.append(f"({year}).")

    if title:
        parts.append(f"{title}.")

    if journal:
        parts.append(f"*{journal}*.")

    parts.append(f"https://doi.org/{doi_str}")

    return " ".join(parts)
