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
    raise NotImplementedError
