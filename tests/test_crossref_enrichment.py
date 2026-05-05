"""Tests for src/coarse/crossref.py — DOI extraction and CrossRef enrichment."""
from __future__ import annotations

import json
import socket
import urllib.error
from unittest.mock import MagicMock, patch

from coarse.crossref import enrich_citations, extract_doi


# ---------------------------------------------------------------------------
# extract_doi
# ---------------------------------------------------------------------------


def test_extract_doi_from_doi_org_url():
    assert extract_doi("https://doi.org/10.1080/00273171.2011.568786") == "10.1080/00273171.2011.568786"


def test_extract_doi_from_dx_doi_org_url():
    assert extract_doi("https://dx.doi.org/10.1038/nature12345") == "10.1038/nature12345"


def test_extract_doi_bare_doi_in_url():
    assert extract_doi("https://example.com/papers?doi=10.1234/abcd.2021.01") == "10.1234/abcd.2021.01"


def test_extract_doi_non_doi_url_returns_none():
    assert extract_doi("https://www.nber.org/papers/w12345") is None
