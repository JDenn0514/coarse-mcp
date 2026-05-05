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


# ---------------------------------------------------------------------------
# enrich_citations helpers
# ---------------------------------------------------------------------------


def _make_crossref_response(doi: str = "10.1234/test", include_journal: bool = True) -> bytes:
    data: dict = {
        "message": {
            "author": [{"family": "Smith", "given": "J."}],
            "published": {"date-parts": [[2020]]},
            "title": ["A Great Paper"],
            "DOI": doi,
        }
    }
    if include_journal:
        data["message"]["container-title"] = ["Journal of Statistics"]
    return json.dumps(data).encode()


def _mock_urlopen(response_bytes: bytes) -> MagicMock:
    """Return a context-manager mock whose .read() yields response_bytes."""
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.return_value = response_bytes
    return mock_resp


# ---------------------------------------------------------------------------
# enrich_citations
# ---------------------------------------------------------------------------


def test_enrich_citations_happy_path():
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(_make_crossref_response())):
        result = enrich_citations(["https://doi.org/10.1234/test"])

    assert len(result) == 1
    ref = result[0]
    assert "Smith, J." in ref
    assert "(2020)" in ref
    assert "A Great Paper" in ref
    assert "*Journal of Statistics*" in ref
    assert "https://doi.org/10.1234/test" in ref


def test_enrich_citations_crossref_404():
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(url="", code=404, msg="Not Found", hdrs=None, fp=None),
    ):
        result = enrich_citations(["https://doi.org/10.1234/notfound"])

    assert result == ["https://doi.org/10.1234/notfound"]


def test_enrich_citations_crossref_timeout():
    with patch("urllib.request.urlopen", side_effect=socket.timeout("timed out")):
        result = enrich_citations(["https://doi.org/10.1234/slow"])

    assert result == ["https://doi.org/10.1234/slow"]


def test_enrich_citations_missing_container_title():
    with patch(
        "urllib.request.urlopen",
        return_value=_mock_urlopen(_make_crossref_response(include_journal=False)),
    ):
        result = enrich_citations(["https://doi.org/10.1234/test"])

    assert len(result) == 1
    ref = result[0]
    assert "Smith, J." in ref
    assert "*Journal" not in ref
    assert "https://doi.org/10.1234/test" in ref


def test_enrich_citations_mixed_doi_and_non_doi():
    with patch(
        "urllib.request.urlopen",
        return_value=_mock_urlopen(_make_crossref_response()),
    ):
        result = enrich_citations([
            "https://doi.org/10.1234/test",
            "https://www.nber.org/papers/w12345",
        ])

    assert len(result) == 2
    assert "Smith, J." in result[0]
    assert result[1] == "https://www.nber.org/papers/w12345"
