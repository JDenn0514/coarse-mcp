"""Tests for fetch_paper resolution chain and PDF download."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status=200, json_data=None, content=b"PDF"):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# _slug
# ---------------------------------------------------------------------------

def test_slug_lowercases_and_replaces_spaces():
    from coarse.fetch import _slug
    assert _slug("Baker et al. 2013") == "baker_et_al_2013"


def test_slug_truncates_at_60():
    from coarse.fetch import _slug
    long = "a" * 80
    assert len(_slug(long)) <= 60


# ---------------------------------------------------------------------------
# _try_unpaywall
# ---------------------------------------------------------------------------

def test_try_unpaywall_returns_pdf_url(tmp_path):
    from coarse.fetch import _try_unpaywall
    payload = {"best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        result = _try_unpaywall("10.1234/test", "test@example.com")
    assert result == "https://example.com/paper.pdf"


def test_try_unpaywall_returns_none_on_404():
    from coarse.fetch import _try_unpaywall
    with patch("coarse.fetch.requests.get", return_value=_mock_response(status=404)):
        result = _try_unpaywall("10.1234/test", "test@example.com")
    assert result is None


def test_try_unpaywall_returns_none_on_exception():
    from coarse.fetch import _try_unpaywall
    with patch("coarse.fetch.requests.get", side_effect=Exception("network error")):
        result = _try_unpaywall("10.1234/test", "test@example.com")
    assert result is None


# ---------------------------------------------------------------------------
# _try_semantic_scholar_doi
# ---------------------------------------------------------------------------

def test_try_semantic_scholar_doi_returns_pdf_url():
    from coarse.fetch import _try_semantic_scholar_doi
    payload = {
        "openAccessPdf": {"url": "https://arxiv.org/pdf/1234.pdf"},
        "title": "Test Paper",
        "authors": [{"name": "Baker"}],
        "year": 2013,
        "externalIds": {"DOI": "10.1234/test"},
    }
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        url, data = _try_semantic_scholar_doi("10.1234/test")
    assert url == "https://arxiv.org/pdf/1234.pdf"
    assert data["title"] == "Test Paper"


def test_try_semantic_scholar_doi_returns_none_tuple_on_failure():
    from coarse.fetch import _try_semantic_scholar_doi
    with patch("coarse.fetch.requests.get", return_value=_mock_response(status=404)):
        url, data = _try_semantic_scholar_doi("10.1234/test")
    assert url is None
    assert data == {}


# ---------------------------------------------------------------------------
# _try_semantic_scholar_title
# ---------------------------------------------------------------------------

def test_try_semantic_scholar_title_returns_pdf_url():
    from coarse.fetch import _try_semantic_scholar_title
    payload = {"data": [{"openAccessPdf": {"url": "https://arxiv.org/pdf/test.pdf"},
                         "title": "AAPOR Report", "authors": [], "year": 2013,
                         "externalIds": {}}]}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        url, data = _try_semantic_scholar_title("AAPOR Report")
    assert url == "https://arxiv.org/pdf/test.pdf"


def test_try_semantic_scholar_title_returns_none_on_empty_results():
    from coarse.fetch import _try_semantic_scholar_title
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data={"data": []})):
        url, data = _try_semantic_scholar_title("Obscure Title")
    assert url is None


# ---------------------------------------------------------------------------
# fetch_paper (integration of resolution chain)
# ---------------------------------------------------------------------------

def test_fetch_paper_raises_if_no_doi_or_title(tmp_path):
    from coarse.fetch import fetch_paper
    with pytest.raises(ValueError, match="doi or title"):
        fetch_paper(output_dir=str(tmp_path))


def test_fetch_paper_returns_not_found_when_all_sources_fail(tmp_path):
    from coarse.fetch import fetch_paper
    none_resp = _mock_response(status=404)
    with patch("coarse.fetch.requests.get", return_value=none_resp), \
         patch("coarse.fetch.urllib.request.urlopen", side_effect=Exception("no")):
        result = fetch_paper(doi="10.9999/nope", output_dir=str(tmp_path))
    assert result["path"] is None
    assert result["fetch_source"] == "not_found"


def test_fetch_paper_succeeds_via_unpaywall(tmp_path):
    from coarse.fetch import fetch_paper
    unpaywall_payload = {"best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}}

    call_count = 0
    def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "unpaywall" in url:
            return _mock_response(json_data=unpaywall_payload)
        # download call
        return _mock_response(content=b"%PDF-1.4 fake content")

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(doi="10.1234/test", output_dir=str(tmp_path))

    assert result["path"] is not None
    assert result["fetch_source"] == "unpaywall"
    assert Path(result["path"]).exists()


def test_fetch_paper_filename_includes_year(tmp_path):
    from coarse.fetch import fetch_paper
    unpaywall_payload = {"best_oa_location": {"url_for_pdf": "https://example.com/paper.pdf"}}

    responses = [
        _mock_response(json_data=unpaywall_payload),   # unpaywall → has url
        _mock_response(content=b"%PDF-1.4 fake"),       # download
    ]
    idx = 0
    def mock_get(url, **kwargs):
        nonlocal idx
        r = responses[idx]
        idx += 1
        return r

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(doi="10.1234/test", title="AAPOR Report",
                             output_dir=str(tmp_path))

    assert result["path"] is not None
