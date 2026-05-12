"""Tests for fetch_paper resolution chain and PDF download."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(status=200, json_data=None, content=b"PDF", content_type="application/pdf"):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.content = content
    resp.raise_for_status = MagicMock()
    resp.headers = {"Content-Type": content_type}
    resp.text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else ""
    # Support use as context manager (for stream=True)
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
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
# _extract_pdf_from_landing_page
# ---------------------------------------------------------------------------

def test_extract_pdf_citation_meta_tag():
    from coarse.fetch import _extract_pdf_from_landing_page
    html = '<meta name="citation_pdf_url" content="https://example.com/paper.pdf">'
    assert _extract_pdf_from_landing_page("https://example.com/", html) == \
        "https://example.com/paper.pdf"


def test_extract_pdf_citation_meta_tag_reversed_attrs():
    from coarse.fetch import _extract_pdf_from_landing_page
    html = '<meta content="https://example.com/paper.pdf" name="citation_pdf_url">'
    assert _extract_pdf_from_landing_page("https://example.com/", html) == \
        "https://example.com/paper.pdf"


def test_extract_pdf_direct_href():
    from coarse.fetch import _extract_pdf_from_landing_page
    html = '<a href="/reports/nonprob_sampling.pdf">Download</a>'
    result = _extract_pdf_from_landing_page("https://pew.org/page/", html)
    assert result == "https://pew.org/reports/nonprob_sampling.pdf"


def test_extract_pdf_returns_none_when_no_link():
    from coarse.fetch import _extract_pdf_from_landing_page
    assert _extract_pdf_from_landing_page("https://example.com/", "<html>No PDF here</html>") is None


# ---------------------------------------------------------------------------
# _resolve_to_pdf
# ---------------------------------------------------------------------------

def test_resolve_to_pdf_returns_url_for_pdf_content_type():
    from coarse.fetch import _resolve_to_pdf
    with patch("coarse.fetch.requests.get",
               return_value=_mock_response(content_type="application/pdf")):
        assert _resolve_to_pdf("https://example.com/paper.pdf") == \
            "https://example.com/paper.pdf"


def test_resolve_to_pdf_follows_html_landing_page():
    from coarse.fetch import _resolve_to_pdf
    html = b'<meta name="citation_pdf_url" content="https://example.com/full.pdf">'
    with patch("coarse.fetch.requests.get",
               return_value=_mock_response(content=html, content_type="text/html")):
        assert _resolve_to_pdf("https://example.com/abstract") == \
            "https://example.com/full.pdf"


def test_resolve_to_pdf_returns_none_when_html_has_no_pdf_link():
    from coarse.fetch import _resolve_to_pdf
    with patch("coarse.fetch.requests.get",
               return_value=_mock_response(content=b"<html>no pdf</html>",
                                           content_type="text/html")):
        assert _resolve_to_pdf("https://example.com/abstract") is None


def test_resolve_to_pdf_returns_none_on_404():
    from coarse.fetch import _resolve_to_pdf
    with patch("coarse.fetch.requests.get", return_value=_mock_response(status=404)):
        assert _resolve_to_pdf("https://example.com/paper") is None


# ---------------------------------------------------------------------------
# _discover_sibling_pages
# ---------------------------------------------------------------------------

def test_discover_sibling_pages_finds_same_depth_links():
    from coarse.fetch import _discover_sibling_pages
    # Simulate Pew report: main page links to sibling section pages
    html = """
    <nav>
      <a href="/methods/2018/01/26/how-different-weighting-methods-work/">Methods</a>
      <a href="/methods/2018/01/26/reducing-bias-on-benchmarks/">Reducing Bias</a>
      <a href="/methods/2018/01/27/unrelated-article/">Unrelated (different date)</a>
      <a href="/methods/2018/">Category page (too shallow)</a>
      <a href="/methods/2018/01/26/how-different-weighting-methods-work/appendix/">Sub-page (too deep)</a>
    </nav>
    """
    url = "https://www.pewresearch.org/methods/2018/01/26/for-weighting-online-opt-in-samples-what-matters-most/"
    siblings = _discover_sibling_pages(url, html)
    assert len(siblings) == 2
    assert any("how-different-weighting-methods-work" in s for s in siblings)
    assert any("reducing-bias-on-benchmarks" in s for s in siblings)
    assert not any("unrelated-article" in s for s in siblings)
    assert not any("appendix" in s for s in siblings)


# ---------------------------------------------------------------------------
# _fetch_html_as_markdown
# ---------------------------------------------------------------------------

def test_fetch_html_as_markdown_saves_md_file(tmp_path):
    from coarse.fetch import _fetch_html_as_markdown
    import types
    mock_tra = types.SimpleNamespace(
        fetch_url=lambda url: "<html>content</html>",
        extract=lambda html, **kw: "## Article\n\nContent here.",
    )
    with patch("coarse.fetch._trafilatura", mock_tra):
        result = _fetch_html_as_markdown(
            "https://pew.org/article", tmp_path, "article_slug", crawl_siblings=False
        )
    assert result is not None
    assert result.suffix == ".md"
    assert result.name == "article_slug.md"
    assert result.read_text(encoding="utf-8") == "## Article\n\nContent here."


def test_fetch_html_as_markdown_returns_none_when_fetch_returns_none(tmp_path):
    from coarse.fetch import _fetch_html_as_markdown
    import types
    mock_tra = types.SimpleNamespace(
        fetch_url=lambda url: None,
        extract=lambda html, **kw: "irrelevant",
    )
    with patch("coarse.fetch._trafilatura", mock_tra):
        result = _fetch_html_as_markdown("https://pew.org/article", tmp_path, "slug")
    assert result is None


def test_fetch_html_as_markdown_returns_none_when_extract_returns_none(tmp_path):
    from coarse.fetch import _fetch_html_as_markdown
    import types
    mock_tra = types.SimpleNamespace(
        fetch_url=lambda url: "<html>content</html>",
        extract=lambda html, **kw: None,
    )
    with patch("coarse.fetch._trafilatura", mock_tra):
        result = _fetch_html_as_markdown("https://pew.org/article", tmp_path, "slug")
    assert result is None


def test_fetch_html_as_markdown_returns_none_when_trafilatura_not_installed(tmp_path):
    from coarse.fetch import _fetch_html_as_markdown
    with patch("coarse.fetch._trafilatura", None):
        result = _fetch_html_as_markdown("https://pew.org/article", tmp_path, "slug")
    assert result is None


def test_fetch_html_as_markdown_crawls_sibling_pages(tmp_path):
    """For multi-page reports, sibling pages at the same path depth are fetched."""
    from coarse.fetch import _fetch_html_as_markdown
    import types

    main_html = (
        '<article>Main content.</article>'
        '<nav><a href="/methods/2018/01/26/section-two/">Section Two</a></nav>'
    )
    section_html = "<article>Section two content.</article>"

    fetch_calls = []
    def mock_fetch(url):
        fetch_calls.append(url)
        return section_html if "section-two" in url else main_html

    mock_tra = types.SimpleNamespace(
        fetch_url=mock_fetch,
        extract=lambda html, **kw: "extracted: " + ("section-two" if "section-two" in html else "main"),
    )
    with patch("coarse.fetch._trafilatura", mock_tra):
        result = _fetch_html_as_markdown(
            "https://www.pewresearch.org/methods/2018/01/26/main-article/",
            tmp_path, "main_article",
        )
    assert result is not None
    content = result.read_text(encoding="utf-8")
    assert "main" in content
    assert "section-two" in content   # sibling was concatenated
    assert len(fetch_calls) == 2      # main + 1 sibling


def test_fetch_html_as_markdown_no_sibling_crawl_when_disabled(tmp_path):
    """crawl_siblings=False fetches only the primary URL."""
    from coarse.fetch import _fetch_html_as_markdown
    import types

    main_html = (
        '<article>Main content.</article>'
        '<nav><a href="/methods/2018/01/26/section-two/">Section Two</a></nav>'
    )
    fetch_calls = []
    mock_tra = types.SimpleNamespace(
        fetch_url=lambda url: fetch_calls.append(url) or main_html,
        extract=lambda html, **kw: "main content only",
    )
    with patch("coarse.fetch._trafilatura", mock_tra):
        result = _fetch_html_as_markdown(
            "https://www.pewresearch.org/methods/2018/01/26/main-article/",
            tmp_path, "main_article",
            crawl_siblings=False,
        )
    assert result is not None
    assert len(fetch_calls) == 1   # only the main page


# ---------------------------------------------------------------------------
# _download HTML guard
# ---------------------------------------------------------------------------

def test_download_rejects_html_content_type(tmp_path):
    from coarse.fetch import _download
    with patch("coarse.fetch.requests.get",
               return_value=_mock_response(content=b"<html>page</html>",
                                           content_type="text/html")):
        assert _download("https://example.com/page", tmp_path / "out.pdf") is False
    assert not (tmp_path / "out.pdf").exists()


# ---------------------------------------------------------------------------
# fetch_paper landing page integration
# ---------------------------------------------------------------------------

def test_fetch_paper_follows_landing_page_to_pdf(tmp_path):
    from coarse.fetch import fetch_paper
    unpaywall_payload = {"best_oa_location": {"url": "https://pew.org/report-page"}}
    html = b'<meta name="citation_pdf_url" content="https://pew.org/report.pdf">'

    def mock_get(url, **kwargs):
        if "unpaywall" in url:
            return _mock_response(json_data=unpaywall_payload, content_type="application/json")
        if "report-page" in url:
            return _mock_response(content=html, content_type="text/html")
        # PDF download
        return _mock_response(content=b"%PDF-1.4 fake", content_type="application/pdf")

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(doi="10.1234/pew", output_dir=str(tmp_path))

    assert result["path"] is not None
    assert result["fetch_source"] == "unpaywall"
    assert Path(result["path"]).exists()


# ---------------------------------------------------------------------------
# _title_similar
# ---------------------------------------------------------------------------

def test_title_similar_accepts_matching_title():
    from coarse.fetch import _title_similar
    assert _title_similar(
        "AAPOR Report on Non-Probability Sampling",
        "AAPOR Report on Non-Probability Sampling",
    )


def test_title_similar_rejects_unrelated_title():
    from coarse.fetch import _title_similar
    assert not _title_similar(
        "Handling non-probability samples through inverse probability weighting",
        "Accurate structure prediction of biomolecular interactions with AlphaFold3",
    )


def test_title_similar_accepts_minor_punctuation_differences():
    from coarse.fetch import _title_similar
    assert _title_similar(
        "How Different Weighting Methods Work",
        "How different weighting methods work",
    )


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
                         "title": "AAPOR Report on Non-Probability Sampling",
                         "authors": [], "year": 2013, "externalIds": {}}]}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        url, data = _try_semantic_scholar_title("AAPOR Report on Non-Probability Sampling")
    assert url == "https://arxiv.org/pdf/test.pdf"


def test_try_semantic_scholar_title_returns_none_on_empty_results():
    from coarse.fetch import _try_semantic_scholar_title
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data={"data": []})):
        url, data = _try_semantic_scholar_title("Obscure Title")
    assert url is None


def test_try_semantic_scholar_title_rejects_mismatched_title():
    from coarse.fetch import _try_semantic_scholar_title
    payload = {"data": [{"openAccessPdf": {"url": "https://arxiv.org/pdf/alphafold.pdf"},
                         "title": "Accurate structure prediction of biomolecular interactions",
                         "authors": [], "year": 2024, "externalIds": {}}]}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        url, data = _try_semantic_scholar_title(
            "Handling non-probability samples through inverse probability weighting"
        )
    assert url is None
    assert data == {}


# ---------------------------------------------------------------------------
# _try_openalex_title
# ---------------------------------------------------------------------------

def test_try_openalex_title_rejects_mismatched_title():
    from coarse.fetch import _try_openalex_title
    payload = {"results": [{
        "title": "Accurate structure prediction of biomolecular interactions",
        "primary_location": {"pdf_url": "https://example.com/alphafold.pdf"},
    }]}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        result = _try_openalex_title(
            "Handling non-probability samples through inverse probability weighting",
            "test@example.com",
        )
    assert result is None


def test_try_openalex_title_accepts_matching_title():
    from coarse.fetch import _try_openalex_title
    payload = {"results": [{
        "title": "Handling non-probability samples through inverse probability weighting methods",
        "primary_location": {"pdf_url": "https://example.com/haziza.pdf"},
    }]}
    with patch("coarse.fetch.requests.get", return_value=_mock_response(json_data=payload)):
        result = _try_openalex_title(
            "Handling non-probability samples through inverse probability weighting methods",
            "test@example.com",
        )
    assert result == "https://example.com/haziza.pdf"


# ---------------------------------------------------------------------------
# fetch_paper (integration of resolution chain)
# ---------------------------------------------------------------------------

def test_fetch_paper_raises_if_no_doi_or_title(tmp_path):
    from coarse.fetch import fetch_paper
    with pytest.raises(ValueError, match="doi"):
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

    def mock_get(url, **kwargs):
        if "unpaywall" in url:
            return _mock_response(json_data=unpaywall_payload, content_type="application/json")
        # _resolve_to_pdf check and _download both hit the PDF URL
        return _mock_response(content=b"%PDF-1.4 fake content", content_type="application/pdf")

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

    def mock_get(url, **kwargs):
        if "unpaywall" in url:
            return _mock_response(json_data=unpaywall_payload, content_type="application/json")
        return _mock_response(content=b"%PDF-1.4 fake", content_type="application/pdf")

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(doi="10.1234/test", title="AAPOR Report",
                             output_dir=str(tmp_path))

    assert result["path"] is not None


# ---------------------------------------------------------------------------
# fetch_paper — url parameter (PDF path)
# ---------------------------------------------------------------------------

def test_fetch_paper_url_resolves_landing_page_to_pdf(tmp_path):
    from coarse.fetch import fetch_paper
    # Simulate: S2 and OpenAlex return nothing; URL is a landing page with
    # a citation_pdf_url meta tag pointing to a PDF.
    html_landing = b'<meta name="citation_pdf_url" content="https://pew.org/report.pdf">'

    def mock_get(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response(status=200, json_data={"data": []})
        if "openalex" in url:
            return _mock_response(status=200, json_data={"results": []})
        if "pew.org/report-page" in url:
            return _mock_response(content=html_landing, content_type="text/html")
        # PDF download
        return _mock_response(content=b"%PDF-1.4 fake", content_type="application/pdf")

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.urllib.request.urlopen", side_effect=Exception("no")), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(
            title="Evaluating Online Nonprobability Surveys",
            url="https://pew.org/report-page",
            output_dir=str(tmp_path),
        )

    assert result["path"] is not None
    assert result["path"].endswith(".pdf")
    assert result["fetch_source"] == "url"
    assert Path(result["path"]).exists()


def test_fetch_paper_url_direct_pdf_downloads_successfully(tmp_path):
    from coarse.fetch import fetch_paper

    def mock_get(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response(status=200, json_data={"data": []})
        if "openalex" in url:
            return _mock_response(status=200, json_data={"results": []})
        # URL itself serves a PDF directly
        return _mock_response(content=b"%PDF-1.4 direct", content_type="application/pdf")

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch.urllib.request.urlopen", side_effect=Exception("no")), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(
            title="Mercer Lau Kennedy 2018 weighting",
            url="https://pew.org/report.pdf",
            output_dir=str(tmp_path),
        )

    assert result["path"] is not None
    assert result["path"].endswith(".pdf")
    assert result["fetch_source"] == "url"
    assert Path(result["path"]).exists()


# ---------------------------------------------------------------------------
# fetch_paper — url parameter (HTML-only path)
# ---------------------------------------------------------------------------

def test_fetch_paper_url_html_only_clips_to_markdown(tmp_path):
    from coarse.fetch import fetch_paper
    import types

    def mock_get(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response(status=200, json_data={"data": []})
        if "openalex" in url:
            return _mock_response(status=200, json_data={"results": []})
        # URL is HTML with no PDF link
        return _mock_response(
            content=b"<html><body>Article text</body></html>",
            content_type="text/html",
        )

    mock_tra = types.SimpleNamespace(
        fetch_url=lambda u: "<html>Article content</html>",
        extract=lambda html, **kw: "## Mercer et al. 2018\n\nFindings here.",
    )

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch._trafilatura", mock_tra), \
         patch("coarse.fetch.urllib.request.urlopen", side_effect=Exception("no")), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(
            title="For Weighting Online Opt-In Samples What Matters Most",
            url="https://www.pewresearch.org/methods/2018/01/26/for-weighting/",
            output_dir=str(tmp_path),
        )

    assert result["path"] is not None
    assert result["path"].endswith(".md")
    assert result["fetch_source"] == "url"
    assert Path(result["path"]).exists()
    text = Path(result["path"]).read_text(encoding="utf-8")
    assert "Mercer" in text


def test_fetch_paper_url_returns_not_found_when_html_extraction_fails(tmp_path):
    from coarse.fetch import fetch_paper
    import types

    def mock_get(url, **kwargs):
        if "semanticscholar" in url:
            return _mock_response(status=200, json_data={"data": []})
        if "openalex" in url:
            return _mock_response(status=200, json_data={"results": []})
        return _mock_response(
            content=b"<html>no pdf</html>",
            content_type="text/html",
        )

    mock_tra = types.SimpleNamespace(
        fetch_url=lambda u: None,   # trafilatura fetch fails
        extract=lambda html, **kw: None,
    )

    with patch("coarse.fetch.requests.get", side_effect=mock_get), \
         patch("coarse.fetch._trafilatura", mock_tra), \
         patch("coarse.fetch.urllib.request.urlopen", side_effect=Exception("no")), \
         patch("coarse.fetch.load_config") as mock_cfg:
        mock_cfg.return_value.unpaywall_email = "test@example.com"
        result = fetch_paper(
            title="Some Pew Report",
            url="https://www.pewresearch.org/methods/unreachable/",
            output_dir=str(tmp_path),
        )

    assert result["path"] is None
    assert result["fetch_source"] == "not_found"
