# Design: Full Bibliographic References in `search_literature`

**Date:** 2026-05-05
**Status:** Approved

## Problem

`search_literature` (the MCP tool in `mcp_server.py`) returns prose with inline `[1]`, `[2]` citation markers but no corresponding reference list. The existing citation-appending code in `LLMClient.complete_text` never fires because `perplexity` is absent from `PROVIDER_ENV_VARS`, causing `resolve_api_key` to fall back to `OPENROUTER_API_KEY`. litellm then routes through OpenRouter's Perplexity proxy, which strips the `citations` field before litellm sees it.

## Goal

Append a formatted bibliography (author, year, title, journal, DOI) after the prose for any citation whose URL contains a resolvable DOI. Non-DOI URLs fall back to the raw URL. Output is deterministic — no LLM enrichment, no hallucination risk.

## Architecture

Three files changed, one new module:

| File | Change |
|---|---|
| `src/coarse/config.py` | Add `"perplexity": "PERPLEXITY_API_KEY"` to `PROVIDER_ENV_VARS` |
| `src/coarse/crossref.py` | New module: DOI extraction + CrossRef API enrichment |
| `src/coarse/llm.py` | `complete_text` calls `enrich_citations()` instead of raw URL append |
| `tests/test_crossref.py` | New test file |

## Data Flow

```
Perplexity response
    → response.citations: list[str]  (URLs)
    → crossref.enrich_citations(urls)
        → per URL: extract_dois() → DOI found?
              yes → GET https://api.crossref.org/works/{doi} (5s timeout)
                    → format: "Author et al. (Year). Title. Journal. https://doi.org/..."
              no  → raw URL
        → parallel via ThreadPoolExecutor(max_workers=5)
        → order preserved
    → appended to content as **Sources:** numbered list
```

### CrossRef fields used

`author[0].family` + `given` initials, `published.date-parts[0][0]` (year), `title[0]`, `container-title[0]` (journal), `DOI`. All fields are optional — any missing field is omitted gracefully.

### Example output

```
**Sources:**
1. Austin, P.C. (2011). An Introduction to Propensity Score Methods... *Journal of Multivariate Behavioral Research*, 46(3). https://doi.org/10.1080/00273171.2011.568786
2. https://www.nber.org/papers/w12345
```

## New Module: `src/coarse/crossref.py`

Two public functions:

```python
def extract_doi(url: str) -> str | None:
    """Return bare DOI string from a URL, or None if not found."""

def enrich_citations(urls: list[str]) -> list[str]:
    """Return formatted bibliography strings; falls back to raw URL on any failure."""
```

DOI patterns handled: `doi.org/{doi}`, `dx.doi.org/{doi}`, inline `10.\d{4}/` bare DOIs in any URL.

CrossRef endpoint: `https://api.crossref.org/works/{doi}` — free, unauthenticated, covers ~135M works.

## Error Handling

All enrichment failures are best-effort. A failure never breaks the search result.

| Condition | Behaviour |
|---|---|
| No `PERPLEXITY_API_KEY` | Falls back to OpenRouter; `response.citations` is empty; no `**Sources:**` block |
| CrossRef network error / timeout | That citation falls back to raw URL |
| CrossRef 404 / unexpected schema | Raw URL fallback |
| Empty `citations` list | `**Sources:**` block omitted entirely |
| Partial failures | Mixed list (some enriched, some raw URLs); order preserved |

All failures logged at `DEBUG` level only.

## Tests: `tests/test_crossref.py`

| Test | Coverage |
|---|---|
| DOI extracted from `doi.org/` URL | `extract_doi` happy path |
| DOI extracted from `dx.doi.org/` URL | alternate prefix |
| DOI extracted from bare `10.xxxx/` in URL | inline pattern |
| Non-DOI URL returns `None` | NBER, PubMed, plain HTTPS |
| Happy path CrossRef response | Full formatted string with author, year, title, journal, DOI |
| CrossRef 404 | Falls back to raw URL, no exception |
| CrossRef timeout | Falls back to raw URL, no exception |
| Missing `container-title` field | Journal omitted, rest formatted correctly |
| Mixed list (DOI + non-DOI URLs) | Order preserved; non-DOI is raw URL |

`llm.py` citation block covered by existing `test_llm.py` patterns via monkeypatched `litellm.completion`.

## Out of Scope

- Scraping non-DOI URLs for metadata
- LLM-based bibliography enrichment
- Scraping non-DOI URLs for metadata (beyond CrossRef)
- LLM-based bibliography enrichment

Note: the `llm.py` change applies to all `complete_text` callers, including `_search_perplexity` in `agents/literature.py`. That call site will also surface enriched citations once `PERPLEXITY_API_KEY` is configured — this is a side effect, not a goal, but it is harmless and desirable.
