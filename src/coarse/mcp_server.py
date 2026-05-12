"""MCP server exposing coarse.chat.ChatSession as tools.

Launched as a subprocess by an MCP client (e.g. Claude Code) per .mcp.json.
Holds a module-level registry of active ChatSession objects keyed by UUID.
Sessions are in-memory only and do not persist across server restarts.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from coarse.agents.citation_verify import CitationVerifyAgent
from coarse.chat import ChatSession, run_literature_query
from coarse.extraction import extract_file
from coarse.fetch import fetch_paper as _fetch_paper
from coarse.llm import LLMClient

mcp = FastMCP("coarse-chat")

_sessions: dict[str, ChatSession] = {}


@mcp.tool()
def ping() -> str:
    """Health check. Returns the literal string 'pong'."""
    return "pong"


@mcp.tool()
def start_chat(paper_path: str, review_path: str, model: str | None = None) -> str:
    """Start a chat session with the coarse reviewer over a paper and a prior review.

    Args:
        paper_path: Absolute path to the paper file (PDF, MD, TXT, TeX, DOCX, HTML, EPUB).
        review_path: Absolute path to the markdown review previously produced by `coarse review`.
        model: Optional LiteLLM model string. Defaults to the user's coarse config default.

    Returns:
        A session id string. Use it with `ask` and `end_session`.
    """
    paper = Path(paper_path)
    review = Path(review_path)
    if not paper.exists():
        raise FileNotFoundError(f"paper not found: {paper}")
    if not review.exists():
        raise FileNotFoundError(f"review not found: {review}")

    session = ChatSession(paper_path=paper, review_path=review, model=model)
    session_id = uuid.uuid4().hex
    _sessions[session_id] = session
    return session_id


@mcp.tool()
def ask(session_id: str, question: str) -> str:
    """Send a question to a running chat session and return the reviewer's reply.

    The session may transparently consult Perplexity Sonar Pro for literature
    lookups (up to 3 hops per turn) before composing the final reply.

    Args:
        session_id: Id returned by `start_chat`.
        question: The user's question for this turn.

    Returns:
        The reviewer's reply as plain markdown text.
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"unknown session: {session_id}")
    return session.ask(question)


@mcp.tool()
def list_sessions() -> list[dict]:
    """List all active chat sessions and their metadata.

    Returns:
        A list of dicts with keys: session_id, paper_path, review_path, model, turns.
        `turns` counts the user questions asked so far in the session (excludes the
        initial bootstrap user message and any literature-search injection messages).
    """
    out = []
    for sid, sess in _sessions.items():
        # Initial history is [system, user_initial]; subsequent user messages are
        # either turn questions or literature-search results. Turn questions are
        # the ones that did NOT come from the search-result injection, which
        # always begins with the literal "Literature search results for `".
        turn_questions = sum(
            1
            for m in sess.history[2:]
            if m["role"] == "user"
            and not m["content"].startswith("Literature search results for `")
        )
        out.append(
            {
                "session_id": sid,
                "paper_path": str(sess.paper_path),
                "review_path": str(sess.review_path),
                "model": sess.model,
                "turns": turn_questions,
            }
        )
    return out


@mcp.tool()
def search_literature(query: str) -> str:
    """Run a one-shot literature search via Perplexity Sonar Pro.

    Unlike `ask`, this does not require a session or a paper — it is a direct
    pass-through to the web-grounded search model. Use it when you want to look
    something up in the literature without attaching a paper context.

    Args:
        query: Free-text search query (e.g. "recent work on regression
            discontinuity with distribution-valued outcomes").

    Returns:
        Markdown text with findings and citations, as returned by the model.
    """
    return run_literature_query(query)


@mcp.tool()
def end_session(session_id: str) -> str:
    """Drop a chat session from memory. Frees its history.

    Args:
        session_id: Id returned by `start_chat`.

    Returns:
        A short confirmation string.
    """
    if session_id not in _sessions:
        raise KeyError(f"unknown session: {session_id}")
    del _sessions[session_id]
    return f"Session {session_id} ended."


@mcp.tool()
def fetch_paper(
    doi: str | None = None,
    title: str | None = None,
    url: str | None = None,
    output_dir: str = ".",
) -> dict:
    """Download a paper PDF by DOI or title.

    Resolution order: Unpaywall -> Semantic Scholar -> OpenAlex -> direct DOI fetch.
    At least one of doi, title, or url must be provided.

    When url is provided it is used as a final fallback if all other sources
    fail. If the URL serves or links to a PDF it is downloaded as usual. If
    the URL is an HTML-only page (e.g. a Pew Research article), the article
    text is extracted with trafilatura and saved as a .md file — in that
    case no extract_paper call is needed before verify_citation.

    Args:
        doi: DOI string (e.g. "10.1234/example"). Optional if title is given.
        title: Full paper title. Used when DOI is unavailable or all DOI-based
            sources fail.
        url: Direct URL to the paper or its landing page. Used as a final
            fallback after all other resolvers fail. Supports PDF URLs,
            HTML landing pages with a citation_pdf_url meta tag, and
            HTML-only pages (saved as .md via trafilatura).
        output_dir: Directory to save the downloaded file. Created if it does
            not exist. Defaults to current directory.

    Returns:
        Dict with keys: path (str or None), title, authors, year, doi,
        fetch_source ("unpaywall"|"semantic_scholar"|"openalex"|"direct"|
        "url"|"not_found"). path is None when no file could be retrieved.
        When fetch_source is "url" and path ends in ".md", the file is
        already extracted markdown — pass it directly to verify_citation.
    """
    return _fetch_paper(doi=doi, title=title, url=url, output_dir=output_dir)


@mcp.tool()
def extract_paper(
    paper_path: str,
    output_path: str | None = None,
) -> dict:
    """Convert a PDF to markdown using coarse's extraction infrastructure.

    Uses docling locally for clean PDFs (free). Falls back to Mistral OCR via
    OpenRouter for scanned documents (incurs cost).

    Args:
        paper_path: Absolute path to the PDF file.
        output_path: Where to save the markdown. Defaults to same directory
            as the PDF with a .md extension.

    Returns:
        Dict with keys: path (absolute path to the .md file), garble_ratio
        (float 0.0-1.0; high values indicate poor extraction quality).
    """
    pdf = Path(paper_path)
    if not pdf.exists():
        raise FileNotFoundError(f"paper not found: {pdf}")

    result = extract_file(pdf)

    out = Path(output_path) if output_path else pdf.with_suffix(".md")
    out.write_text(result.full_markdown, encoding="utf-8")

    return {"path": str(out.resolve()), "garble_ratio": result.garble_ratio}


@mcp.tool()
def verify_citation(
    paper_markdown_path: str,
    citation_claim: str,
) -> dict:
    """Adversarial two-phase citation accuracy check.

    Phase 1: reads the paper independently (no citation context) and produces
    a neutral summary. Phase 2: compares that summary against the citation
    claim using an adversarial reviewer instructed to find mismatches.

    Args:
        paper_markdown_path: Absolute path to the extracted .md file (not the
            PDF). Run extract_paper first if you only have a PDF.
        citation_claim: The claim a document makes about this paper — the
            sentence or paragraph that cites it.

    Returns:
        Dict with keys: verdict ("Supports"|"Weak"|"Mismatch"), paper_summary
        (Phase 1 independent summary), adversarial_notes (what the reviewer
        found), sop_revision_needed (bool).
    """
    md_path = Path(paper_markdown_path)
    if not md_path.exists():
        raise FileNotFoundError(f"markdown not found: {md_path}")

    paper_text = md_path.read_text(encoding="utf-8")
    client = LLMClient()
    agent = CitationVerifyAgent(client=client)
    return agent.run(paper_text=paper_text, citation_claim=citation_claim)


def main() -> None:
    """Entry point for the `coarse-mcp` console script."""
    mcp.run()
