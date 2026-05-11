"""Tests for the coarse MCP server."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coarse.mcp_server import (
    _sessions,
    ask,
    end_session,
    list_sessions,
    mcp,
    ping,
    search_literature,
    start_chat,
)


def test_ping_returns_pong() -> None:
    assert ping() == "pong"


def test_server_name_is_coarse_chat() -> None:
    assert mcp.name == "coarse-chat"


@pytest.fixture(autouse=True)
def _clear_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture
def fake_paper(tmp_path: Path) -> Path:
    p = tmp_path / "paper.md"
    p.write_text("# Title\n\nThis paper studies X.\n", encoding="utf-8")
    return p


@pytest.fixture
def fake_review(tmp_path: Path) -> Path:
    r = tmp_path / "review.md"
    r.write_text("## Overview\n\nThe paper is about X.\n", encoding="utf-8")
    return r


def test_start_chat_returns_uuid_string(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    assert isinstance(session_id, str)
    assert len(session_id) >= 32  # uuid4 hex is 32+ chars


def test_start_chat_stores_session_in_registry(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    assert session_id in _sessions
    assert "studies X" in _sessions[session_id].paper_text


def test_start_chat_passes_model_through(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review), model="openai/gpt-4o-mini")
    assert _sessions[session_id].model == "openai/gpt-4o-mini"


def test_start_chat_raises_when_paper_missing(tmp_path: Path, fake_review: Path) -> None:
    missing = tmp_path / "does_not_exist.md"
    with pytest.raises(FileNotFoundError, match="paper"):
        start_chat(str(missing), str(fake_review))


def test_start_chat_raises_when_review_missing(fake_paper: Path, tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.md"
    with pytest.raises(FileNotFoundError, match="review"):
        start_chat(str(fake_paper), str(missing))


def test_ask_returns_assistant_reply(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    _sessions[session_id]._client = MagicMock(
        complete_text=MagicMock(return_value="Hello, author.")
    )
    reply = ask(session_id, "What is the paper about?")
    assert reply == "Hello, author."


def test_ask_appends_user_question_to_history(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    _sessions[session_id]._client = MagicMock(
        complete_text=MagicMock(return_value="Reply.")
    )
    ask(session_id, "Question one?")
    history = _sessions[session_id].history
    user_messages = [m["content"] for m in history if m["role"] == "user"]
    assert any("Question one?" in m for m in user_messages)


def test_ask_raises_for_unknown_session_id() -> None:
    with pytest.raises(KeyError, match="unknown session"):
        ask("nonexistent-id", "Hi.")


def test_list_sessions_empty_by_default() -> None:
    assert list_sessions() == []


def test_list_sessions_returns_metadata(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review), model="openai/gpt-4o-mini")
    sessions = list_sessions()
    assert len(sessions) == 1
    entry = sessions[0]
    assert entry["session_id"] == session_id
    assert entry["paper_path"].endswith("paper.md")
    assert entry["review_path"].endswith("review.md")
    assert entry["model"] == "openai/gpt-4o-mini"
    assert entry["turns"] == 0


def test_list_sessions_turns_counts_user_questions(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    _sessions[session_id]._client = MagicMock(
        complete_text=MagicMock(return_value="reply")
    )
    ask(session_id, "Q1?")
    ask(session_id, "Q2?")
    [entry] = list_sessions()
    assert entry["turns"] == 2


def test_end_session_removes_session(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    assert session_id in _sessions
    end_session(session_id)
    assert session_id not in _sessions


def test_end_session_returns_confirmation_string(fake_paper: Path, fake_review: Path) -> None:
    session_id = start_chat(str(fake_paper), str(fake_review))
    msg = end_session(session_id)
    assert session_id in msg
    assert "ended" in msg.lower()


def test_end_session_raises_for_unknown_id() -> None:
    with pytest.raises(KeyError, match="unknown session"):
        end_session("nonexistent-id")


def test_search_literature_calls_run_literature_query(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_run(query: str) -> str:
        captured["query"] = query
        return "## Findings\n\n- Paper A (2024) shows..."

    monkeypatch.setattr("coarse.mcp_server.run_literature_query", fake_run)
    reply = search_literature("recent work on RDD with distribution outcomes")
    assert captured["query"] == "recent work on RDD with distribution outcomes"
    assert "Findings" in reply


def test_all_tools_registered_on_mcp() -> None:
    """Every public tool function should be exposed by the MCP server."""
    expected = {
        "ping",
        "start_chat",
        "ask",
        "list_sessions",
        "end_session",
        "search_literature",
    }
    # FastMCP exposes registered tools via list_tools(). The exact accessor
    # may differ across mcp SDK versions; try the documented async API first
    # and fall back to the internal registry if needed.
    try:
        tools = asyncio.run(mcp.list_tools())
        names = {t.name for t in tools}
    except AttributeError:
        # Fallback for older/newer SDK shapes: dig into the tool manager.
        names = set(mcp._tool_manager._tools.keys())  # type: ignore[attr-defined]
    assert expected.issubset(names), f"missing tools: {expected - names}"


# ---------------------------------------------------------------------------
# fetch_paper tool
# ---------------------------------------------------------------------------

def test_fetch_paper_tool_registered():
    from coarse.mcp_server import fetch_paper
    assert callable(fetch_paper)


def test_fetch_paper_tool_delegates_to_fetch_module(tmp_path):
    from coarse.mcp_server import fetch_paper
    with patch("coarse.mcp_server._fetch_paper") as mock_fetch:
        mock_fetch.return_value = {
            "path": str(tmp_path / "paper.pdf"),
            "title": "Test",
            "authors": [],
            "year": 2013,
            "doi": "10.x/y",
            "fetch_source": "unpaywall",
        }
        result = fetch_paper(doi="10.x/y", output_dir=str(tmp_path))
    mock_fetch.assert_called_once_with(doi="10.x/y", title=None, output_dir=str(tmp_path))
    assert result["fetch_source"] == "unpaywall"


# ---------------------------------------------------------------------------
# extract_paper tool
# ---------------------------------------------------------------------------

def test_extract_paper_tool_registered():
    from coarse.mcp_server import extract_paper
    assert callable(extract_paper)


def test_extract_paper_raises_if_file_missing(tmp_path):
    from coarse.mcp_server import extract_paper
    with pytest.raises(FileNotFoundError):
        extract_paper(paper_path=str(tmp_path / "missing.pdf"))


def test_extract_paper_writes_markdown_and_returns_path(tmp_path):
    from coarse.mcp_server import extract_paper
    from coarse.types import PaperText
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF fake")

    mock_result = PaperText(full_markdown="# Title\n\nContent.", token_estimate=100, garble_ratio=0.01)
    with patch("coarse.mcp_server.extract_file", return_value=mock_result):
        result = extract_paper(paper_path=str(fake_pdf))

    assert result["garble_ratio"] == pytest.approx(0.01)
    md_path = Path(result["path"])
    assert md_path.exists()
    assert md_path.read_text() == "# Title\n\nContent."
    assert md_path.suffix == ".md"


def test_extract_paper_respects_output_path(tmp_path):
    from coarse.mcp_server import extract_paper
    from coarse.types import PaperText
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF fake")
    custom_out = tmp_path / "custom_output.md"

    mock_result = PaperText(full_markdown="Content.", token_estimate=50, garble_ratio=0.0)
    with patch("coarse.mcp_server.extract_file", return_value=mock_result):
        result = extract_paper(
            paper_path=str(fake_pdf),
            output_path=str(custom_out),
        )

    assert Path(result["path"]) == custom_out.resolve()
    assert custom_out.read_text() == "Content."


# ---------------------------------------------------------------------------
# verify_citation tool
# ---------------------------------------------------------------------------

def test_verify_citation_tool_registered():
    from coarse.mcp_server import verify_citation
    assert callable(verify_citation)


def test_verify_citation_raises_if_markdown_missing(tmp_path):
    from coarse.mcp_server import verify_citation
    with pytest.raises(FileNotFoundError):
        verify_citation(
            paper_markdown_path=str(tmp_path / "missing.md"),
            citation_claim="Some claim.",
        )


def test_verify_citation_returns_verdict_dict(tmp_path):
    from coarse.mcp_server import verify_citation
    fake_md = tmp_path / "paper.md"
    fake_md.write_text("# Paper\n\nContent.", encoding="utf-8")

    mock_verdict = {
        "verdict": "Mismatch",
        "paper_summary": "Paper argues X.",
        "adversarial_notes": "Claim overstates scope.",
        "sop_revision_needed": True,
    }
    with patch("coarse.mcp_server.CitationVerifyAgent") as MockAgent:
        MockAgent.return_value.run.return_value = mock_verdict
        result = verify_citation(
            paper_markdown_path=str(fake_md),
            citation_claim="Paper proves universal Y.",
        )

    assert result["verdict"] == "Mismatch"
    assert result["sop_revision_needed"] is True
