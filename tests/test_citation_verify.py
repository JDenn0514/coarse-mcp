"""Tests for CitationVerifyAgent."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from tests.conftest import make_mock_client


def _make_agent():
    from coarse.agents.citation_verify import CitationVerifyAgent
    return CitationVerifyAgent(client=make_mock_client())


def test_citation_verify_agent_returns_verdict_keys():
    from coarse.agents.citation_verify import CitationVerifyAgent, _Phase1Summary, _Phase2Verdict

    client = make_mock_client()
    client.complete.side_effect = [
        _Phase1Summary(summary="The paper argues X using method Y."),
        _Phase2Verdict(
            verdict="Supports",
            adversarial_notes="No issues found.",
            sop_revision_needed=False,
        ),
    ]
    agent = CitationVerifyAgent(client=client)
    result = agent.run(
        paper_text="Full paper text here.",
        citation_claim="This paper proves X.",
    )
    assert set(result.keys()) == {
        "verdict", "paper_summary", "adversarial_notes", "sop_revision_needed"
    }


def test_citation_verify_calls_llm_twice():
    from coarse.agents.citation_verify import CitationVerifyAgent, _Phase1Summary, _Phase2Verdict

    client = make_mock_client()
    client.complete.side_effect = [
        _Phase1Summary(summary="Paper argues X."),
        _Phase2Verdict(verdict="Mismatch", adversarial_notes="Doesn't support.", sop_revision_needed=True),
    ]
    agent = CitationVerifyAgent(client=client)
    agent.run(paper_text="Paper text.", citation_claim="Paper proves Y.")
    assert client.complete.call_count == 2


def test_citation_verify_phase1_does_not_receive_citation_claim():
    """Phase 1 prompt must not contain the citation_claim to avoid anchoring."""
    from coarse.agents.citation_verify import CitationVerifyAgent, _Phase1Summary, _Phase2Verdict

    client = make_mock_client()
    client.complete.side_effect = [
        _Phase1Summary(summary="Summary."),
        _Phase2Verdict(verdict="Weak", adversarial_notes="Partial.", sop_revision_needed=False),
    ]
    agent = CitationVerifyAgent(client=client)
    agent.run(paper_text="Paper text.", citation_claim="SECRET CLAIM XYZ")

    # First complete() call is Phase 1 — its messages must not contain the claim
    phase1_call_args = client.complete.call_args_list[0]
    messages = phase1_call_args[0][0]  # first positional arg
    all_content = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"])
        for m in messages
    )
    assert "SECRET CLAIM XYZ" not in all_content


def test_citation_verify_phase2_receives_both_summary_and_claim():
    """Phase 2 must receive both the Phase 1 summary and the citation claim."""
    from coarse.agents.citation_verify import CitationVerifyAgent, _Phase1Summary, _Phase2Verdict

    client = make_mock_client()
    client.complete.side_effect = [
        _Phase1Summary(summary="THE PHASE ONE SUMMARY"),
        _Phase2Verdict(verdict="Supports", adversarial_notes="OK.", sop_revision_needed=False),
    ]
    agent = CitationVerifyAgent(client=client)
    agent.run(paper_text="Paper text.", citation_claim="THE CITATION CLAIM")

    phase2_call_args = client.complete.call_args_list[1]
    messages = phase2_call_args[0][0]
    all_content = " ".join(
        m["content"] if isinstance(m["content"], str)
        else " ".join(b.get("text", "") for b in m["content"])
        for m in messages
    )
    assert "THE PHASE ONE SUMMARY" in all_content
    assert "THE CITATION CLAIM" in all_content


def test_citation_verify_truncates_long_paper():
    """Papers over 400k chars must be truncated before Phase 1."""
    from coarse.agents.citation_verify import CitationVerifyAgent, _Phase1Summary, _Phase2Verdict, _MAX_PAPER_CHARS

    client = make_mock_client()
    client.complete.side_effect = [
        _Phase1Summary(summary="Summary."),
        _Phase2Verdict(verdict="Supports", adversarial_notes="Fine.", sop_revision_needed=False),
    ]
    agent = CitationVerifyAgent(client=client)
    long_paper = "x" * (_MAX_PAPER_CHARS + 50_000)
    agent.run(paper_text=long_paper, citation_claim="Claim.")

    phase1_user_content = client.complete.call_args_list[0][0][0][1]["content"]
    if isinstance(phase1_user_content, list):
        text = " ".join(b.get("text", "") for b in phase1_user_content)
    else:
        text = phase1_user_content
    assert len(text) <= _MAX_PAPER_CHARS + 200  # allow for prompt prefix
