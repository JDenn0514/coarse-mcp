"""Citation verify agent — two-phase adversarial citation accuracy check.

Phase 1: independent read of the paper with no citation context.
Phase 2: adversarial comparison of the Phase 1 summary against the citation claim.

Agents must not import from: pipeline, cli, synthesis, extraction, extraction_qa.
Paper text is provided as a pre-extracted string by the caller.
"""
from __future__ import annotations

from pydantic import BaseModel

from coarse.agents.base import ReviewAgent
from coarse.prompts import CITATION_VERIFY_PHASE1_SYSTEM, CITATION_VERIFY_PHASE2_SYSTEM

_TEMPERATURE = 0.1
_MAX_PAPER_CHARS = 400_000


class _Phase1Summary(BaseModel):
    summary: str


class _Phase2Verdict(BaseModel):
    verdict: str
    adversarial_notes: str
    sop_revision_needed: bool


class CitationVerifyAgent(ReviewAgent):
    """Adversarial two-phase citation check.

    Phase 1 reads the paper cold (no citation context) and produces a neutral
    summary. Phase 2 receives that summary plus the citation claim and is
    instructed to find reasons the paper does not support the claim.
    """

    def run(  # type: ignore[override]
        self,
        paper_text: str,
        citation_claim: str,
    ) -> dict:
        truncated = paper_text[:_MAX_PAPER_CHARS]

        p1_messages = self._build_messages(
            CITATION_VERIFY_PHASE1_SYSTEM,
            f"Paper content:\n\n{truncated}",
        )
        p1: _Phase1Summary = self.client.complete(
            p1_messages,
            _Phase1Summary,
            max_tokens=2048,
            temperature=_TEMPERATURE,
        )

        p2_user = (
            f"## Paper summary (independent read)\n\n{p1.summary}\n\n"
            f"## Cited as\n\n{citation_claim}"
        )
        p2_messages = self._build_messages(CITATION_VERIFY_PHASE2_SYSTEM, p2_user)
        p2: _Phase2Verdict = self.client.complete(
            p2_messages,
            _Phase2Verdict,
            max_tokens=1024,
            temperature=_TEMPERATURE,
        )

        return {
            "verdict": p2.verdict,
            "paper_summary": p1.summary,
            "adversarial_notes": p2.adversarial_notes,
            "sop_revision_needed": p2.sop_revision_needed,
        }
