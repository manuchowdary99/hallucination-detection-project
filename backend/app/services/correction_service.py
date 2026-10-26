"""Evidence-grounded self-correction of unreliable claims."""

from app.schemas.verification import ClaimResult, ClaimStatus
from app.services.llm_provider import CorrectionIssue, LLMProvider
from app.services.text_utils import similarity

ISSUE_STATUSES = {ClaimStatus.CONTRADICTED, ClaimStatus.UNVERIFIABLE}


def _suspect_terms(c: ClaimResult) -> list[str]:
    head = c.text.lower().removeprefix("the ")
    return [e.text for e in c.entities if not head.startswith(e.text.lower().removeprefix("the "))]


def collect_issues(claims: list[ClaimResult]) -> list[CorrectionIssue]:
    return [
        CorrectionIssue(
            claim=c.text,
            status=c.status.value,
            source_sentence=c.source_sentence,
            evidence=[
                {"source_name": e.source_name, "title": e.title, "url": e.url, "snippet": e.snippet,
                 "label": e.label.value if e.label else None}
                for e in c.evidence
            ],
            suspect_terms=_suspect_terms(c),
        )
        for c in claims
        if c.status in ISSUE_STATUSES
    ]


async def correct_answer(provider: LLMProvider, answer: str, claims: list[ClaimResult]) -> tuple[str, int]:
    """Ask the provider to revise only problematic claims. Returns (revised_answer, number_of_issues)."""
    issues = collect_issues(claims)
    if not issues:
        return answer, 0
    revised = await provider.correct(answer, issues)
    return revised.strip(), len(issues)


def mark_changes(original: list[ClaimResult], corrected: list[ClaimResult], threshold: float = 0.85) -> int:
    """Flag claims that differ between the original and corrected answers. Returns #changed corrected claims."""
    changed = 0
    for c in corrected:
        if max((similarity(c.text, o.text) for o in original), default=0.0) < threshold:
            c.changed_in_correction = True
            changed += 1
    for o in original:
        if max((similarity(o.text, c.text) for c in corrected), default=0.0) < threshold:
            o.changed_in_correction = True
    return changed
