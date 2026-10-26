"""Evidence-grounded self-correction of unreliable claims."""

from app.schemas.verification import ClaimChange, ClaimResult, ClaimStatus, EvidenceRef
from app.services.llm_provider import CorrectionIssue, LLMProvider
from app.services.text_utils import leading_subject, similarity

ISSUE_STATUSES = {ClaimStatus.CONTRADICTED, ClaimStatus.UNVERIFIABLE}


def _is_head(c: ClaimResult, text: str) -> bool:
    return c.text.lower().removeprefix("the ").startswith(text.lower().removeprefix("the "))


def _value_entities(c: ClaimResult) -> list[dict]:
    """Non-subject entities (place, date, person ...): the parts of a claim that can be right or wrong."""
    return [e.model_dump() for e in c.entities if not _is_head(c, e.text)]


def _subject_terms(c: ClaimResult) -> list[str]:
    terms = [e.text for e in c.entities if _is_head(c, e.text)]
    lead = leading_subject(c.text)
    return terms + ([lead] if lead and lead not in terms else [])


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
            suspect_terms=[e["text"] for e in _value_entities(c)],
            subject_terms=_subject_terms(c),
            value_entities=_value_entities(c),
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


def build_changes(original: list[ClaimResult], corrected: list[ClaimResult]) -> list[ClaimChange]:
    """Pair changed original claims with their rewritten versions and cite the re-verification evidence."""
    removed = [o for o in original if o.changed_in_correction]
    used: set[int] = set()
    changes: list[ClaimChange] = []
    for c in corrected:
        if not c.changed_in_correction:
            continue
        candidates = [o for o in removed if id(o) not in used]
        match = max(candidates, key=lambda o: similarity(o.text, c.text), default=None)
        if match is not None and similarity(match.text, c.text) < 0.2:
            match = None
        if match is not None:
            used.add(id(match))
        sources = [EvidenceRef(source_name=e.source_name, title=e.title, url=e.url)
                   for e in c.evidence if e.label == c.status and c.status in (ClaimStatus.SUPPORTED,
                                                                              ClaimStatus.CONTRADICTED)][:3]
        changes.append(ClaimChange(original_text=match.text if match else None,
                                   original_status=match.status if match else None,
                                   corrected_text=c.text, corrected_status=c.status, sources=sources))
    changes += [ClaimChange(original_text=o.text, original_status=o.status) for o in removed if id(o) not in used]
    return changes
