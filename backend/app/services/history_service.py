"""Persistence and read-models for verification runs."""

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models.verification import ClaimRecord, EvidenceRecord, VerificationRun
from app.schemas.verification import (
    ClaimResult,
    ClaimStatus,
    CorrectionInfo,
    CorrectionStatus,
    Entity,
    Evidence,
    HistoryItem,
    InputMode,
    NLIScores,
    ProviderInfo,
    RunResponse,
    ScoreBreakdown,
    StatsResponse,
    StatusCounts,
    VerificationResult,
)


def _claim_records(claims: list[ClaimResult], phase: str) -> list[ClaimRecord]:
    out = []
    for c in claims:
        rec = ClaimRecord(
            phase=phase, position=c.index, text=c.text, source_sentence=c.source_sentence, status=c.status.value,
            confidence=c.confidence, entailment=c.nli.entailment if c.nli else None,
            contradiction=c.nli.contradiction if c.nli else None, neutral=c.nli.neutral if c.nli else None,
            explanation=c.explanation, entities=[e.model_dump() for e in c.entities],
            conflicting_evidence=c.conflicting_evidence, changed_in_correction=c.changed_in_correction,
        )
        rec.evidence = [
            EvidenceRecord(
                position=i, source_name=e.source_name, title=e.title, url=e.url, snippet=e.snippet,
                retrieved_at=e.retrieved_at, relevance=e.relevance,
                entailment=e.nli.entailment if e.nli else None, contradiction=e.nli.contradiction if e.nli else None,
                neutral=e.nli.neutral if e.nli else None, label=e.label.value if e.label else None,
            )
            for i, e in enumerate(c.evidence)
        ]
        out.append(rec)
    return out


def _apply(run: VerificationRun, final_answer: str, original: VerificationResult,
           corrected: VerificationResult | None, correction: CorrectionInfo, processing_ms: float,
           warnings: list[str]) -> None:
    final = corrected or original
    run.final_answer = final_answer
    run.reliability_score = original.score.score
    run.final_reliability_score = final.score.score
    run.score_label = final.score.label
    run.original_score = original.score.model_dump()
    run.corrected_score = corrected.score.model_dump() if corrected else None
    run.correction = correction.model_dump(mode="json")
    run.correction_status = correction.status.value
    run.processing_time_ms = processing_ms
    run.original_latency_ms = original.latency_ms
    run.corrected_latency_ms = corrected.latency_ms if corrected else None
    run.warnings = warnings
    run.original_warnings = original.warnings
    run.corrected_warnings = corrected.warnings if corrected else None
    run.claims = _claim_records(original.claims, "original") + (
        _claim_records(corrected.claims, "corrected") if corrected else []
    )


def save_run(db: Session, *, mode: InputMode, question: str | None, provider: ProviderInfo | None,
             original_answer: str, final_answer: str, original: VerificationResult,
             corrected: VerificationResult | None, correction: CorrectionInfo, processing_ms: float,
             warnings: list[str]) -> VerificationRun:
    run = VerificationRun(
        mode=mode.value, question=question, original_answer=original_answer,
        provider_name=provider.name if provider else None, provider_model=provider.model if provider else None,
        provider_is_demo=provider.is_demo if provider else False,
    )
    _apply(run, final_answer, original, corrected, correction, processing_ms, warnings)
    try:
        db.add(run)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return get_run(db, run.id)


def update_run_correction(db: Session, run: VerificationRun, final_answer: str, original: VerificationResult,
                          corrected: VerificationResult | None, correction: CorrectionInfo,
                          processing_ms: float, warnings: list[str]) -> VerificationRun:
    try:
        _apply(run, final_answer, original, corrected, correction, processing_ms, warnings)
        db.commit()
    except Exception:
        db.rollback()
        raise
    return get_run(db, run.id)


def get_run(db: Session, run_id: int) -> VerificationRun | None:
    db.expire_all()
    return db.execute(
        select(VerificationRun).where(VerificationRun.id == run_id)
        .options(selectinload(VerificationRun.claims).selectinload(ClaimRecord.evidence))
    ).scalar_one_or_none()


def _claims_from(records: list[ClaimRecord], phase: str) -> list[ClaimResult]:
    out = []
    for r in records:
        if r.phase != phase:
            continue
        nli = NLIScores(entailment=r.entailment, contradiction=r.contradiction, neutral=r.neutral) \
            if r.entailment is not None else None
        out.append(ClaimResult(
            index=r.position, text=r.text, source_sentence=r.source_sentence,
            entities=[Entity(**e) for e in r.entities or []], status=ClaimStatus(r.status),
            confidence=r.confidence, nli=nli, explanation=r.explanation,
            conflicting_evidence=r.conflicting_evidence, changed_in_correction=r.changed_in_correction,
            evidence=[
                Evidence(source_name=e.source_name, title=e.title, url=e.url, snippet=e.snippet,
                         retrieved_at=e.retrieved_at, relevance=e.relevance,
                         nli=NLIScores(entailment=e.entailment, contradiction=e.contradiction, neutral=e.neutral)
                         if e.entailment is not None else None,
                         label=ClaimStatus(e.label) if e.label else None)
                for e in r.evidence
            ],
        ))
    return out


def original_result(run: VerificationRun) -> VerificationResult:
    return VerificationResult(claims=_claims_from(run.claims, "original"),
                              score=ScoreBreakdown(**run.original_score), warnings=run.original_warnings or [],
                              latency_ms=run.original_latency_ms or 0.0)


def to_response(run: VerificationRun) -> RunResponse:
    corrected = None
    if run.corrected_score is not None:
        corrected = VerificationResult(claims=_claims_from(run.claims, "corrected"),
                                       score=ScoreBreakdown(**run.corrected_score),
                                       warnings=run.corrected_warnings or [],
                                       latency_ms=run.corrected_latency_ms or 0.0)
    provider = ProviderInfo(name=run.provider_name, model=run.provider_model, is_demo=run.provider_is_demo) \
        if run.provider_name else None
    return RunResponse(
        id=run.id, created_at=run.created_at, mode=InputMode(run.mode), question=run.question, provider=provider,
        original_answer=run.original_answer, final_answer=run.final_answer,
        original_verification=original_result(run), corrected_verification=corrected,
        correction=CorrectionInfo(**run.correction), processing_time_ms=run.processing_time_ms,
        warnings=run.warnings or [],
    )


def _summary(run: VerificationRun) -> str:
    text = (run.question or run.original_answer).strip().replace("\n", " ")
    return text if len(text) <= 120 else text[:117] + "..."


def to_history_item(run: VerificationRun) -> HistoryItem:
    score = run.corrected_score or run.original_score
    return HistoryItem(
        id=run.id, created_at=run.created_at, mode=InputMode(run.mode), query_summary=_summary(run),
        reliability_score=run.reliability_score, final_reliability_score=run.final_reliability_score,
        score_label=run.score_label, counts=StatusCounts(**score["counts"]),
        correction_status=CorrectionStatus(run.correction_status), processing_time_ms=run.processing_time_ms,
        provider=run.provider_name, is_demo=run.provider_is_demo,
    )


def list_runs(db: Session, limit: int = 50, offset: int = 0) -> tuple[list[HistoryItem], int]:
    total = db.scalar(select(func.count(VerificationRun.id))) or 0
    runs = db.execute(
        select(VerificationRun).order_by(VerificationRun.created_at.desc(), VerificationRun.id.desc())
        .limit(limit).offset(offset)
    ).scalars().all()
    return [to_history_item(r) for r in runs], total


def delete_run(db: Session, run_id: int) -> bool:
    run = db.get(VerificationRun, run_id)
    if run is None:
        return False
    db.delete(run)
    db.commit()
    return True


def clear_runs(db: Session) -> int:
    runs = db.execute(select(VerificationRun)).scalars().all()
    for r in runs:
        db.delete(r)
    db.commit()
    return len(runs)


def stats(db: Session) -> StatsResponse:
    """Dashboard statistics computed from stored runs (final verification phase of each run)."""
    runs = db.execute(select(VerificationRun)).scalars().all()
    counts = StatusCounts()
    for r in runs:
        c = (r.corrected_score or r.original_score)["counts"]
        counts.supported += c["supported"]
        counts.contradicted += c["contradicted"]
        counts.unverifiable += c["unverifiable"]
        counts.unavailable += c.get("unavailable", 0)
        counts.total += c["total"]
    pct = {k: (round(100 * getattr(counts, k) / counts.total, 1) if counts.total else 0.0)
           for k in ("supported", "contradicted", "unverifiable", "unavailable")}
    scored = [r.final_reliability_score for r in runs if r.final_reliability_score is not None]
    recent, _ = list_runs(db, limit=10)
    return StatsResponse(
        total_runs=len(runs), total_claims=counts.total, counts=counts, percentages=pct,
        average_latency_ms=round(sum(r.processing_time_ms for r in runs) / len(runs), 1) if runs else None,
        average_reliability=round(sum(scored) / len(scored), 1) if scored else None,
        corrections_attempted=sum(r.correction_status not in ("not_requested",) for r in runs),
        recent=recent,
    )
