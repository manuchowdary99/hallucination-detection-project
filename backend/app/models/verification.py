"""ORM models for verification runs, claims and evidence."""

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


class VerificationRun(Base):
    __tablename__ = "verification_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    mode: Mapped[str] = mapped_column(String(16))
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_answer: Mapped[str] = mapped_column(Text)
    final_answer: Mapped[str] = mapped_column(Text)
    provider_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    provider_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider_is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_reliability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    score_label: Mapped[str] = mapped_column(String(32))
    original_score: Mapped[dict] = mapped_column(JSON)
    corrected_score: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    correction: Mapped[dict] = mapped_column(JSON)
    correction_status: Mapped[str] = mapped_column(String(32))
    processing_time_ms: Mapped[float] = mapped_column(Float)
    original_latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    corrected_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    original_warnings: Mapped[list] = mapped_column(JSON, default=list)
    corrected_warnings: Mapped[list | None] = mapped_column(JSON, nullable=True)

    claims: Mapped[list["ClaimRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="ClaimRecord.position", passive_deletes=True
    )


class ClaimRecord(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("verification_runs.id", ondelete="CASCADE"), index=True)
    phase: Mapped[str] = mapped_column(String(16))  # "original" | "corrected"
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    source_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    entailment: Mapped[float | None] = mapped_column(Float, nullable=True)
    contradiction: Mapped[float | None] = mapped_column(Float, nullable=True)
    neutral: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[str] = mapped_column(Text)
    entities: Mapped[list] = mapped_column(JSON, default=list)
    conflicting_evidence: Mapped[bool] = mapped_column(Boolean, default=False)
    changed_in_correction: Mapped[bool] = mapped_column(Boolean, default=False)

    run: Mapped[VerificationRun] = relationship(back_populates="claims")
    evidence: Mapped[list["EvidenceRecord"]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", order_by="EvidenceRecord.position", passive_deletes=True
    )


class EvidenceRecord(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    claim_id: Mapped[int] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    source_name: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(512))
    url: Mapped[str] = mapped_column(String(1024))
    snippet: Mapped[str] = mapped_column(Text)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    relevance: Mapped[float] = mapped_column(Float, default=0.0)
    entailment: Mapped[float | None] = mapped_column(Float, nullable=True)
    contradiction: Mapped[float | None] = mapped_column(Float, nullable=True)
    neutral: Mapped[float | None] = mapped_column(Float, nullable=True)
    label: Mapped[str | None] = mapped_column(String(32), nullable=True)

    claim: Mapped[ClaimRecord] = relationship(back_populates="evidence")
