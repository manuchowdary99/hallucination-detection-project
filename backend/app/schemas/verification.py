"""Pydantic request/response schemas for the verification API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings

_MAX = get_settings().max_input_chars


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    VERIFICATION_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"


class InputMode(str, Enum):
    VERIFY = "verify"
    GENERATE = "generate"


class CorrectionStatus(str, Enum):
    NOT_REQUESTED = "not_requested"
    NOT_NEEDED = "not_needed"
    CORRECTED = "corrected"
    NO_CHANGES = "no_changes"
    FAILED = "failed"


def _strip_nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty or whitespace")
    return value


class VerifyRequest(BaseModel):
    response_text: str = Field(..., min_length=1, max_length=_MAX, description="AI-generated text to verify")
    question: str | None = Field(default=None, max_length=_MAX)
    auto_correct: bool = False

    _v = field_validator("response_text")(_strip_nonempty)


class GenerateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=_MAX)
    provider: str | None = Field(default=None, pattern="^(demo|openai)$")
    auto_correct: bool = False

    _v = field_validator("question")(_strip_nonempty)


class CorrectRequest(BaseModel):
    run_id: int = Field(..., ge=1)
    provider: str | None = Field(default=None, pattern="^(demo|openai)$")


class NLIScores(BaseModel):
    entailment: float
    contradiction: float
    neutral: float


class Entity(BaseModel):
    text: str
    label: str


class Evidence(BaseModel):
    source_name: str
    title: str
    url: str
    snippet: str
    retrieved_at: datetime | None = None
    relevance: float = 0.0
    nli: NLIScores | None = None
    label: ClaimStatus | None = None


class ClaimResult(BaseModel):
    index: int
    text: str
    source_sentence: str | None = None
    entities: list[Entity] = []
    status: ClaimStatus
    confidence: float | None = Field(default=None, description="Model confidence (0-100) in the assigned status")
    nli: NLIScores | None = None
    explanation: str
    evidence: list[Evidence] = []
    conflicting_evidence: bool = False
    changed_in_correction: bool = False


class StatusCounts(BaseModel):
    supported: int = 0
    contradicted: int = 0
    unverifiable: int = 0
    unavailable: int = 0
    total: int = 0


class ScoreBreakdown(BaseModel):
    score: float | None = Field(default=None, description="Reliability estimate 0-100; null when insufficient")
    label: str
    counts: StatusCounts
    components: dict[str, float] = {}
    formula: str


class VerificationResult(BaseModel):
    claims: list[ClaimResult]
    score: ScoreBreakdown
    warnings: list[str] = []
    latency_ms: float = 0.0


class ProviderInfo(BaseModel):
    name: str
    model: str | None = None
    is_demo: bool


class CorrectionInfo(BaseModel):
    status: CorrectionStatus = CorrectionStatus.NOT_REQUESTED
    message: str | None = None
    provider: ProviderInfo | None = None
    changed_claims: int = 0
    score_before: float | None = None
    score_after: float | None = None


class RunResponse(BaseModel):
    id: int
    created_at: datetime
    mode: InputMode
    question: str | None
    provider: ProviderInfo | None
    original_answer: str
    final_answer: str
    original_verification: VerificationResult
    corrected_verification: VerificationResult | None = None
    correction: CorrectionInfo
    processing_time_ms: float
    warnings: list[str] = []


class HistoryItem(BaseModel):
    id: int
    created_at: datetime
    mode: InputMode
    query_summary: str
    reliability_score: float | None
    final_reliability_score: float | None
    score_label: str
    counts: StatusCounts
    correction_status: CorrectionStatus
    processing_time_ms: float
    provider: str | None
    is_demo: bool


class StatsResponse(BaseModel):
    total_runs: int
    total_claims: int
    counts: StatusCounts
    percentages: dict[str, float]
    average_latency_ms: float | None
    average_reliability: float | None
    corrections_attempted: int
    recent: list[HistoryItem]


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    llm: dict
    nli: ComponentHealth
    nlp: ComponentHealth
    knowledge_providers: list[str]
    database: ComponentHealth
