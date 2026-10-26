"""End-to-end verification pipeline: extraction → retrieval → NLI → scoring → (optional) correction."""

import asyncio
import time

from app.core.config import Settings, get_settings
from app.schemas.verification import (
    ClaimResult,
    ClaimStatus,
    CorrectionInfo,
    CorrectionStatus,
    Entity,
    Evidence,
    ProviderInfo,
    VerificationResult,
)
from app.services.claim_extractor import ClaimExtractor, ExtractedClaim, get_claim_extractor
from app.services.confidence_scorer import score_claims
from app.services.correction_service import correct_answer, mark_changes
from app.services.knowledge_retriever import (
    EvidencePassage,
    KnowledgeRetriever,
    RetrievalResult,
    get_knowledge_retriever,
)
from app.services.llm_provider import LLMProvider, LLMProviderError
from app.services.nli_verifier import NLIUnavailable, NLIVerifier, PassageJudgement, aggregate, get_nli_verifier


def provider_info(p: LLMProvider) -> ProviderInfo:
    return ProviderInfo(name=p.name, model=p.model, is_demo=p.is_demo)


def _premise(p: EvidencePassage) -> str:
    return p.snippet if p.snippet.startswith(p.title) else f"{p.title}: {p.snippet}"


class VerificationPipeline:
    def __init__(self, extractor: ClaimExtractor | None = None, retriever: KnowledgeRetriever | None = None,
                 nli: NLIVerifier | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.extractor = extractor or get_claim_extractor()
        self.retriever = retriever or get_knowledge_retriever()
        self.nli = nli or get_nli_verifier()

    async def verify_text(self, text: str) -> VerificationResult:
        start = time.perf_counter()
        extracted, warnings = await asyncio.to_thread(self.extractor.extract, text)
        if not extracted:
            warnings.append("No verifiable factual claims were detected in the response.")
            return VerificationResult(claims=[], score=score_claims([]), warnings=warnings,
                                      latency_ms=round((time.perf_counter() - start) * 1000, 1))

        retrievals: list[RetrievalResult] = await asyncio.gather(
            *(self.retriever.retrieve(c.text, c.search_terms) for c in extracted)
        )
        for r in retrievals:
            for w in r.warnings:
                if w not in warnings:
                    warnings.append(w)

        pairs = [(_premise(p), c.text) for c, r in zip(extracted, retrievals) for p in r.passages]
        nli_scores = None
        try:
            nli_scores = await asyncio.to_thread(self.nli.predict, pairs)
        except NLIUnavailable as exc:
            warnings.append(f"NLI verification unavailable: {exc}. Claims with evidence are marked "
                            "VERIFICATION_UNAVAILABLE; no heuristic verdicts were substituted.")

        claims: list[ClaimResult] = []
        cursor = 0
        for idx, (c, r) in enumerate(zip(extracted, retrievals)):
            n = len(r.passages)
            scores = nli_scores[cursor : cursor + n] if nli_scores is not None else None
            cursor += n
            claims.append(self._build_claim(idx, c, r.passages, scores))

        return VerificationResult(claims=claims, score=score_claims(claims), warnings=warnings,
                                  latency_ms=round((time.perf_counter() - start) * 1000, 1))

    def _build_claim(self, idx: int, c: ExtractedClaim, passages: list[EvidencePassage], scores) -> ClaimResult:
        base = dict(index=idx, text=c.text, source_sentence=c.sentence,
                    entities=[Entity(**e) for e in c.entities])
        evidence = [
            Evidence(source_name=p.source_name, title=p.title, url=p.url, snippet=p.snippet,
                     retrieved_at=p.retrieved_at, relevance=p.relevance)
            for p in passages
        ]
        if scores is None and passages:
            return ClaimResult(**base, status=ClaimStatus.VERIFICATION_UNAVAILABLE, evidence=evidence,
                               explanation="Evidence was retrieved but the NLI model was unavailable, so the claim "
                                           "was not classified.")
        judgements = [PassageJudgement(s, p.relevance, p.novel_relevance, p.title) for s, p in zip(scores or [], passages)]
        verdict = aggregate(judgements, self.settings)
        for ev, s, label in zip(evidence, scores or [], verdict.passage_labels):
            ev.nli, ev.label = s, label
        if verdict.best_index is not None:
            evidence.insert(0, evidence.pop(verdict.best_index))
        return ClaimResult(**base, status=verdict.status, confidence=verdict.confidence, nli=verdict.best,
                           explanation=verdict.explanation, evidence=evidence,
                           conflicting_evidence=verdict.conflicting)

    async def correct(self, provider: LLMProvider, answer: str, original: VerificationResult
                      ) -> tuple[CorrectionInfo, str, VerificationResult | None]:
        info = CorrectionInfo(provider=provider_info(provider), score_before=original.score.score)
        if not any(c.status in (ClaimStatus.CONTRADICTED, ClaimStatus.UNVERIFIABLE) for c in original.claims):
            info.status = CorrectionStatus.NOT_NEEDED
            info.message = ("No contradicted or unverifiable claims were found, so no correction was needed."
                            if original.claims else "No factual claims to correct.")
            return info, answer, None
        try:
            revised, n_issues = await correct_answer(provider, answer, original.claims)
        except LLMProviderError as exc:
            info.status, info.message = CorrectionStatus.FAILED, f"Correction failed: {exc}"
            return info, answer, None
        if " ".join(revised.split()) == " ".join(answer.split()):
            info.status = CorrectionStatus.NO_CHANGES
            info.message = (f"{n_issues} claim(s) needed attention but the provider made no changes"
                            + (" (the demo provider only rewrites contradicted claims that have quotable evidence)."
                               if provider.is_demo else "."))
            return info, answer, None
        corrected = await self.verify_text(revised)
        info.changed_claims = mark_changes(original.claims, corrected.claims)
        info.status = CorrectionStatus.CORRECTED
        info.score_after = corrected.score.score
        remaining = sum(c.status != ClaimStatus.SUPPORTED for c in corrected.claims)
        info.message = (f"Revised {n_issues} flagged claim(s) and re-verified the result"
                        + (f"; {remaining} claim(s) are still not supported." if remaining else "."))
        if provider.is_demo:
            info.message += " Demo provider: extractive correction quoting retrieved evidence."
        return info, revised, corrected


_pipeline: VerificationPipeline | None = None


def get_pipeline() -> VerificationPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = VerificationPipeline()
    return _pipeline
