"""Regression tests for evidence-grounded correction and strict re-verification."""

import asyncio

from app.schemas.verification import ClaimStatus, NLIScores
from app.services.knowledge_retriever import RetrievalResult
from app.services.llm_provider import CorrectionIssue, DemoProvider, slot_correction, split_values
from tests.conftest import FAKE_NLI_RULES, FakeNLI, FakeRetriever, passage


def run_correction(pipeline, answer, provider=None):
    async def go():
        original = await pipeline.verify_text(answer)
        info, final, corrected = await pipeline.correct(provider or DemoProvider(), answer, original)
        return original, info, final, corrected
    return asyncio.run(go())


def by_text(result):
    return {c.text: c for c in result.claims}


def issue(claim, sentence, values, snippet, label="CONTRADICTED", source="Wikipedia"):
    return CorrectionIssue(
        claim=claim, status="CONTRADICTED", source_sentence=sentence, subject_terms=["Einstein"],
        value_entities=values, suspect_terms=[v["text"] for v in values],
        evidence=[{"source_name": source, "title": "Albert Einstein", "url": "u", "snippet": snippet, "label": label}],
    )


def test_wrong_location_is_replaced_and_correct_date_kept_unit():
    i = issue("Einstein died in London in 1955.", "He died in London in 1955.",
              [{"text": "London", "label": "GPE"}, {"text": "1955", "label": "DATE"}],
              "Albert Einstein was born in Ulm in 1879. He died in Princeton, New Jersey, in 1955.")
    kept, wrong = split_values(i)
    assert [v["text"] for v in kept] == ["1955"] and [v["text"] for v in wrong] == ["London"]
    assert slot_correction(i) == "He died in Princeton in 1955."


def test_wrong_location_corrected_with_date_preserved(make_pipeline):
    answer = "Albert Einstein was born in Ulm. He died in London in 1955."
    original, info, final, corrected = run_correction(make_pipeline(), answer)
    assert by_text(original)["Albert Einstein died in London in 1955."].status == ClaimStatus.CONTRADICTED
    assert final == "Albert Einstein was born in Ulm. He died in Princeton in 1955."
    fixed = by_text(corrected)["Albert Einstein died in Princeton in 1955."]
    assert fixed.status == ClaimStatus.SUPPORTED and fixed.changed_in_correction
    assert info.unsupported_changed_claims == 0
    [change] = info.changes
    assert change.original_text == "Albert Einstein died in London in 1955."
    assert change.original_status == ClaimStatus.CONTRADICTED
    assert change.corrected_status == ClaimStatus.SUPPORTED
    assert change.sources and change.sources[0].url == "https://en.wikipedia.org/wiki/Albert_Einstein"
    assert fixed.evidence and fixed.evidence[0].label == ClaimStatus.SUPPORTED


def test_wrong_date_corrected_with_location_preserved(make_pipeline, monkeypatch):
    rules = [("1960", NLIScores(entailment=0.02, contradiction=0.95, neutral=0.03))] + FAKE_NLI_RULES
    monkeypatch.setattr("tests.conftest.FAKE_NLI_RULES", rules)
    answer = "Albert Einstein was born in Ulm. He died in Princeton in 1960."
    original, info, final, corrected = run_correction(make_pipeline(), answer)
    assert by_text(original)["Albert Einstein died in Princeton in 1960."].status == ClaimStatus.CONTRADICTED
    assert final == "Albert Einstein was born in Ulm. He died in Princeton in 1955."
    assert by_text(corrected)["Albert Einstein died in Princeton in 1955."].status == ClaimStatus.SUPPORTED


class StaticRetriever(FakeRetriever):
    def __init__(self, title, snippet, warnings=None):
        super().__init__(warnings)
        self.title, self.snippet = title, snippet

    async def retrieve(self, claim, terms):
        self.calls.append(claim)
        return RetrievalResult([passage(self.title, self.snippet, claim)], list(self.warnings))


def test_irrelevant_passage_cannot_decide_death_location(make_pipeline):
    # The NLI model scores this as a contradiction (fake rule on "london"), but the passage is about Margot
    # Einstein's correspondence, not Albert Einstein's death, so it must not make the claim CONTRADICTED.
    margot = ("Margot Einstein permitted the personal letters to be made available to the public, but requested "
              "that it not be done until twenty years after her death (she died in 1986).")
    pipeline = make_pipeline(retriever=StaticRetriever("Albert Einstein", margot))
    result = asyncio.run(pipeline.verify_text("Albert Einstein died in London in 1955."))
    [c] = result.claims
    assert c.status == ClaimStatus.UNVERIFIABLE and c.confidence is None
    assert c.evidence[0].label == ClaimStatus.UNVERIFIABLE
    assert "do not address this claim directly" in c.explanation
    assert result.score.score is None and result.score.label == "Insufficient evidence"


def test_conflicting_evidence_is_flagged_and_not_overconfident(make_pipeline, monkeypatch):
    class TwoPassages(FakeRetriever):
        async def retrieve(self, claim, terms):
            return RetrievalResult([
                passage("Albert Einstein", "Albert Einstein died in Princeton in 1955.", claim),
                passage("Albert Einstein", "Albert Einstein died in Zurich in 1955.", claim),
            ], [])

    class SplitNLI(FakeNLI):
        def predict(self, pairs):
            return [NLIScores(entailment=0.9, contradiction=0.05, neutral=0.05) if "Princeton" in p
                    else NLIScores(entailment=0.05, contradiction=0.7, neutral=0.25) for p, _ in pairs]

    pipeline = make_pipeline(retriever=TwoPassages())
    pipeline.nli = SplitNLI()
    [c] = asyncio.run(pipeline.verify_text("Albert Einstein died in Princeton in 1955.")).claims
    assert c.status == ClaimStatus.SUPPORTED and c.conflicting_evidence
    assert c.model_confidence == 90.0 and c.confidence < c.model_confidence * (1 - 0.5 * 0.7) + 0.1
    assert "review both sources" in c.explanation


def test_no_useful_evidence_is_unverifiable_and_left_alone(make_pipeline):
    answer = "Zorvathian engineers built the Kelmora Bridge in 1642."
    original, info, final, corrected = run_correction(make_pipeline(), answer)
    [c] = original.claims
    assert c.status == ClaimStatus.UNVERIFIABLE and not c.evidence and c.confidence is None
    assert original.score.score is None
    assert info.status.value == "no_changes" and final == answer and corrected is None


class FabricatingProvider(DemoProvider):
    async def correct(self, answer, issues):
        return answer.replace("He died in London in 1955.",
                              "He died in Princeton in 1955. Einstein kept a parrot named Quantum.")


def test_correction_that_adds_unsupported_claim_is_not_trusted(make_pipeline):
    answer = "Albert Einstein was born in Ulm. He died in London in 1955."
    _, info, final, corrected = run_correction(make_pipeline(), answer, FabricatingProvider())
    claims = by_text(corrected)
    parrot = next(c for t, c in claims.items() if "parrot" in t)
    assert parrot.changed_in_correction and parrot.status == ClaimStatus.UNVERIFIABLE
    assert claims["Albert Einstein died in Princeton in 1955."].status == ClaimStatus.SUPPORTED
    assert info.unsupported_changed_claims == 1
    assert "not supported by retrieved evidence" in info.message
    added = [ch for ch in info.changes if ch.corrected_text and "parrot" in ch.corrected_text]
    assert added and added[0].corrected_status == ClaimStatus.UNVERIFIABLE and not added[0].sources


class FailAfterFirstNLI(FakeNLI):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        if self.calls > 1:
            self.unavailable = True
        return super().predict(pairs)


def test_nli_failure_during_reverification_never_marks_correction_supported(make_pipeline):
    pipeline = make_pipeline()
    pipeline.nli = FailAfterFirstNLI()
    _, info, final, corrected = run_correction(pipeline, "Albert Einstein was born in Ulm. He died in London in 1955.")
    assert "Princeton" in final
    assert all(c.status == ClaimStatus.VERIFICATION_UNAVAILABLE for c in corrected.claims)
    assert corrected.score.score is None and info.score_after is None
    assert info.unsupported_changed_claims >= 1
    assert any("NLI verification unavailable" in w for w in corrected.warnings)


class FailAfterFirstRetrieval(FakeRetriever):
    def __init__(self):
        super().__init__()
        self.failed = False

    async def retrieve(self, claim, terms):
        if self.failed:
            return RetrievalResult([], ["Wikipedia: request timed out after retries"])
        return await super().retrieve(claim, terms)


def test_knowledge_source_failure_during_reverification(make_pipeline):
    retriever = FailAfterFirstRetrieval()
    pipeline = make_pipeline(retriever=retriever)
    answer = "Albert Einstein was born in Ulm. He died in London in 1955."

    async def go():
        original = await pipeline.verify_text(answer)
        retriever.failed = True
        return await pipeline.correct(DemoProvider(), answer, original)

    info, final, corrected = asyncio.run(go())
    assert all(c.status == ClaimStatus.UNVERIFIABLE for c in corrected.claims)
    assert corrected.score.label == "Insufficient evidence"
    assert info.unsupported_changed_claims >= 1
    assert "Wikipedia: request timed out after retries" in corrected.warnings
