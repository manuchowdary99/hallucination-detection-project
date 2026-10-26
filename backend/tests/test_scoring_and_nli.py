from app.core.config import get_settings
from app.schemas.verification import ClaimResult, ClaimStatus, Evidence, NLIScores
from app.services.confidence_scorer import score_claims
from app.services.nli_verifier import PassageJudgement, aggregate


def j(e, c, rel=0.8, novel=0.8, title="T"):
    return PassageJudgement(NLIScores(entailment=e, contradiction=c, neutral=round(1 - e - c, 4)), rel, novel, title)


def test_no_evidence_is_unverifiable_not_contradicted():
    v = aggregate([])
    assert v.status == ClaimStatus.UNVERIFIABLE and v.confidence is None
    assert "does not mean it is false" in v.explanation


def test_supported_and_contradicted():
    assert aggregate([j(0.95, 0.01)]).status == ClaimStatus.SUPPORTED
    assert aggregate([j(0.02, 0.9)]).status == ClaimStatus.CONTRADICTED
    assert aggregate([j(0.3, 0.3)]).status == ClaimStatus.UNVERIFIABLE


def test_irrelevant_contradiction_is_ignored():
    # contradiction from a passage that shares nothing with the claim beyond the article subject
    v = aggregate([j(0.01, 0.95, rel=0.4, novel=0.0)])
    assert v.status == ClaimStatus.UNVERIFIABLE


def test_conflicting_evidence_is_flagged():
    v = aggregate([j(0.9, 0.05), j(0.05, 0.85)])
    assert v.status == ClaimStatus.SUPPORTED and v.conflicting
    v = aggregate([j(0.65, 0.05), j(0.05, 0.95)])
    assert v.status == ClaimStatus.CONTRADICTED and v.conflicting
    v = aggregate([j(0.70, 0.05), j(0.05, 0.80)])
    assert v.status == ClaimStatus.UNVERIFIABLE and v.conflicting


def claim(status, e=0.0, c=0.0, rel=1.0):
    nli = NLIScores(entailment=e, contradiction=c, neutral=0.0)
    ev = [Evidence(source_name="W", title="t", url="u", snippet="s", relevance=rel, nli=nli, label=status)]
    return ClaimResult(index=0, text="x", status=status, nli=nli, explanation="", evidence=ev)


def test_score_formula():
    s = score_claims([claim(ClaimStatus.SUPPORTED, e=1.0), claim(ClaimStatus.SUPPORTED, e=1.0)])
    assert s.score == 100.0 and s.label == "High"
    s = score_claims([claim(ClaimStatus.SUPPORTED, e=1.0), claim(ClaimStatus.CONTRADICTED, c=1.0)])
    assert s.score == 25.0 and s.label == "Low"  # 100*(0.5 - 0.5*0.5)
    s = score_claims([claim(ClaimStatus.SUPPORTED, e=1.0), claim(ClaimStatus.UNVERIFIABLE)])
    assert s.score == 62.5  # 100*(0.5 + 0.25*0.5)


def test_insufficient_evidence_has_no_score():
    s = score_claims([claim(ClaimStatus.UNVERIFIABLE), claim(ClaimStatus.UNVERIFIABLE)])
    assert s.score is None and s.label == "Insufficient evidence"
    assert score_claims([]).label == "No factual claims"
    assert score_claims([claim(ClaimStatus.VERIFICATION_UNAVAILABLE)]).label == "Verification unavailable"


def test_thresholds_configured():
    s = get_settings()
    assert 0 < s.nli_entail_threshold < 1 and 0 < s.nli_contradict_threshold < 1
