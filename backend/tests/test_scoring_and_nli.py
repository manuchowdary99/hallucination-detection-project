from app.core.config import get_settings
from app.schemas.verification import ClaimResult, ClaimStatus, Evidence, NLIScores
from app.services.confidence_scorer import score_claims
from app.services.nli_verifier import PassageJudgement, aggregate, evidence_confidence


def j(e, c, rel=0.8, novel=0.8, title="T", direct=True):
    return PassageJudgement(NLIScores(entailment=e, contradiction=c, neutral=round(1 - e - c, 4)), rel, novel, title,
                            direct)


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
    p = e if status == ClaimStatus.SUPPORTED else c if status == ClaimStatus.CONTRADICTED else None
    conf = evidence_confidence(p, rel) if p is not None else None
    return ClaimResult(index=0, text="x", status=status, confidence=conf, nli=nli, explanation="", evidence=ev)


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


def test_weak_evidence_lowers_confidence():
    strong, weak = aggregate([j(0.95, 0.01, rel=1.0)]), aggregate([j(0.95, 0.01, rel=0.2)])
    assert strong.status == weak.status == ClaimStatus.SUPPORTED
    assert strong.confidence == 95.0 and weak.confidence == 57.0  # 95 x (0.5 + 0.5 x 0.2)
    s_strong = score_claims([claim(ClaimStatus.SUPPORTED, e=0.95, rel=1.0)])
    s_weak = score_claims([claim(ClaimStatus.SUPPORTED, e=0.95, rel=0.2)])
    assert s_weak.score < s_strong.score and s_weak.label != "High"


def test_contradictory_evidence_lowers_confidence():
    clean = aggregate([j(0.9, 0.05, rel=1.0)])
    conflicted = aggregate([j(0.9, 0.05, rel=1.0), j(0.05, 0.7, rel=1.0)])
    assert conflicted.status == ClaimStatus.SUPPORTED and conflicted.conflicting
    assert conflicted.confidence == round(90 * (1 - 0.5 * 0.7), 1) < clean.confidence
    tied = aggregate([j(0.70, 0.05), j(0.05, 0.80)])
    assert tied.status == ClaimStatus.UNVERIFIABLE and tied.conflicting and tied.confidence is None


def test_indirect_contradiction_is_ignored():
    v = aggregate([j(0.0, 0.99, rel=0.71, novel=0.6, direct=False)])
    assert v.status == ClaimStatus.UNVERIFIABLE and v.confidence is None
    assert "do not address this claim directly" in v.explanation


def test_decisive_passage_prefers_relevant_evidence():
    v = aggregate([j(0.0, 0.996, rel=0.5), j(0.0, 0.990, rel=0.9)])
    assert v.status == ClaimStatus.CONTRADICTED and v.best_index == 1


def test_unverifiable_and_insufficient_evidence_have_no_confidence():
    assert aggregate([j(0.2, 0.2)]).confidence is None
    s = score_claims([claim(ClaimStatus.SUPPORTED, e=0.9), claim(ClaimStatus.UNVERIFIABLE),
                      claim(ClaimStatus.UNVERIFIABLE), claim(ClaimStatus.UNVERIFIABLE)])
    assert s.score == round(100 * (0.9 / 4 + 0.25 * 3 / 4), 1)
