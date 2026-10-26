"""Transparent reliability scoring.

Formula (documented on the Methodology page too). Over the N claims that were classified by NLI
(S supported, C contradicted, U unverifiable; claims with VERIFICATION_UNAVAILABLE are excluded):

    q_i      = 0.5 + 0.5 * relevance_i                 evidence-quality weight in [0.5, 1]
    support  = sum_{i in S} (entailment_i * q_i) / N
    partial  = 0.25 * U / N                            small credit: unverifiable is not the same as false
    penalty  = 0.5 * sum_{i in C} (contradiction_i * q_i) / N
    score    = 100 * clamp(support + partial - penalty, 0, 1)

If no claim could be classified, or none has conclusive (supported/contradicted) evidence, no score is
given and the label is "Insufficient evidence". The score is a heuristic estimate of how well the
response is backed by retrieved evidence. It is NOT a calibrated probability of factual correctness.
"""

from app.schemas.verification import ClaimResult, ClaimStatus, ScoreBreakdown, StatusCounts

FORMULA = (
    "score = 100 × clamp(Σ_supported(entailment·q)/N + 0.25·U/N − 0.5·Σ_contradicted(contradiction·q)/N, 0, 1), "
    "q = 0.5 + 0.5·evidence relevance, N = claims classified by NLI"
)


def count_statuses(claims: list[ClaimResult]) -> StatusCounts:
    c = StatusCounts(total=len(claims))
    for claim in claims:
        if claim.status == ClaimStatus.SUPPORTED:
            c.supported += 1
        elif claim.status == ClaimStatus.CONTRADICTED:
            c.contradicted += 1
        elif claim.status == ClaimStatus.UNVERIFIABLE:
            c.unverifiable += 1
        else:
            c.unavailable += 1
    return c


def _quality(claim: ClaimResult) -> float:
    labelled = [e for e in claim.evidence if e.label == claim.status]
    rel = max((e.relevance for e in labelled), default=max((e.relevance for e in claim.evidence), default=0.0))
    return 0.5 + 0.5 * min(1.0, rel)


def score_claims(claims: list[ClaimResult]) -> ScoreBreakdown:
    counts = count_statuses(claims)
    n = counts.supported + counts.contradicted + counts.unverifiable
    if counts.total == 0:
        return ScoreBreakdown(score=None, label="No factual claims", counts=counts, formula=FORMULA)
    if n == 0:
        return ScoreBreakdown(score=None, label="Verification unavailable", counts=counts, formula=FORMULA)
    if counts.supported + counts.contradicted == 0:
        return ScoreBreakdown(score=None, label="Insufficient evidence", counts=counts, formula=FORMULA,
                              components={"unverifiable_share": round(counts.unverifiable / n, 4)})
    support = sum((c.nli.entailment if c.nli else 0) * _quality(c) for c in claims if c.status == ClaimStatus.SUPPORTED) / n
    penalty = 0.5 * sum(
        (c.nli.contradiction if c.nli else 0) * _quality(c) for c in claims if c.status == ClaimStatus.CONTRADICTED
    ) / n
    partial = 0.25 * counts.unverifiable / n
    score = round(100 * max(0.0, min(1.0, support + partial - penalty)), 1)
    label = "High" if score >= 75 else "Moderate" if score >= 50 else "Low"
    return ScoreBreakdown(
        score=score, label=label, counts=counts, formula=FORMULA,
        components={"support": round(support, 4), "unverifiable_credit": round(partial, 4),
                    "contradiction_penalty": round(penalty, 4), "classified_claims": n},
    )
