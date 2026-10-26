"""Transparent reliability scoring.

Formula (documented on the Methodology page too). Over the N claims that were classified by NLI
(S supported, C contradicted, U unverifiable; claims with VERIFICATION_UNAVAILABLE are excluded):

    a_i      = p_i * q_i * (1 - 0.5 * o_i)             claim confidence (ClaimResult.confidence / 100)
               p_i = NLI probability of the assigned label from the decisive passage
               q_i = 0.5 + 0.5 * relevance_i           evidence-quality weight in [0.5, 1]
               o_i = strongest opposing probability    (conflicting evidence lowers confidence)
    support  = sum_{i in S} a_i / N
    partial  = 0.25 * U / N                            small credit: unverifiable is not the same as false
    penalty  = 0.5 * sum_{i in C} a_i / N
    score    = 100 * clamp(support + partial - penalty, 0, 1)

If no claim could be classified, or none has conclusive (supported/contradicted) evidence, no score is
given and the label is "Insufficient evidence". The score is a heuristic estimate of how well the
response is backed by retrieved evidence. It is NOT a calibrated probability of factual correctness, and
the raw NLI probability (ClaimResult.model_confidence) is only the model's confidence in its label.
"""

from app.schemas.verification import ClaimResult, ClaimStatus, ScoreBreakdown, StatusCounts

FORMULA = (
    "score = 100 × clamp(Σ_supported a/N + 0.25·U/N − 0.5·Σ_contradicted a/N, 0, 1), "
    "a = NLI probability · q · (1 − 0.5·opposing probability), q = 0.5 + 0.5·evidence relevance, "
    "N = claims classified by NLI"
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
    support = sum((c.confidence or 0) / 100 for c in claims if c.status == ClaimStatus.SUPPORTED) / n
    penalty = 0.5 * sum((c.confidence or 0) / 100 for c in claims if c.status == ClaimStatus.CONTRADICTED) / n
    partial = 0.25 * counts.unverifiable / n
    score = round(100 * max(0.0, min(1.0, support + partial - penalty)), 1)
    label = "High" if score >= 75 else "Moderate" if score >= 50 else "Low"
    return ScoreBreakdown(
        score=score, label=label, counts=counts, formula=FORMULA,
        components={"support": round(support, 4), "unverifiable_credit": round(partial, 4),
                    "contradiction_penalty": round(penalty, 4), "classified_claims": n},
    )
