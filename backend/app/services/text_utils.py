"""Lightweight lexical helpers used for passage ranking and claim comparison."""

import re
import unicodedata

STOPWORDS = frozenset(
    """a an the and or but if of at by for with about against between into through during before after above below
    to from up down in out on off over under again further then once here there when where why how all any both each
    few more most other some such no nor not only own same so than too very s t can will just don should now is are
    was were be been being have has had having do does did doing it its it's this that these those he she they them
    his her their i you we our your which who whom what as also one known widely first""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|you are now|"
    r"system prompt|<\s*/?\s*(system|assistant|instructions?)\s*>)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower()


def tokens(text: str) -> list[str]:
    return [t.replace(",", "") for t in _TOKEN_RE.findall(normalize(text))]


def content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())}


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(tok) > len(suf) + 3 and tok.endswith(suf):
            return tok[: -len(suf)]
    return tok


def relevance(claim: str, passage: str, key_terms: set[str] | None = None) -> float:
    """Weighted fraction of the claim's content tokens that appear in the passage (0-1).

    Numbers and entity tokens (``key_terms``) are weighted more heavily because they carry most of a
    claim's factual content.
    """
    claim_toks = content_tokens(claim)
    if not claim_toks:
        return 0.0
    passage_stems = {_stem(t) for t in content_tokens(passage)}
    key = {t for kt in (key_terms or set()) for t in content_tokens(kt)}
    total = hit = 0.0
    for tok in claim_toks:
        weight = 2.0 if tok.isdigit() or tok in key else 1.0
        total += weight
        if _stem(tok) in passage_stems:
            hit += weight
    return round(hit / total, 4)


def similarity(a: str, b: str) -> float:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def sanitize_evidence(text: str, max_chars: int = 1200) -> tuple[str, bool]:
    """Strip control characters, collapse whitespace and flag instruction-like content.

    Retrieved text is always treated as untrusted data; this does not make it trusted, it only reduces noise
    and lets callers surface a warning.
    """
    text = _CONTROL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    suspicious = bool(_INJECTION_RE.search(text))
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return text, suspicious


def split_sentences(text: str) -> list[str]:
    """Regex sentence splitter used for evidence passages and as a fallback when spaCy is unavailable."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


_PRONOUN_START_RE = re.compile(r"^(he|she|it|they|his|her|its|their)\b", re.IGNORECASE)
_LEADING_NAME_RE = re.compile(r"^((?:The\s+)?(?:[A-Z][\w'.-]*\s+)*[A-Z][\w'.-]*)")


def stems(text: str) -> set[str]:
    return {_stem(t) for t in content_tokens(text)}


def leading_subject(claim: str) -> str | None:
    """The capitalised name a claim starts with ("Einstein died ..." -> "Einstein"), if any."""
    m = _LEADING_NAME_RE.match(claim.strip())
    return m.group(1) if m else None


def predicate_stems(claim: str, subject_terms: list[str], value_terms: list[str]) -> set[str]:
    """Claim words that are neither the subject nor a value ("died", "designed"): they select the relation."""
    return stems(claim) - stems(" ".join(subject_terms + value_terms))


def sentence_about_subject(sentence: str, subject_terms: list[str], title: str | None = None) -> bool:
    """True if the sentence's grammatical lead is the subject (or a pronoun standing for the page subject)."""
    s = re.sub(r"^(the|a|an)\s+", "", sentence.strip(), flags=re.IGNORECASE)
    if _PRONOUN_START_RE.match(s):
        return True
    first = tokens(s)[:1]
    return bool(first) and _stem(first[0]) in stems(" ".join(subject_terms + ([title] if title else [])))


def is_direct_evidence(claim: str, subject_terms: list[str], value_terms: list[str], title: str,
                       snippet: str) -> bool:
    """Whether a passage speaks to *this* claim rather than to another aspect of the same subject.

    Direct if it mentions one of the claim's values (place, date, person ...), or if one of its sentences
    is about the subject and uses the claim's predicate ("Albert Einstein died in Princeton.").
    A passage like "Margot Einstein ... (she died in 1986)" is not direct evidence for "Einstein died in
    London in 1955": it shares the predicate but is about someone else and mentions none of the values.
    """
    if stems(" ".join(value_terms)) & stems(snippet):
        return True
    preds = predicate_stems(claim, subject_terms, value_terms)
    for sentence in split_sentences(snippet):
        if (not preds or preds & stems(sentence)) and sentence_about_subject(sentence, subject_terms, title):
            return True
    return False
