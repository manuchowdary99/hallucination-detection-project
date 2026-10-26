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
