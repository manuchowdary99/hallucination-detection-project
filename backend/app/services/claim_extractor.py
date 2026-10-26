"""Extract atomic, independently verifiable factual claims from a response.

Approach:
1. spaCy sentence segmentation.
2. Discard sentences that are not factual assertions (questions, imperatives, first-person opinions,
   hedged statements, sentences without any entity/number/proper noun).
3. Split coordinated clauses ("X was born in A and won B") into atomic claims, copying the subject
   into clauses that lack one.
4. Resolve sentence-initial pronouns (it/he/she/they/its/his/her/their) to the most recent salient
   subject so each claim can be verified on its own.
"""

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.services.entity_recognizer import (
    FACTUAL_LABELS,
    SEARCHABLE_LABELS,
    EntityRecognizer,
    fallback_entities,
    get_entity_recognizer,
)
from app.services.text_utils import split_sentences

OPINION_PATTERNS = re.compile(
    r"^\s*(i|we)\s+(think|believe|feel|guess|suppose|hope|recommend|suggest|love|like|prefer)\b|"
    r"\b(in my opinion|personally|i'm not sure|i am not sure|as an ai|i don't know|i do not know)\b",
    re.IGNORECASE,
)
SUBJECT_PRONOUNS = {"it", "he", "she", "they"}
POSSESSIVE_PRONOUNS = {"its", "his", "her", "their"}
SUBJ_DEPS = {"nsubj", "nsubjpass", "csubj", "expl"}


@dataclass
class ExtractedClaim:
    text: str
    sentence: str
    entities: list[dict] = field(default_factory=list)
    subject: str | None = None

    @property
    def search_terms(self) -> list[str]:
        terms = [e["text"] for e in self.entities if e["label"] in SEARCHABLE_LABELS | {"PROPN"}]
        if self.subject and self.subject not in terms:
            terms.insert(0, self.subject)
        return terms


class ClaimExtractor:
    def __init__(self, recognizer: EntityRecognizer | None = None, max_claims: int | None = None):
        self.recognizer = recognizer or get_entity_recognizer()
        self.max_claims = max_claims or get_settings().max_claims

    def extract(self, text: str) -> tuple[list[ExtractedClaim], list[str]]:
        warnings: list[str] = []
        if self.recognizer.available:
            claims = self._extract_spacy(text)
        else:
            warnings.append("spaCy model unavailable; using a simpler regex-based claim extractor.")
            claims = self._extract_fallback(text)
        seen, unique = set(), []
        for c in claims:
            key = re.sub(r"\W+", " ", c.text.lower()).strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        if len(unique) > self.max_claims:
            warnings.append(f"Response contained {len(unique)} claims; only the first {self.max_claims} were verified.")
            unique = unique[: self.max_claims]
        return unique, warnings

    # ---- spaCy path -------------------------------------------------------------------------
    def _extract_spacy(self, text: str) -> list[ExtractedClaim]:
        doc = self.recognizer.doc(text)
        claims: list[ExtractedClaim] = []
        salient: str | None = None
        for sent in doc.sents:
            s_text = sent.text.strip()
            if not self._is_factual_sentence(sent, s_text):
                # Still track the topic ("Python is a language.") so later pronouns can be resolved.
                subj = next((t for t in sent.root.children if t.dep_ in SUBJ_DEPS), None)
                if subj is not None and subj.text[:1].isupper() and subj.lower_ not in SUBJECT_PRONOUNS:
                    salient = _span_text(subj)
                continue
            for clause_text, subj_span in self._split_clauses(sent):
                clause_text, subject = self._resolve_pronoun(clause_text, subj_span, salient)
                clause_text = _tidy(clause_text)
                if len(clause_text.split()) < 3:
                    continue
                if subject and subject[:1].isupper() and subject.lower() not in SUBJECT_PRONOUNS:
                    salient = subject
                ents = self.recognizer.entities(clause_text)
                if not _has_factual_content(ents, clause_text):
                    continue
                claims.append(ExtractedClaim(text=clause_text, sentence=s_text, entities=ents, subject=subject))
            if salient is None or not any(t.pos_ == "PROPN" for t in sent if t.dep_ in SUBJ_DEPS):
                first_ent = next((e for e in sent.ents if e.label_ in SEARCHABLE_LABELS), None)
                if salient is None and first_ent is not None:
                    salient = _strip_det(first_ent.text)
        return claims

    @staticmethod
    def _is_factual_sentence(sent, s_text: str) -> bool:
        if s_text.endswith("?") or len(sent) < 4:
            return False
        if OPINION_PATTERNS.search(s_text):
            return False
        root = sent.root
        if root.pos_ not in {"VERB", "AUX"}:
            return False
        subj = next((t for t in root.children if t.dep_ in SUBJ_DEPS), None)
        if subj is None:  # imperative ("Please verify this.") or fragment
            return False
        if any(e.label_ in FACTUAL_LABELS for e in sent.ents) or any(t.pos_ in {"PROPN", "NUM"} for t in sent):
            return True
        # "He developed the theory of relativity." — third-person pronoun later resolved to an entity.
        return subj.lower_ in SUBJECT_PRONOUNS and any(t.pos_ == "NOUN" for t in sent)

    @staticmethod
    def _split_clauses(sent) -> list[tuple[str, str | None]]:
        root = sent.root
        heads = [root] + [t for t in root.children if t.dep_ == "conj" and t.pos_ in {"VERB", "AUX"}]
        root_subj = next((t for t in root.children if t.dep_ in SUBJ_DEPS), None)
        root_subj_text = _span_text(root_subj) if root_subj is not None else None
        if len(heads) == 1:
            return [(sent.text, root_subj_text)]
        excluded: set[int] = set()
        for h in heads[1:]:
            excluded.update(t.i for t in h.subtree)
            excluded.update(t.i for t in root.children if t.dep_ == "cc" and t.i < h.i)
        clauses = []
        for h in heads:
            if h is root:
                toks = [t for t in root.subtree if t.i not in excluded]
            else:
                toks = list(h.subtree)
                nested = [c for c in h.children if c.dep_ == "conj" and c.pos_ in {"VERB", "AUX"}]
                drop = {t.i for n in nested for t in n.subtree}
                toks = [t for t in toks if t.i not in drop]
            words = "".join(t.text_with_ws for t in toks).strip(" ,;")
            subj = next((t for t in h.children if t.dep_ in SUBJ_DEPS), None)
            if subj is None and root_subj_text:
                words = f"{root_subj_text} {words}"
                subj_text = root_subj_text
            else:
                subj_text = _span_text(subj) if subj is not None else None
            clauses.append((words, subj_text))
        return clauses

    @staticmethod
    def _resolve_pronoun(clause: str, subject: str | None, salient: str | None) -> tuple[str, str | None]:
        m = re.match(r"^(\w+)\b", clause)
        if not m or not salient:
            return clause, subject
        first = m.group(1).lower()
        d = re.match(r"^the (\w+)\b", clause, flags=re.I)
        if d and salient.lower().endswith(d.group(1).lower()) and len(salient.split()) > 1 \
                and not salient.lower().startswith(d.group(0).lower()):
            # Definite description referring back to the topic: "The tower" -> "The Eiffel Tower".
            topic = salient if salient.lower().startswith("the ") else f"The {salient}"
            return topic + clause[d.end():], topic
        if first in SUBJECT_PRONOUNS:
            return salient + clause[m.end():], salient
        if first in POSSESSIVE_PRONOUNS:
            return f"{salient}'s" + clause[m.end():], subject
        if subject and subject.lower() in SUBJECT_PRONOUNS:
            return clause, salient
        return clause, subject

    # ---- fallback path ----------------------------------------------------------------------
    def _extract_fallback(self, text: str) -> list[ExtractedClaim]:
        out = []
        for s in split_sentences(text):
            if s.endswith("?") or OPINION_PATTERNS.search(s) or len(s.split()) < 4:
                continue
            ents = fallback_entities(s)
            if not ents:
                continue
            out.append(ExtractedClaim(text=_tidy(s), sentence=s, entities=ents, subject=ents[0]["text"]))
        return out


def _span_text(token) -> str:
    toks = sorted(token.subtree, key=lambda t: t.i)
    return _strip_det("".join(t.text_with_ws for t in toks).strip())


def _strip_det(text: str) -> str:
    return text.strip()


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().strip(",;:")
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    if text and text[-1] not in ".!":
        text += "."
    return text[0].upper() + text[1:] if text else text


def _has_factual_content(ents: list[dict], text: str) -> bool:
    if any(e["label"] in FACTUAL_LABELS | {"PROPN"} for e in ents):
        return True
    return bool(re.search(r"\d", text)) or bool(re.search(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+", text))


_extractor: ClaimExtractor | None = None


def get_claim_extractor() -> ClaimExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClaimExtractor()
    return _extractor
