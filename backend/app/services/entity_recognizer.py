"""Named Entity Recognition using spaCy, with a capitalisation-based fallback."""

import logging
import re
import threading

from app.core.config import get_settings

log = logging.getLogger(__name__)

SEARCHABLE_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "EVENT", "WORK_OF_ART", "PRODUCT", "LAW", "LANGUAGE",
}
FACTUAL_LABELS = SEARCHABLE_LABELS | {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}


class NLPUnavailable(RuntimeError):
    pass


class EntityRecognizer:
    """Loads the spaCy pipeline once and exposes NER helpers."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or get_settings().spacy_model
        self._nlp = None
        self._error: str | None = None
        self._lock = threading.Lock()

    @property
    def status(self) -> tuple[str, str | None]:
        if self._nlp is not None:
            return "loaded", self.model_name
        if self._error:
            return "fallback", self._error
        return "not_loaded", self.model_name

    def load(self):
        if self._nlp is not None or self._error:
            return self._nlp
        with self._lock:
            if self._nlp is None and not self._error:
                try:
                    import spacy

                    self._nlp = spacy.load(self.model_name)
                except Exception as exc:  # model missing or spaCy not installed
                    self._error = f"spaCy model '{self.model_name}' unavailable: {exc}"
                    log.warning(self._error)
        return self._nlp

    @property
    def available(self) -> bool:
        return self.load() is not None

    def doc(self, text: str):
        nlp = self.load()
        if nlp is None:
            raise NLPUnavailable(self._error or "spaCy unavailable")
        return nlp(text)

    def entities(self, text: str) -> list[dict]:
        nlp = self.load()
        if nlp is None:
            return fallback_entities(text)
        seen, out = set(), []
        for ent in nlp(text).ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                out.append({"text": ent.text, "label": ent.label_})
        return out


_CAP_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+(?:of|the|de|von|van)\s+)?)+(?:\s+[A-Z][a-z]+)*")
_NUM_RE = re.compile(r"\b\d[\d,.]*\b")


def fallback_entities(text: str) -> list[dict]:
    """Very rough NER used only when spaCy cannot be loaded."""
    out, seen = [], set()
    for m in _CAP_RE.finditer(text):
        span = m.group(0).strip()
        if span.lower() in seen or span in {"The", "A", "An", "It", "He", "She", "They", "This"}:
            continue
        seen.add(span.lower())
        out.append({"text": span, "label": "PROPN"})
    for m in _NUM_RE.finditer(text):
        out.append({"text": m.group(0), "label": "CARDINAL"})
    return out


_recognizer: EntityRecognizer | None = None


def get_entity_recognizer() -> EntityRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = EntityRecognizer()
    return _recognizer
