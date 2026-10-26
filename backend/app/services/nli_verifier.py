"""Natural Language Inference (NLI) verification of claims against evidence passages."""

import logging
import threading
from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.schemas.verification import ClaimStatus, NLIScores

log = logging.getLogger(__name__)


class NLIUnavailable(RuntimeError):
    pass


class NLIVerifier:
    """Wraps a Hugging Face sequence-classification NLI model (premise=evidence, hypothesis=claim).

    The model is loaded once (lazily or at startup) and reused for every request.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.nli_model
        self._model = None
        self._tokenizer = None
        self._label_idx: dict[str, int] = {}
        self._error: str | None = None if self.settings.nli_enabled else "NLI disabled by configuration"
        self._lock = threading.Lock()

    @property
    def status(self) -> tuple[str, str | None]:
        if self._model is not None:
            return "loaded", self.model_name
        if self._error:
            return "unavailable", self._error
        return "not_loaded", self.model_name

    @property
    def available(self) -> bool:
        try:
            self.load()
            return True
        except NLIUnavailable:
            return False

    def load(self) -> None:
        if self._model is not None:
            return
        if self._error:
            raise NLIUnavailable(self._error)
        with self._lock:
            if self._model is not None:
                return
            if self._error:
                raise NLIUnavailable(self._error)
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                model.eval()
                idx = {}
                for i, lab in model.config.id2label.items():
                    lab = lab.lower()
                    for key in ("entail", "contradict", "neutral"):
                        if key in lab:
                            idx[key] = int(i)
                if set(idx) != {"entail", "contradict", "neutral"}:
                    raise ValueError(f"model labels {model.config.id2label} are not NLI labels")
                self._tokenizer, self._model, self._label_idx = tok, model, idx
                log.info("Loaded NLI model %s", self.model_name)
            except Exception as exc:
                self._error = f"NLI model '{self.model_name}' could not be loaded: {exc}"
                log.error(self._error)
                raise NLIUnavailable(self._error) from exc

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Score (premise, hypothesis) pairs. Raises NLIUnavailable if the model cannot run."""
        if not pairs:
            return []
        self.load()
        import torch

        out: list[NLIScores] = []
        try:
            bs = self.settings.nli_batch_size
            for start in range(0, len(pairs), bs):
                batch = pairs[start : start + bs]
                enc = self._tokenizer(
                    [p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                    max_length=self.settings.nli_max_length, padding=True, return_tensors="pt",
                )
                with torch.inference_mode():
                    probs = torch.softmax(self._model(**enc).logits, dim=-1).tolist()
                for row in probs:
                    out.append(NLIScores(
                        entailment=round(row[self._label_idx["entail"]], 4),
                        contradiction=round(row[self._label_idx["contradict"]], 4),
                        neutral=round(row[self._label_idx["neutral"]], 4),
                    ))
        except Exception as exc:
            raise NLIUnavailable(f"NLI inference failed: {exc}") from exc
        return out


@dataclass
class PassageJudgement:
    scores: NLIScores
    relevance: float
    novel_relevance: float
    title: str


@dataclass
class ClaimJudgement:
    status: ClaimStatus
    confidence: float | None
    best: NLIScores | None
    best_index: int | None
    conflicting: bool
    explanation: str
    passage_labels: list[ClaimStatus]


def passage_label(j: PassageJudgement, s: Settings) -> ClaimStatus:
    if j.scores.entailment >= s.nli_entail_threshold:
        return ClaimStatus.SUPPORTED
    if j.scores.contradiction >= s.nli_contradict_threshold and _contradiction_relevant(j, s):
        return ClaimStatus.CONTRADICTED
    return ClaimStatus.UNVERIFIABLE


def _contradiction_relevant(j: PassageJudgement, s: Settings) -> bool:
    # A passage about a different aspect of the same entity often looks "contradictory" to NLI models.
    # Only count a contradiction if the passage shares content with the claim beyond the article subject.
    return j.novel_relevance >= s.contradiction_min_relevance or j.relevance >= 0.75


def aggregate(judgements: list[PassageJudgement], settings: Settings | None = None) -> ClaimJudgement:
    """Aggregate per-passage NLI results into one claim-level verdict.

    - SUPPORTED if the strongest entailing passage passes the entailment threshold.
    - CONTRADICTED if the strongest (relevant) contradicting passage passes the contradiction threshold.
    - If both happen, the evidence conflicts (``conflicting`` is reported): contradiction wins only if it
      exceeds the best entailment by ``nli_conflict_margin``; entailment wins if it is at least as strong;
      otherwise the claim is UNVERIFIABLE.
    - Otherwise (no evidence, or neutral/weak evidence) the claim is UNVERIFIABLE. Missing evidence never
      makes a claim CONTRADICTED.
    """
    s = settings or get_settings()
    if not judgements:
        return ClaimJudgement(ClaimStatus.UNVERIFIABLE, None, None, None, False,
                              "No relevant evidence was found in the configured sources, so the claim could not be "
                              "verified. This does not mean it is false.", [])
    labels = [passage_label(j, s) for j in judgements]
    sup = [(j.scores.entailment, i) for i, (j, l) in enumerate(zip(judgements, labels)) if l == ClaimStatus.SUPPORTED]
    con = [(j.scores.contradiction, i) for i, (j, l) in enumerate(zip(judgements, labels)) if l == ClaimStatus.CONTRADICTED]
    best_sup, best_con = max(sup, default=None), max(con, default=None)
    conflicting = bool(best_sup and best_con)

    if best_con and (not best_sup or best_con[0] - best_sup[0] >= s.nli_conflict_margin):
        i = best_con[1]
        msg = (f"Evidence from '{judgements[i].title}' contradicts the claim "
               f"(contradiction {best_con[0]:.2f}).")
        if conflicting:
            msg += f" However, another passage supports it (entailment {best_sup[0]:.2f}); review both sources."
        return ClaimJudgement(ClaimStatus.CONTRADICTED, round(best_con[0] * 100, 1), judgements[i].scores, i,
                              conflicting, msg, labels)
    if best_sup and (not best_con or best_sup[0] >= best_con[0]):
        # Entailment wins ties: NLI contradiction scores are more prone to false positives when a passage
        # discusses a different aspect of the same subject.
        i = best_sup[1]
        msg = (f"Evidence from '{judgements[i].title}' entails the claim "
               f"(entailment {best_sup[0]:.2f}).")
        if conflicting:
            msg += f" However, another passage appears to contradict it (contradiction {best_con[0]:.2f}); review both sources."
        return ClaimJudgement(ClaimStatus.SUPPORTED, round(best_sup[0] * 100, 1), judgements[i].scores, i,
                              conflicting, msg, labels)
    if conflicting:
        i = best_con[1]
        return ClaimJudgement(ClaimStatus.UNVERIFIABLE, None, judgements[i].scores, i, True,
                              f"Sources disagree: one passage entails the claim ({best_sup[0]:.2f}) and another "
                              f"contradicts it ({best_con[0]:.2f}). The claim is marked unverifiable.", labels)
    i = max(range(len(judgements)), key=lambda k: max(judgements[k].scores.entailment, judgements[k].scores.contradiction))
    top = judgements[i].scores
    return ClaimJudgement(ClaimStatus.UNVERIFIABLE, round(top.neutral * 100, 1), top, i, False,
                          f"Retrieved evidence was related but inconclusive (best entailment {top.entailment:.2f}, "
                          f"contradiction {top.contradiction:.2f}).", labels)


_verifier: NLIVerifier | None = None


def get_nli_verifier() -> NLIVerifier:
    global _verifier
    if _verifier is None:
        _verifier = NLIVerifier()
    return _verifier
