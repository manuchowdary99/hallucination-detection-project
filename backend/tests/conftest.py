"""Shared fixtures. Unit tests use fake retrieval/NLI components so they are fast and offline."""

import os
from datetime import datetime, timezone

os.environ.setdefault("NLI_PRELOAD", "false")
os.environ.setdefault("LLM_PROVIDER", "demo")
os.environ.setdefault("LLM_API_KEY", "")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_veriai.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.api.routes import pipeline_dep  # noqa: E402
from app.main import app  # noqa: E402
from app.models.database import get_db, init_db, make_engine  # noqa: E402
from app.schemas.verification import NLIScores  # noqa: E402
from app.services.claim_extractor import ClaimExtractor  # noqa: E402
from app.services.knowledge_retriever import EvidencePassage, RetrievalResult  # noqa: E402
from app.services.nli_verifier import NLIUnavailable  # noqa: E402
from app.services.text_utils import relevance  # noqa: E402
from app.services.verification_pipeline import VerificationPipeline  # noqa: E402

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

KB = {
    "Eiffel Tower": ("https://en.wikipedia.org/wiki/Eiffel_Tower",
                     "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889 "
                     "and designed by the company of Gustave Eiffel."),
    "Albert Einstein": ("https://en.wikipedia.org/wiki/Albert_Einstein",
                        "Albert Einstein was born in Ulm in 1879. He died in Princeton, New Jersey, in 1955."),
}


def passage(title: str, snippet: str, claim: str) -> EvidencePassage:
    url = KB.get(title, (f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}", ""))[0]
    rel = relevance(claim, f"{title} {snippet}")
    return EvidencePassage("Wikipedia", title, url, snippet, NOW, relevance=rel, novel_relevance=max(rel, 0.5))


class FakeRetriever:
    provider_names = ["FakeWiki"]

    def __init__(self, warnings: list[str] | None = None):
        self.calls: list[str] = []
        self.warnings = warnings or []

    async def retrieve(self, claim: str, terms: list[str]) -> RetrievalResult:
        self.calls.append(claim)
        for title, (_, text) in KB.items():
            if title.split()[-1].lower() in claim.lower():
                return RetrievalResult([passage(title, text, claim)], list(self.warnings))
        return RetrievalResult([], list(self.warnings))


# Deterministic NLI keyed by keywords in the hypothesis (claim).
FAKE_NLI_RULES = [
    ("thomas edison", NLIScores(entailment=0.01, contradiction=0.97, neutral=0.02)),
    ("london", NLIScores(entailment=0.02, contradiction=0.95, neutral=0.03)),
    ("paris", NLIScores(entailment=0.97, contradiction=0.01, neutral=0.02)),
    ("1889", NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03)),
    ("ulm", NLIScores(entailment=0.96, contradiction=0.01, neutral=0.03)),
    ("princeton", NLIScores(entailment=0.94, contradiction=0.02, neutral=0.04)),
]


class FakeNLI:
    status = ("loaded", "fake-nli")

    def __init__(self, unavailable: bool = False):
        self.unavailable = unavailable

    def predict(self, pairs):
        if self.unavailable:
            raise NLIUnavailable("model download failed")
        out = []
        for _, hyp in pairs:
            h = hyp.lower()
            out.append(next((s for k, s in FAKE_NLI_RULES if k in h),
                            NLIScores(entailment=0.1, contradiction=0.1, neutral=0.8)))
        return out


@pytest.fixture
def make_pipeline():
    def _make(nli_unavailable: bool = False, retriever=None):
        return VerificationPipeline(extractor=ClaimExtractor(), retriever=retriever or FakeRetriever(),
                                    nli=FakeNLI(unavailable=nli_unavailable))
    return _make


@pytest.fixture
def client(tmp_path, make_pipeline):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    def _db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    state = {"pipeline": make_pipeline()}
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[pipeline_dep] = lambda: state["pipeline"]
    with TestClient(app) as c:
        c.state = state
        yield c
    app.dependency_overrides.clear()
    engine.dispose()
