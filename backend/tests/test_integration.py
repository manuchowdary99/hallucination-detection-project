"""Live tests: real NLI model + Wikipedia/Wikidata. Run with `pytest -m integration`."""

import pytest

from app.services.verification_pipeline import VerificationPipeline

pytestmark = pytest.mark.integration


async def test_live_pipeline():
    p = VerificationPipeline()
    r = await p.verify_text("The Eiffel Tower is located in Paris, France. The Great Wall of China is located in Japan.")
    status = {c.text: c.status.value for c in r.claims}
    assert status["The Eiffel Tower is located in Paris, France."] == "SUPPORTED"
    assert status["The Great Wall of China is located in Japan."] == "CONTRADICTED"
