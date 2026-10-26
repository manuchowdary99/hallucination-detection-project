import httpx
import respx

from app.core.config import get_settings
from tests.conftest import FakeRetriever


def verify(client, text, **kw):
    return client.post("/api/verify", json={"response_text": text, **kw})


def statuses(run, phase="original_verification"):
    return {c["text"]: c["status"] for c in run[phase]["claims"]}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["llm"]["demo_available"] is True
    assert body["knowledge_providers"] == ["FakeWiki"]


def test_supported_claim(client):
    r = verify(client, "The Eiffel Tower is located in Paris.")
    assert r.status_code == 201
    run = r.json()
    claim = run["original_verification"]["claims"][0]
    assert claim["status"] == "SUPPORTED"
    assert claim["evidence"][0]["url"].startswith("https://en.wikipedia.org/wiki/")
    assert claim["nli"]["entailment"] > 0.9 and claim["model_confidence"] == 97.0
    # evidence-weighted confidence = model probability x evidence quality (never above the raw probability)
    assert 48.5 <= claim["confidence"] <= claim["model_confidence"]
    assert run["original_verification"]["score"]["score"] == claim["confidence"]


def test_contradicted_claim(client):
    run = verify(client, "The Eiffel Tower was designed by Thomas Edison.").json()
    claim = run["original_verification"]["claims"][0]
    assert claim["status"] == "CONTRADICTED"
    assert "contradicts" in claim["explanation"]


def test_unverifiable_claim_without_evidence(client):
    run = verify(client, "The Zorblax Institute was founded in Atlantis in 1712.").json()
    claim = run["original_verification"]["claims"][0]
    assert claim["status"] == "UNVERIFIABLE" and claim["evidence"] == []
    assert run["original_verification"]["score"]["score"] is None
    assert run["original_verification"]["score"]["label"] == "Insufficient evidence"


def test_multiple_claims(client):
    run = verify(client, "The Eiffel Tower is located in Paris and was completed in 1889. "
                         "Albert Einstein died in London in 1955.").json()
    s = statuses(run)
    assert s == {
        "The Eiffel Tower is located in Paris.": "SUPPORTED",
        "The Eiffel Tower was completed in 1889.": "SUPPORTED",
        "Albert Einstein died in London in 1955.": "CONTRADICTED",
    }
    counts = run["original_verification"]["score"]["counts"]
    assert counts == {"supported": 2, "contradicted": 1, "unverifiable": 0, "unavailable": 0, "total": 3}


def test_no_factual_claims(client):
    run = verify(client, "Hello! How can I help you today?").json()
    assert run["original_verification"]["claims"] == []
    assert run["original_verification"]["score"]["label"] == "No factual claims"
    assert any("No verifiable factual claims" in w for w in run["warnings"])


def test_empty_and_malformed_input(client):
    assert verify(client, "   ").status_code == 422
    assert client.post("/api/verify", json={}).status_code == 422
    r = client.post("/api/verify", content="not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    too_long = "a" * (get_settings().max_input_chars + 1)
    assert verify(client, too_long).status_code == 422
    r = client.post("/api/verify", content="x" * 70_000, headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_missing_api_key_for_explicit_openai_provider(client):
    r = client.post("/api/generate-and-verify", json={"question": "Eiffel Tower?", "provider": "openai"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "provider_not_configured"


def test_generate_with_demo_provider_is_labelled(client):
    r = client.post("/api/generate-and-verify", json={"question": "Tell me about the Eiffel Tower"})
    assert r.status_code == 201
    run = r.json()
    assert run["provider"]["is_demo"] is True and run["mode"] == "generate"
    assert any("demo provider" in w for w in run["warnings"])
    assert "Thomas Edison" in run["original_answer"]
    assert statuses(run)["The Eiffel Tower was designed by Thomas Edison."] == "CONTRADICTED"


def test_llm_provider_failure(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "llm_api_key", "test-key")
    with respx.mock(assert_all_called=True) as mock:
        mock.post(f"{s.llm_base_url}/chat/completions").mock(return_value=httpx.Response(500, json={}))
        r = client.post("/api/generate-and-verify", json={"question": "Who was Einstein?", "provider": "openai"})
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "llm_provider_error"
    assert "test-key" not in r.text


def test_openai_provider_success_and_correction(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "llm_api_key", "test-key")
    answers = iter(["Albert Einstein died in London in 1955.", "Albert Einstein died in Princeton in 1955."])
    with respx.mock() as mock:
        route = mock.post(f"{s.llm_base_url}/chat/completions").mock(
            side_effect=lambda req: httpx.Response(200, json={"choices": [{"message": {"content": next(answers)}}]})
        )
        r = client.post("/api/generate-and-verify",
                        json={"question": "Where did Einstein die?", "provider": "openai", "auto_correct": True})
        sent = route.calls[1].request.content.decode()
    assert r.status_code == 201
    run = r.json()
    assert run["provider"] == {"name": "openai", "model": s.llm_model, "is_demo": False}
    assert run["correction"]["status"] == "corrected"
    assert run["final_answer"] == "Albert Einstein died in Princeton in 1955."
    assert statuses(run, "corrected_verification") == {"Albert Einstein died in Princeton in 1955.": "SUPPORTED"}
    assert "<evidence" in sent and "untrusted DATA" in sent


def test_nli_unavailable(client, make_pipeline):
    client.state["pipeline"] = make_pipeline(nli_unavailable=True)
    run = verify(client, "The Eiffel Tower is located in Paris. The Zorblax Institute is in Atlantis.").json()
    s = statuses(run)
    assert s["The Eiffel Tower is located in Paris."] == "VERIFICATION_UNAVAILABLE"
    assert s["The Zorblax Institute is in Atlantis."] == "UNVERIFIABLE"
    assert run["original_verification"]["score"]["score"] is None
    assert any("NLI verification unavailable" in w for w in run["warnings"])


def test_retrieval_warning_propagates(client, make_pipeline):
    client.state["pipeline"] = make_pipeline(retriever=FakeRetriever(["Wikipedia retrieval failed: timed out"]))
    run = verify(client, "The Zorblax Institute was founded in Atlantis in 1712.").json()
    assert "Wikipedia retrieval failed: timed out" in run["warnings"]
    assert run["original_verification"]["claims"][0]["status"] == "UNVERIFIABLE"


def test_demo_correction_and_reverification(client):
    run = verify(client, "The Eiffel Tower is located in Paris. Albert Einstein died in London in 1955.").json()
    r = client.post("/api/correct", json={"run_id": run["id"]})
    assert r.status_code == 200
    corrected = r.json()
    assert corrected["correction"]["status"] == "corrected"
    assert "London" not in corrected["final_answer"] and "Princeton" in corrected["final_answer"]
    assert corrected["original_answer"] == run["original_answer"]
    new = statuses(corrected, "corrected_verification")
    assert "CONTRADICTED" not in new.values()
    changed = [c for c in corrected["corrected_verification"]["claims"] if c["changed_in_correction"]]
    assert changed and corrected["correction"]["changed_claims"] == len(changed)
    assert corrected["correction"]["score_after"] > corrected["correction"]["score_before"]
    # persisted
    again = client.get(f"/api/history/{run['id']}").json()
    assert again["final_answer"] == corrected["final_answer"]
    assert again["corrected_verification"]["claims"]


def test_correction_not_needed(client):
    run = verify(client, "The Eiffel Tower is located in Paris.", auto_correct=True).json()
    assert run["correction"]["status"] == "not_needed"
    assert run["corrected_verification"] is None


def test_correct_unknown_run(client):
    assert client.post("/api/correct", json={"run_id": 999}).status_code == 404


def test_history_endpoints(client):
    ids = [verify(client, t).json()["id"] for t in
           ["The Eiffel Tower is located in Paris.", "Albert Einstein died in London in 1955."]]
    r = client.get("/api/history")
    assert r.status_code == 200 and r.headers["X-Total-Count"] == "2"
    items = r.json()
    assert [i["id"] for i in items] == ids[::-1]
    detail = client.get(f"/api/history/{ids[0]}").json()
    assert detail["original_verification"]["claims"][0]["evidence"][0]["snippet"]

    stats = client.get("/api/stats").json()
    assert stats["total_runs"] == 2 and stats["counts"]["supported"] == 1 and stats["counts"]["contradicted"] == 1
    assert stats["percentages"]["supported"] == 50.0 and stats["average_latency_ms"] is not None

    assert client.delete(f"/api/history/{ids[0]}").status_code == 204
    assert client.get(f"/api/history/{ids[0]}").status_code == 404
    assert client.delete(f"/api/history/{ids[0]}").status_code == 404
    assert client.delete("/api/history").status_code == 400
    assert client.delete("/api/history?confirm=true").json() == {"deleted": 1}
    assert client.get("/api/history").json() == []
    assert client.get("/api/stats").json()["total_runs"] == 0
