# VeriAI — Real-Time LLM Hallucination Detection & Verification System

**Trust Every Claim. Verify Every Answer.**

VeriAI is a verification layer for AI-generated text. It splits an answer into atomic factual claims,
retrieves evidence from Wikipedia and Wikidata, classifies every claim with a natural-language-inference
(NLI) model as `SUPPORTED`, `CONTRADICTED` or `UNVERIFIABLE`, computes a transparent reliability score,
optionally rewrites unreliable claims using only the retrieved evidence, and re-verifies the result.
Every run is stored in SQLite and can be revisited from the History page and summarised on the Dashboard.

It works fully offline from paid APIs: a deterministic **demo provider** produces canned answers (with
deliberate mistakes) and performs extractive, evidence-quoting corrections. An optional
**OpenAI-compatible provider** can be configured through environment variables.

## Architecture

```
frontend/  React 18 + TypeScript + Vite + Tailwind + Recharts + Lucide
  src/pages        Landing, Workspace (verify / generate), Dashboard, History, RunDetail, About
  src/components   RunView (original vs corrected), ClaimCard, ScoreCard, Layout, ui helpers
  src/services     api.ts – typed fetch client with structured error handling
backend/   FastAPI + Pydantic + SQLAlchemy + SQLite
  app/api          routes.py (REST endpoints), errors.py (consistent error envelope)
  app/services     claim_extractor  – spaCy sentence segmentation, NER, clause splitting, pronoun resolution
                   entity_recognizer – spaCy NER with regex fallback
                   knowledge_retriever – Wikipedia (MediaWiki API) + Wikidata (search + SPARQL); timeouts,
                                         retries with backoff, TTL cache, sanitisation, provider diversity
                   nli_verifier     – Hugging Face DeBERTa-v3 NLI (premise = evidence, hypothesis = claim)
                   confidence_scorer – documented reliability formula
                   llm_provider     – DemoProvider / OpenAICompatibleProvider
                   correction_service, verification_pipeline, history_service
  app/models       SQLAlchemy ORM (runs, claims, evidence)
  tests/           pytest suite (offline; fake retriever + fake NLI) and a live integration test
  eval/            labelled dataset + evaluation script
```

Pipeline: **input → (optional generation) → claim extraction → NER & query building → Wikipedia/Wikidata
evidence → NLI classification → reliability score → (optional correction) → re-verification → persistence.**

## Setup

Requirements: Python 3.10+, Node 18+ (tested with Node 20), ~2 GB disk for the NLI model. A GPU is not needed.

### Backend

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m spacy download en_core_web_sm
cp .env.example .env            # optional – defaults work without a .env
.venv/bin/uvicorn app.main:app --reload --port 8000
```

The NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, ~740 MB) is downloaded from Hugging Face on
first start and preloaded in the background. Until it is available, `/api/health` reports its status and
claims that have evidence are marked `VERIFICATION_UNAVAILABLE` — VeriAI never substitutes a heuristic
verdict for the model. If the spaCy model is missing, a regex sentence/entity fallback is used and a warning
is returned.

### Frontend

```bash
cd frontend
npm install
npm run dev          # http://localhost:5173 (proxies /api to http://localhost:8000)
```

For a production build: `npm run build` then serve `frontend/dist` (set `VITE_API_BASE_URL` to the backend
URL if it is not served from the same origin, and add that origin to the backend `FRONTEND_ORIGIN`).

## Environment variables (`backend/.env`)

| Variable | Default | Purpose |
|---|---|---|
| `FRONTEND_ORIGIN` | `http://localhost:5173` | Only origin allowed by CORS (comma-separate several) |
| `DATABASE_URL` | `sqlite:///./veriai.db` | SQLAlchemy URL |
| `LLM_PROVIDER` | `demo` | `demo` or `openai` |
| `LLM_API_KEY` | – | API key for the OpenAI-compatible provider (never logged or returned) |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible `/chat/completions` endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `NLI_ENABLED` / `NLI_MODEL` / `NLI_PRELOAD` | `true` / DeBERTa-v3 / `true` | NLI configuration |
| `NLI_ENTAIL_THRESHOLD` / `NLI_CONTRADICT_THRESHOLD` | `0.6` / `0.6` | Minimum probability for a verdict |
| `KNOWLEDGE_PROVIDERS` | `wikipedia,wikidata` | Enabled evidence sources |
| `HTTP_TIMEOUT_SECONDS` / `HTTP_RETRIES` | `8` / `2` | External request policy |
| `MAX_INPUT_CHARS` / `MAX_CLAIMS` | `8000` / `15` | Input limits |

Frontend (`frontend/.env`): `VITE_API_BASE_URL` (empty = same origin / dev proxy), `VITE_PROXY_TARGET`.

## API

All errors use `{"error": {"code": "...", "message": "...", "details": [...]}}`.

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | Component status (database, spaCy, NLI model, LLM configuration, knowledge sources) |
| POST | `/api/verify` | `{response_text, question?, auto_correct?}` → verifies an existing answer (201, `RunResponse`) |
| POST | `/api/generate-and-verify` | `{question, provider?: "demo"\|"openai", auto_correct?}` → generates then verifies |
| POST | `/api/correct` | `{run_id, provider?}` → corrects contradicted/unverifiable claims and re-verifies |
| GET | `/api/history?limit&offset` | Saved runs (summary), `X-Total-Count` header |
| GET | `/api/history/{id}` | Full run with claims, evidence, scores and correction |
| DELETE | `/api/history/{id}` | Delete one run (204) |
| DELETE | `/api/history?confirm=true` | Clear all history (400 without `confirm=true`) |
| GET | `/api/stats` | Dashboard aggregates computed from the database |

Interactive docs: http://localhost:8000/docs.

Example:

```bash
curl -s -X POST localhost:8000/api/verify -H 'Content-Type: application/json' \
  -d '{"response_text": "The Eiffel Tower is in Paris. It was designed by Thomas Edison."}'
```

## How verdicts and scores are produced

* **Claim extraction** – spaCy dependency parses split coordinated clauses ("X was born in A and won B")
  into atomic claims, resolve leading pronouns/definite descriptions ("He", "The tower") to the current
  topic, and skip questions, opinions, imperatives and fragments.
* **Retrieval** – entity-driven Wikipedia searches; each page is cut into two-sentence windows ranked by
  lexical relevance to the claim. Wikidata statements for the main entity are rendered as short sentences
  ("Albert Einstein died in Princeton."). Retrieved text is sanitised, capped, wrapped in delimiters and
  treated strictly as data; instruction-like text is flagged with a warning.
* **NLI** – each (evidence, claim) pair gets entailment / contradiction / neutral probabilities.
  Entailment ≥ threshold from relevant evidence ⇒ `SUPPORTED`. Contradiction ≥ threshold counts only from
  *direct* evidence: a passage that mentions one of the claim's values, or has a sentence about the claim's
  subject using the claim's relation (`text_utils.is_direct_evidence`). A passage about someone else, e.g.
  "Margot Einstein ... (she died in 1986)" for "Einstein died in London", is ignored even if NLI calls it a
  contradiction. The decisive passage is the one with the highest probability × evidence quality. Anything
  else, including no evidence ⇒ `UNVERIFIABLE`; disagreeing sources are flagged.
* **Claim confidence** – `confidence = 100 · p · q · (1 − 0.5·o)`, where `p` is the NLI probability of the
  verdict, `q = 0.5 + 0.5·relevance` and `o` is the strongest opposing probability (conflicting evidence).
  `model_confidence` is the raw NLI probability. Neither is a calibrated probability that the claim is true.
  `UNVERIFIABLE` claims have no confidence.
* **Reliability score** (heuristic, not a calibrated probability):

  `score = 100 × clamp(Σ_supported a/N + 0.25·U/N − 0.5·Σ_contradicted a/N, 0, 1)` with `a = confidence/100`,
  `N` classified claims and `U` unverifiable claims. No claims ⇒ "No factual claims"; no decidable claim ⇒
  "Insufficient evidence" (score `null`).
* **Correction** – the claim's non-subject values (place, date, person …) are split into values that the
  relevant evidence confirms and values it does not. The demo provider replaces only the unconfirmed values
  with a value of the same type quoted from evidence about the same subject and relation. For example,
  "He died in London in 1955." becomes "He died in Princeton in 1955.". If that is not possible, it falls back
  to quoting the best evidence sentence. The OpenAI-compatible provider gets the same keep/check hints and
  delimited evidence.
* **Re-verification** – the corrected answer runs through retrieval and NLI again. A rewritten claim counts as
  fixed only if that fresh evidence supports it. `correction.changes` lists each before → after pair with its
  new status and citations. `correction.unsupported_changed_claims` counts rewritten or added claims that are not
  supported, and these are flagged in the UI.

## Tests and evaluation

```bash
cd backend && .venv/bin/python -m pytest -m "not integration"   # offline suite
cd backend && .venv/bin/python -m pytest -m integration          # live Wikipedia/Wikidata + NLI
cd backend && .venv/bin/python -m eval.run_eval                  # writes eval/results.{json,md}
cd frontend && npm test && npm run lint && npm run build
```

The offline suite covers supported / contradicted / unverifiable claims, multiple claims, no claims,
malformed and oversized input, missing API keys, provider failures, retrieval timeouts and rate limits,
prompt-injection text in evidence, NLI unavailability, correction and re-verification, persistence, history
and stats endpoints. `tests/test_correction.py` adds regression tests for: fixing a wrong location while keeping
a correct date, fixing a wrong date while keeping a correct location, conflicting evidence, irrelevant passages,
no useful evidence, corrections that add an unsupported claim, and NLI or knowledge-source failure during
re-verification. See [`backend/eval/results.md`](backend/eval/results.md) for the latest measured
evaluation numbers.

## Limitations

* Wikipedia/Wikidata can be incomplete or outdated; missing evidence means `UNVERIFIABLE`, not false.
* Retrieval is lexical; relevant passages that use different wording can be missed, and irrelevant
  passages can be retrieved. NLI models struggle with numbers, negation and long passages.
* Claim extraction is rule-based and English-only.
* The demo provider is not an LLM: its answers are canned and its corrections are extractive quotes.
* Scores are heuristic. VeriAI is an academic prototype, not validated for medical, legal or financial use.
* CPU inference: a typical 4-claim answer takes ~5–25 s depending on network latency and cache state.

## Future improvements

Dense/semantic retrieval and reranking, more sources (news, scholarly APIs), claim-level calibration on a
larger labelled set, streaming progress events, multilingual support, and user feedback on verdicts.
