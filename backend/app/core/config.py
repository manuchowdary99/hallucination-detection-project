"""Application configuration loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "VeriAI"
    frontend_origin: str = "http://localhost:5173"
    database_url: str = "sqlite:///./veriai.db"

    # LLM provider: "demo" (no key needed) or "openai" (any OpenAI-compatible API).
    llm_provider: str = "demo"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30.0
    llm_temperature: float = 0.2

    # NLP / NLI models.
    spacy_model: str = "en_core_web_sm"
    nli_enabled: bool = True
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_preload: bool = True
    nli_max_length: int = 512
    nli_batch_size: int = 8
    nli_entail_threshold: float = 0.60
    nli_contradict_threshold: float = 0.60
    nli_conflict_margin: float = 0.15
    contradiction_min_relevance: float = 0.40

    # Knowledge retrieval.
    knowledge_providers: str = "wikipedia,wikidata"
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    wikidata_api_url: str = "https://www.wikidata.org/w/api.php"
    wikidata_sparql_url: str = "https://query.wikidata.org/sparql"
    http_user_agent: str = "VeriAI/1.0 (https://github.com/manuchowdary99/hallucination-detection-project; academic project)"
    http_timeout_seconds: float = 8.0
    http_retries: int = 2
    http_backoff_seconds: float = 0.5
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 512
    wikipedia_pages_per_claim: int = 3
    passages_per_page: int = 3
    max_evidence_per_claim: int = 5
    min_passage_relevance: float = 0.2

    # Request limits / behaviour.
    max_input_chars: int = Field(default=8000, ge=100)
    max_request_bytes: int = 64_000
    max_claims: int = 15
    auto_correct: bool = False

    @property
    def provider_list(self) -> list[str]:
        return [p.strip().lower() for p in self.knowledge_providers.split(",") if p.strip()]

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider.lower() == "openai" and bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
