"""Evidence retrieval from trusted knowledge sources (Wikipedia, Wikidata) via a provider interface.

Retrieved text is treated as untrusted data: it is sanitised, never executed, and only used as NLI
premises or as clearly delimited context for the correction LLM.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote

import httpx

from app.core.config import Settings, get_settings
from app.services.text_utils import content_tokens, relevance, sanitize_evidence, split_sentences

log = logging.getLogger(__name__)


@dataclass
class EvidencePassage:
    source_name: str
    title: str
    url: str
    snippet: str
    retrieved_at: datetime
    relevance: float = 0.0
    novel_relevance: float = 0.0
    suspicious: bool = False


@dataclass
class RetrievalResult:
    passages: list[EvidencePassage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class RetrievalError(RuntimeError):
    pass


class TTLCache:
    def __init__(self, ttl: int, max_entries: int):
        self.ttl, self.max_entries = ttl, max_entries
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()

    def get(self, key: str):
        item = self._data.get(key)
        if item is None:
            return None
        ts, value = item
        if time.monotonic() - ts > self.ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value) -> None:
        self._data[key] = (time.monotonic(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


class HttpFetcher:
    """GET JSON with timeouts, retries with exponential backoff and an in-memory TTL cache."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.cache = TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)
        self._client = client

    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.settings.http_timeout_seconds,
                headers={"User-Agent": self.settings.http_user_agent, "Accept": "application/json"},
                follow_redirects=True,
            )
        return self._client

    async def get_json(self, url: str, params: dict) -> dict:
        key = hashlib.sha256((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(self.settings.http_retries + 1):
            try:
                resp = await self.client().get(url, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RetrievalError(f"HTTP {resp.status_code} from {httpx.URL(url).host}")
                resp.raise_for_status()
                data = resp.json()
                self.cache.set(key, data)
                return data
            except (httpx.TimeoutException, httpx.TransportError, RetrievalError, ValueError) as exc:
                last_exc = exc
                if attempt < self.settings.http_retries:
                    await asyncio.sleep(self.settings.http_backoff_seconds * (2**attempt))
            except httpx.HTTPStatusError as exc:
                raise RetrievalError(f"HTTP {exc.response.status_code} from {httpx.URL(url).host}") from exc
        kind = "timed out" if isinstance(last_exc, httpx.TimeoutException) else f"failed ({last_exc})"
        raise RetrievalError(f"Request to {httpx.URL(url).host} {kind}")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


class KnowledgeProvider(ABC):
    name: str

    def __init__(self, fetcher: HttpFetcher, settings: Settings):
        self.fetcher, self.settings = fetcher, settings

    @abstractmethod
    async def retrieve(self, claim: str, terms: list[str]) -> list[EvidencePassage]:
        """Return candidate evidence passages for a claim. Raise RetrievalError on failure."""


def _make_passage(source, title, url, text, claim, terms, now) -> EvidencePassage:
    snippet, suspicious = sanitize_evidence(text)
    title_toks = content_tokens(title)
    novel_claim = " ".join(t for t in sorted(content_tokens(claim)) if t not in title_toks)
    return EvidencePassage(
        source_name=source,
        title=title,
        url=url,
        snippet=snippet,
        retrieved_at=now,
        relevance=relevance(claim, f"{title} {snippet}", set(terms)),
        novel_relevance=relevance(novel_claim, snippet, set(terms)) if novel_claim else 0.0,
        suspicious=suspicious,
    )


class WikipediaProvider(KnowledgeProvider):
    name = "Wikipedia"

    async def _search(self, query: str, limit: int) -> list[str]:
        data = await self.fetcher.get_json(
            self.settings.wikipedia_api_url,
            {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json",
             "formatversion": 2, "srprop": ""},
        )
        return [r["title"] for r in data.get("query", {}).get("search", [])]

    async def _page(self, title: str) -> tuple[str, str, str] | None:
        data = await self.fetcher.get_json(
            self.settings.wikipedia_api_url,
            {"action": "query", "prop": "extracts|info", "explaintext": 1, "exsectionformat": "wiki",
             "inprop": "url", "titles": title, "redirects": 1, "format": "json", "formatversion": 2},
        )
        pages = data.get("query", {}).get("pages", [])
        if not pages or pages[0].get("missing") or not pages[0].get("extract"):
            return None
        p = pages[0]
        url = p.get("fullurl") or f"https://en.wikipedia.org/wiki/{quote(p['title'].replace(' ', '_'))}"
        return p["title"], url, p["extract"]

    async def retrieve(self, claim: str, terms: list[str]) -> list[EvidencePassage]:
        n_pages = self.settings.wikipedia_pages_per_claim
        searches = [self._search(t, 1) for t in terms[:2]] + [self._search(claim, n_pages)]
        results = await asyncio.gather(*searches, return_exceptions=True)
        errors = [r for r in results if isinstance(r, Exception)]
        if len(errors) == len(results):
            raise errors[0]
        titles: list[str] = []
        for r in results:
            if not isinstance(r, Exception):
                titles.extend(t for t in r if t not in titles)
        titles = titles[:n_pages]
        pages = await asyncio.gather(*(self._page(t) for t in titles), return_exceptions=True)
        now = datetime.now(timezone.utc)
        passages: list[EvidencePassage] = []
        for page in pages:
            if isinstance(page, Exception) or page is None:
                continue
            title, url, extract = page
            candidates = []
            for section in re.split(r"\n\s*==+[^=\n]+==+\s*\n", "\n" + extract):
                if any(h in section[:40] for h in ("References", "External links", "See also")):
                    continue
                sents = split_sentences(section)
                for i in range(len(sents)):
                    window = " ".join(sents[i : i + 2])
                    candidates.append(_make_passage(self.name, title, url, window, claim, terms, now))
            # Ties are common between overlapping windows; prefer clean, quote-free, shorter prose.
            # Every window of a page shares the title tokens, so rank by the claim's *other* tokens too and
            # prefer clean, quote-free, shorter prose.
            candidates.sort(key=lambda p: (p.relevance + 0.5 * p.novel_relevance
                                           - (0.1 if re.search(r'["“”]', p.snippet) else 0.0),
                                           -len(p.snippet)), reverse=True)
            passages.extend(candidates[: self.settings.passages_per_page])
        return passages


# Natural-language phrasing for common Wikidata properties: NLI models judge sentences far more
# reliably than "property: value" fragments.
PROPERTY_TEMPLATES = {
    "place of birth": "{s} was born in {v}.",
    "date of birth": "{s} was born on {v}.",
    "place of death": "{s} died in {v}.",
    "date of death": "{s} died on {v}.",
    "country of citizenship": "{s} was a citizen of {v}.",
    "award received": "{s} received the {v}.",
    "educated at": "{s} was educated at {v}.",
    "spouse": "{s} was married to {v}.",
    "occupation": "{s} worked as {v}.",
    "country": "{s} is in {v}.",
    "located in the administrative territorial entity": "{s} is located in {v}.",
    "capital of": "{s} is the capital of {v}.",
    "capital": "The capital of {s} is {v}.",
    "inception": "{s} was established on {v}.",
    "architect": "{s} was designed by {v}.",
    "founded by": "{s} was founded by {v}.",
    "developer": "{s} was developed by {v}.",
    "designed by": "{s} was designed by {v}.",
    "elevation above sea level": "{s} has an elevation of {v} metres above sea level.",
    "height": "{s} has a height of {v} metres.",
    "continent": "{s} is in {v}.",
    "official language": "The official language of {s} is {v}.",
}


class WikidataProvider(KnowledgeProvider):
    name = "Wikidata"

    SPARQL = """
    SELECT ?pLabel ?o ?oLabel WHERE {
      wd:%s ?wdt ?o .
      ?p wikibase:directClaim ?wdt .
      FILTER(STRSTARTS(STR(?o), "http://www.wikidata.org/entity/Q") || DATATYPE(?o) = xsd:dateTime
             || DATATYPE(?o) = xsd:decimal)
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    } LIMIT 400
    """

    async def _entity(self, term: str) -> dict | None:
        data = await self.fetcher.get_json(
            self.settings.wikidata_api_url,
            {"action": "wbsearchentities", "search": term, "language": "en", "limit": 1, "format": "json"},
        )
        hits = data.get("search", [])
        return hits[0] if hits else None

    async def _entity_via_wikipedia(self, term: str) -> dict | None:
        """Wikipedia search ranks ambiguous names ("Einstein") far better than wbsearchentities."""
        data = await self.fetcher.get_json(
            self.settings.wikipedia_api_url,
            {"action": "query", "generator": "search", "gsrsearch": term, "gsrlimit": 1, "prop": "pageprops",
             "ppprop": "wikibase_item", "redirects": 1, "format": "json", "formatversion": 2},
        )
        for page in data.get("query", {}).get("pages", []):
            qid = page.get("pageprops", {}).get("wikibase_item")
            if qid:
                return {"id": qid, "label": page.get("title")}
        return None

    async def retrieve(self, claim: str, terms: list[str]) -> list[EvidencePassage]:
        if not terms:
            return []
        term = re.sub(r"^(the|a|an)\s+", "", terms[0], flags=re.I).replace("'s", "")
        try:
            entity = await self._entity_via_wikipedia(term)
        except RetrievalError as exc:
            log.info("Wikipedia entity resolution failed for %r: %s", term, exc)
            entity = None
        entity = entity or await self._entity(term)
        if not entity or not re.fullmatch(r"Q\d+", entity.get("id", "")):
            return []
        qid, label = entity["id"], entity.get("label") or terms[0]
        url = f"https://www.wikidata.org/wiki/{qid}"
        now = datetime.now(timezone.utc)
        statements: list[str] = []
        if entity.get("description"):
            statements.append(f"{label}: {entity['description']}.")
        data = await self.fetcher.get_json(
            self.settings.wikidata_sparql_url, {"query": self.SPARQL % qid, "format": "json"}
        )
        grouped: OrderedDict[str, list[str]] = OrderedDict()
        for row in data.get("results", {}).get("bindings", []):
            prop = row.get("pLabel", {}).get("value")
            raw = row.get("o", {})
            value = row.get("oLabel", {}).get("value") or raw.get("value")
            if not prop or not value or re.fullmatch(r"Q\d+", value):
                continue
            if raw.get("datatype", "").endswith("dateTime"):
                value = _format_wikidata_date(raw["value"])
            vals = grouped.setdefault(prop, [])
            if value not in vals and len(vals) < 6:
                vals.append(value)
        for verb, place, date in (("was born", "place of birth", "date of birth"), ("died", "place of death", "date of death")):
            if len(grouped.get(place, [])) == 1 and len(grouped.get(date, [])) == 1:
                statements.append(f"{label} {verb} in {grouped[place][0]} on {grouped[date][0]}.")
        for prop, vals in grouped.items():
            template = PROPERTY_TEMPLATES.get(prop)
            joined = ", ".join(vals) if template is None or len(vals) == 1 else " or ".join(vals)
            statements.append(template.format(s=label, v=joined) if template else f"{label} — {prop}: {joined}.")
        passages = [_make_passage(self.name, label, url, s, claim, terms, now) for s in statements]
        # The claim's predicate words ("died", "designed", "born") are what select the right property;
        # entity tokens alone over-rank unrelated statements that merely mention the same place or year.
        key = {t for term in terms for t in content_tokens(term)} | content_tokens(label)
        predicate = " ".join(sorted(content_tokens(claim) - key))
        for p in passages:
            if predicate and relevance(predicate, p.snippet) > 0:
                p.relevance = round(max(p.relevance, 0.5 + 0.5 * p.relevance), 4)
                p.novel_relevance = round(max(p.novel_relevance, 0.5 + 0.5 * p.novel_relevance), 4)
        passages.sort(key=lambda p: p.relevance, reverse=True)
        return passages[:4]


def _format_wikidata_date(value: str) -> str:
    m = re.match(r"^\+?(-?\d{1,4})-(\d{2})-(\d{2})", value)
    if not m:
        return value
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if month == 0 or (month == 1 and day == 1):
        return str(year)
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    return f"{day} {months[month - 1]} {year}" if day else f"{months[month - 1]} {year}"


PROVIDERS: dict[str, type[KnowledgeProvider]] = {"wikipedia": WikipediaProvider, "wikidata": WikidataProvider}


class KnowledgeRetriever:
    def __init__(self, settings: Settings | None = None, providers: list[KnowledgeProvider] | None = None,
                 fetcher: HttpFetcher | None = None):
        self.settings = settings or get_settings()
        self.fetcher = fetcher or HttpFetcher(self.settings)
        if providers is None:
            providers = [PROVIDERS[n](self.fetcher, self.settings) for n in self.settings.provider_list if n in PROVIDERS]
        self.providers = providers
        self._sem = asyncio.Semaphore(4)

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    async def retrieve(self, claim: str, terms: list[str]) -> RetrievalResult:
        result = RetrievalResult()
        async with self._sem:
            outputs = await asyncio.gather(*(p.retrieve(claim, terms) for p in self.providers), return_exceptions=True)
        passages: list[EvidencePassage] = []
        for provider, out in zip(self.providers, outputs):
            if isinstance(out, Exception):
                log.warning("%s retrieval failed: %s", provider.name, out)
                result.warnings.append(f"{provider.name} retrieval failed: {out}")
            else:
                passages.extend(out)
        seen, unique = set(), []
        for p in sorted(passages, key=lambda p: p.relevance, reverse=True):
            if p.relevance < self.settings.min_passage_relevance or p.snippet in seen:
                continue
            seen.add(p.snippet)
            unique.append(p)
        if any(p.suspicious for p in unique):
            result.warnings.append("Some retrieved evidence contained instruction-like text; it was treated as data only.")
        result.passages = self._diversify(unique, self.settings.max_evidence_per_claim)
        return result

    @staticmethod
    def _diversify(passages: list[EvidencePassage], limit: int, per_source: int = 2) -> list[EvidencePassage]:
        """Keep the best passages overall, but reserve up to ``per_source`` slots for each source so a
        concise structured fact is not crowded out by many similar prose windows."""
        chosen: list[EvidencePassage] = []
        for source in dict.fromkeys(p.source_name for p in passages):
            chosen.extend([p for p in passages if p.source_name == source][:per_source])
        chosen = chosen[:limit]
        chosen.extend(p for p in passages if p not in chosen)
        chosen = chosen[:limit]
        return sorted(chosen, key=lambda p: p.relevance, reverse=True)


_retriever: KnowledgeRetriever | None = None


def get_knowledge_retriever() -> KnowledgeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever
