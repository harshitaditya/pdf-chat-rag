"""Lightweight free web search using open RSS & JSON feeds (no extra API keys)."""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import re
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import xml.etree.ElementTree as ET

import httpx


GREETING_ONLY_WORDS = {
    "hi",
    "hii",
    "hiii",
    "hello",
    "hey",
    "heyy",
    "yo",
    "sup",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "thx",
    "bye",
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "how are you",
}


@dataclass(frozen=True)
class WebHit:
    """Normalized live web search item."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    published_at: str = ""


def should_search_web(question: str, has_attached_docs: bool = False) -> bool:
    """Skip live search for trivial greetings or pure document-summary commands."""
    normalized = re.sub(r"[?!.,]+$", "", question.strip().lower())
    if not normalized or normalized in GREETING_ONLY_WORDS:
        return False
    if len(normalized) <= 3:
        return False
    if has_attached_docs and any(
        phrase in normalized
        for phrase in ("this pdf", "this document", "attached file", "summarize the pdf")
    ):
        return False
    return True


def _clean_text(value: Optional[str], limit: int = 260) -> str:
    if not value:
        return ""
    stripped = re.sub(r"<[^>]+>", " ", html.unescape(value))
    collapsed = re.sub(r"\s+", " ", stripped).strip()
    return collapsed[:limit]


def _unwrap_bing_url(raw_url: str) -> str:
    parsed = urlparse(raw_url)
    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/news/apiclick"):
        target = parse_qs(parsed.query).get("url", [None])[0]
        if target:
            decoded = unquote(target)
            if decoded.startswith(("http://", "https://")):
                return decoded
    return raw_url


def _parse_bing_news_rss(xml_text: str, limit: int = 4) -> List[WebHit]:
    hits: List[WebHit] = []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item")[:limit]:
        title = _clean_text(item.findtext("title"), 160)
        raw_link = (item.findtext("link") or "").strip()
        url = _unwrap_bing_url(raw_link)
        desc = _clean_text(item.findtext("description"), 260)
        pub_date = _clean_text(item.findtext("pubDate"), 40)
        if title and url.startswith(("http://", "https://")):
            hits.append(
                WebHit(
                    title=title,
                    url=url,
                    snippet=desc,
                    source=urlparse(url).netloc.replace("www.", ""),
                    published_at=pub_date,
                )
            )
    return hits


def _parse_google_news_rss(xml_text: str, limit: int = 4) -> List[WebHit]:
    hits: List[WebHit] = []
    root = ET.fromstring(xml_text)
    for item in root.findall(".//item")[:limit]:
        title = _clean_text(item.findtext("title"), 160)
        url = (item.findtext("link") or "").strip()
        pub_date = _clean_text(item.findtext("pubDate"), 40)
        src_el = item.find("source")
        src = _clean_text(src_el.text if src_el is not None else "", 50)
        if title and url.startswith(("http://", "https://")):
            hits.append(
                WebHit(
                    title=title,
                    url=url,
                    snippet="",
                    source=src or "Google News",
                    published_at=pub_date,
                )
            )
    return hits


def _parse_duckduckgo_json(data: Dict, limit: int = 2) -> List[WebHit]:
    hits: List[WebHit] = []
    abstract = _clean_text(data.get("AbstractText") or data.get("Abstract"), 300)
    abstract_url = (data.get("AbstractURL") or "").strip()
    heading = _clean_text(data.get("Heading"), 120) or "DuckDuckGo Summary"
    if abstract and abstract_url.startswith(("http://", "https://")):
        hits.append(
            WebHit(
                title=heading,
                url=abstract_url,
                snippet=abstract,
                source=urlparse(abstract_url).netloc.replace("www.", "") or "DuckDuckGo",
            )
        )

    for topic in data.get("RelatedTopics", []):
        if len(hits) >= limit:
            break
        if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
            url = str(topic["FirstURL"]).strip()
            text = _clean_text(topic["Text"], 220)
            if url.startswith(("http://", "https://")) and text:
                hits.append(
                    WebHit(
                        title=text.split(" - ")[0][:100],
                        url=url,
                        snippet=text,
                        source="DuckDuckGo",
                    )
                )
    return hits[:limit]


def _parse_wikipedia_json(data: Dict, limit: int = 2) -> List[WebHit]:
    hits: List[WebHit] = []
    for item in data.get("query", {}).get("search", [])[:limit]:
        title = _clean_text(item.get("title"), 120)
        snippet = _clean_text(item.get("snippet"), 260)
        if title:
            slug = quote_plus(title.replace(" ", "_"), safe="_()-")
            hits.append(
                WebHit(
                    title=f"{title} - Wikipedia",
                    url=f"https://en.wikipedia.org/wiki/{slug}",
                    snippet=snippet,
                    source="Wikipedia",
                )
            )
    return hits


def format_web_context(hits: List[WebHit]) -> Optional[str]:
    """Render fetched web hits into a prompt-ready context block."""
    if not hits:
        return None
    now_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M UTC")
    lines = [f"Live web search results (fetched {now_str}):"]
    for idx, hit in enumerate(hits, start=1):
        meta_parts = [p for p in (hit.source, hit.published_at) if p]
        meta = f" ({' | '.join(meta_parts)})" if meta_parts else ""
        lines.append(f"{idx}. {hit.title}{meta}")
        if hit.snippet:
            lines.append(f"   {hit.snippet}")
        lines.append(f"   URL: {hit.url}")
    return "\n".join(lines)


class WebSearchClient:
    """Async multi-source free web search client (Bing News, Google News, DDG, Wikipedia)."""

    USER_AGENT = "LocalAIChat/1.0 (https://github.com/harshitaditya/pdf-chat-rag)"

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    async def _fetch_bing(self, client: httpx.AsyncClient, query: str) -> List[WebHit]:
        resp = await client.get(
            "https://www.bing.com/news/search",
            params={"q": query, "format": "rss"},
        )
        if resp.status_code != 200:
            return []
        return _parse_bing_news_rss(resp.text, limit=4)

    async def _fetch_google_news(self, client: httpx.AsyncClient, query: str) -> List[WebHit]:
        resp = await client.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        )
        if resp.status_code != 200:
            return []
        return _parse_google_news_rss(resp.text, limit=3)

    async def _fetch_ddg(self, client: httpx.AsyncClient, query: str) -> List[WebHit]:
        resp = await client.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
        )
        if resp.status_code not in (200, 202):
            return []
        return _parse_duckduckgo_json(resp.json(), limit=2)

    async def _fetch_wikipedia(self, client: httpx.AsyncClient, query: str) -> List[WebHit]:
        resp = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 2,
            },
        )
        if resp.status_code != 200:
            return []
        return _parse_wikipedia_json(resp.json(), limit=2)

    async def search(self, query: str, max_results: int = 6) -> List[WebHit]:
        """Search free web sources concurrently; never raises on network errors."""
        clean_q = query.strip()
        if not clean_q:
            return []

        headers = {"User-Agent": self.USER_AGENT}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
                follow_redirects=True,
            ) as client:
                results = await asyncio.gather(
                    self._fetch_bing(client, clean_q),
                    self._fetch_google_news(client, clean_q),
                    self._fetch_ddg(client, clean_q),
                    self._fetch_wikipedia(client, clean_q),
                    return_exceptions=True,
                )
        except Exception:
            return []

        merged: List[WebHit] = []
        seen_urls = set()
        seen_titles = set()
        for group in results:
            if not isinstance(group, list):
                continue
            for hit in group:
                title_key = hit.title.lower()[:70]
                if hit.url in seen_urls or title_key in seen_titles:
                    continue
                seen_urls.add(hit.url)
                seen_titles.add(title_key)
                merged.append(hit)
                if len(merged) >= max_results:
                    return merged
        return merged
