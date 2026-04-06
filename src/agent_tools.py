"""Allowlisted tools for Parlor: bounded HTTP to public APIs only (no arbitrary URLs)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

DEFAULT_TIMEOUT_S = float(os.environ.get("PARLOR_HTTP_TIMEOUT", "12"))
MAX_QUERY_LEN = int(os.environ.get("PARLOR_MAX_SEARCH_QUERY_LEN", "240"))
MAX_WEB_BODY_BYTES = int(os.environ.get("PARLOR_MAX_WEB_BODY_BYTES", "400000"))
MAX_COMBINED_SEARCH_CHARS = int(os.environ.get("PARLOR_MAX_WEB_SEARCH_CHARS", "4000"))

_USER_AGENT = "Parlor/0.1 (local voice assistant; +https://github.com/fikrikarim/parlor)"


def _fetch(url: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "application/json,text/plain,*/*"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(MAX_WEB_BODY_BYTES)


def _format_duckduckgo_instant(j: dict[str, Any]) -> str:
    chunks: list[str] = []
    if j.get("AbstractText"):
        chunks.append(str(j["AbstractText"]).strip())
    if j.get("Answer"):
        chunks.append(str(j["Answer"]).strip())
    for t in j.get("RelatedTopics", [])[:10]:
        if isinstance(t, dict) and t.get("Text"):
            chunks.append(str(t["Text"]).strip())
        elif isinstance(t, dict) and "Topics" in t:
            for sub in t.get("Topics", [])[:3]:
                if isinstance(sub, dict) and sub.get("Text"):
                    chunks.append(str(sub["Text"]).strip())
    return "\n".join(c for c in chunks if c)


def _duckduckgo_instant(query: str) -> str:
    q = urllib.parse.quote(query, safe="")
    url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&no_redirect=1"
    raw = _fetch(url)
    j = json.loads(raw.decode("utf-8", errors="replace"))
    return _format_duckduckgo_instant(j)


def _wikipedia_snippets(query: str) -> str:
    base = "https://en.wikipedia.org/w/api.php"
    q = urllib.parse.quote(query[:120])
    search_url = f"{base}?action=opensearch&search={q}&limit=3&namespace=0&format=json"
    raw = _fetch(search_url)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    if not titles or not isinstance(titles[0], str):
        return ""
    title = titles[0]
    tq = urllib.parse.quote(title)
    ex_url = (
        f"{base}?action=query&prop=extracts&exintro=1&explaintext=1"
        f"&titles={tq}&format=json&origin=*"
    )
    raw2 = _fetch(ex_url)
    j = json.loads(raw2.decode("utf-8", errors="replace"))
    pages = (j.get("query") or {}).get("pages") or {}
    for page in pages.values():
        if isinstance(page, dict):
            ex = page.get("extract")
            if isinstance(ex, str) and ex.strip():
                return f"{title}: {ex.strip()}"
    return ""


def web_search(query: str) -> str:
    """Look up current or general knowledge using DuckDuckGo and Wikipedia.

    Use when the user asks for recent events, facts you are unsure about,
    weather, prices, or anything that benefits from a quick web lookup.

    Args:
        query: A focused search query (people, places, topics, questions).
    """
    q = (query or "").strip()
    if not q:
        return "Empty search query."
    if len(q) > MAX_QUERY_LEN:
        q = q[:MAX_QUERY_LEN]

    parts: list[str] = []
    try:
        ddg = _duckduckgo_instant(q)
        if ddg:
            parts.append("[DuckDuckGo]\n" + ddg)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        parts.append(f"[DuckDuckGo failed: {e}]")

    combined = "\n\n".join(parts)
    if len(combined.strip()) < 120:
        try:
            wiki = _wikipedia_snippets(q)
            if wiki:
                parts.append("[Wikipedia]\n" + wiki)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            parts.append(f"[Wikipedia failed: {e}]")

    out = "\n\n".join(parts).strip()
    if not out:
        return "No useful results for that query. Try different keywords."
    if len(out) > MAX_COMBINED_SEARCH_CHARS:
        return out[: MAX_COMBINED_SEARCH_CHARS - 20] + "\n…[truncated]"
    return out


def get_current_utc_time() -> str:
    """Return the current date and time in UTC (ISO-style). Use when the user asks for the time or date."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
