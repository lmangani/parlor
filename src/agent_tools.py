"""Allowlisted tools for Parlor: bounded HTTP to public APIs only (no arbitrary URLs)."""

from __future__ import annotations

import html as html_lib
import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

# Gemma / LiteRT may emit different JSON keys for the search string.
_SEARCH_ARG_KEYS = (
    "query",
    "q",
    "search_query",
    "topic",
    "keywords",
    "search",
    "text",
    "question",
)


def extract_page_url(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None
    for key in _PAGE_URL_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_search_query(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None
    for key in _SEARCH_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None

DEFAULT_TIMEOUT_S = float(os.environ.get("PARLOR_HTTP_TIMEOUT", "12"))
MAX_QUERY_LEN = int(os.environ.get("PARLOR_MAX_SEARCH_QUERY_LEN", "240"))
MAX_WEB_BODY_BYTES = int(os.environ.get("PARLOR_MAX_WEB_BODY_BYTES", "400000"))
MAX_COMBINED_SEARCH_CHARS = int(os.environ.get("PARLOR_MAX_WEB_SEARCH_CHARS", "4000"))
MAX_READ_PAGE_CHARS = int(os.environ.get("PARLOR_MAX_READ_PAGE_CHARS", "10000"))
MAX_PAGE_URL_LEN = int(os.environ.get("PARLOR_MAX_PAGE_URL_LEN", "2048"))

_USER_AGENT = "Parlor/0.1 (local voice assistant; +https://github.com/fikrikarim/parlor)"

# Gemma / LiteRT may emit different keys for a page URL.
_PAGE_URL_ARG_KEYS = (
    "url",
    "link",
    "href",
    "page_url",
    "address",
    "website",
    "page",
)


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


def _strip_tags(fragment: str) -> str:
    t = re.sub(r"<[^>]+>", " ", fragment)
    return html_lib.unescape(re.sub(r"\s+", " ", t).strip())


def _normalize_result_url(href: str) -> str:
    h = (href or "").strip()
    if h.startswith("//"):
        h = "https:" + h
    try:
        parsed = urllib.parse.urlparse(h)
    except ValueError:
        return h
    if "duckduckgo.com" in (parsed.netloc or "") and parsed.query and "uddg=" in parsed.query:
        qs = urllib.parse.parse_qs(parsed.query)
        inner = (qs.get("uddg") or [None])[0]
        if inner:
            return urllib.parse.unquote(inner)
    return h


def normalize_user_url(url: str) -> str:
    """Public alias for normalizing tracking redirects (e.g. DDG) to a real destination."""
    return _normalize_result_url(url).strip()


def url_is_allowed_for_fetch(url: str) -> bool:
    """http(s) only, no obvious SSRF targets, and passes user-visible URL hygiene."""
    u = normalize_user_url(url)
    if not u or len(u) > MAX_PAGE_URL_LEN:
        return False
    if not url_is_safe_for_user(u):
        return False
    try:
        parsed = urllib.parse.urlparse(u)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host or "@" in (parsed.netloc or ""):
        return False
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "metadata.google.internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    except ValueError:
        pass
    return True


def url_is_safe_for_user(url: str) -> bool:
    """Drop sponsored / tracking DuckDuckGo and ad redirects from user-visible results."""
    u = (url or "").strip()
    if not u:
        return False
    low = u.lower()
    if not low.startswith(("http://", "https://")):
        return False
    bad_fragments = (
        "duckduckgo.com/y.js",
        "/y.js?",
        "ad_provider=",
        "ad_domain=",
        "ad_type=",
        "bing.com/aclick",
        "googleadservices",
        "doubleclick.net",
        "pagead2.googlesyndication",
    )
    if any(b in low for b in bad_fragments):
        return False
    try:
        p = urllib.parse.urlparse(low)
    except ValueError:
        return False
    host = p.netloc or ""
    if host.endswith("duckduckgo.com") and p.path.rstrip("/").endswith("y.js"):
        return False
    return True


def _add_source(
    sources: list[dict[str, str]],
    seen_urls: set[str],
    title: str,
    url: str,
) -> None:
    u = _normalize_result_url(url).strip()
    t = (title or "").strip()
    if not u or u in seen_urls:
        return
    if not url_is_safe_for_user(u):
        return
    if not t:
        t = u[:80]
    seen_urls.add(u)
    sources.append({"title": t[:240], "url": u[:2048]})


def _sources_from_ddg_json(j: dict[str, Any], sources: list[dict[str, str]], seen_urls: set[str]) -> None:
    if j.get("AbstractURL") and j.get("AbstractText"):
        _add_source(sources, seen_urls, str(j.get("Heading", "Summary")), str(j["AbstractURL"]))
    for t in j.get("RelatedTopics", [])[:12]:
        if not isinstance(t, dict):
            continue
        url = t.get("FirstURL")
        text = t.get("Text")
        if url and text:
            _add_source(sources, seen_urls, str(text).strip(), str(url))
        for sub in t.get("Topics", []) or []:
            if isinstance(sub, dict) and sub.get("FirstURL") and sub.get("Text"):
                _add_source(sources, seen_urls, str(sub["Text"]).strip(), str(sub["FirstURL"]))


def _duckduckgo_html_with_sources(
    query: str,
    sources: list[dict[str, str]],
    seen_urls: set[str],
) -> str:
    """HTML results when the instant-answer API has little text (broader coverage)."""
    body = urllib.parse.urlencode({"q": query}).encode()
    url = "https://html.duckduckgo.com/html/"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
        html_page = resp.read(MAX_WEB_BODY_BYTES).decode("utf-8", errors="replace")
    lines: list[str] = []
    seen_titles: set[str] = set()

    pat_a = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]*)</a>'
    for m in re.finditer(pat_a, html_page, re.I):
        href, raw_title = m.group(1), m.group(2)
        title = _strip_tags(raw_title)
        if len(title) < 3 or title in seen_titles:
            continue
        norm = _normalize_result_url(href)
        if not url_is_safe_for_user(norm):
            continue
        _add_source(sources, seen_urls, title, href)
        seen_titles.add(title)
        lines.append(f"{len(lines) + 1}. {title}\n   URL: {norm}")
        if len(lines) >= 10:
            break

    if len(lines) < 4:
        for pat in (
            r'class="result__title"[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>([^<]+)</a>',
            r'rel="nofollow"[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>',
        ):
            for m in re.finditer(pat, html_page, re.I):
                href, raw_title = m.group(1), m.group(2)
                title = _strip_tags(raw_title)
                if len(title) < 3 or title in seen_titles:
                    continue
                norm = _normalize_result_url(href)
                if not url_is_safe_for_user(norm):
                    continue
                _add_source(sources, seen_urls, title, href)
                seen_titles.add(title)
                lines.append(f"{len(lines) + 1}. {title}\n   URL: {norm}")
                if len(lines) >= 10:
                    break
            if len(lines) >= 4:
                break

    for m in re.finditer(
        r'class="result__snippet"[^>]*>([^<]+(?:<[^/][^<]*</[^>]+>[^<]+)*)',
        html_page,
    ):
        snip = _strip_tags(m.group(1))
        if len(snip) > 40:
            key = snip[:80]
            if key not in seen_titles:
                seen_titles.add(key)
                lines.append(snip)
        if len(lines) >= 14:
            break
    return "\n".join(lines)


def _duckduckgo_html(query: str) -> str:
    """HTML-only result text (no source accumulation)."""
    return _duckduckgo_html_with_sources(query, [], set())


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


def web_search(
    query: str | None = None,
    q: str | None = None,
    search_query: str | None = None,
    topic: str | None = None,
    keywords: str | None = None,
    search: str | None = None,
    text: str | None = None,
    question: str | None = None,
) -> str:
    """Find pointers and short snippets on the open web—use often, not only as a last resort.

    Call when the user asks about news, weather, events, places, products, or anything to verify.
    Use short keyword queries; you may call more than once. Paraphrase for the user in speech;
    do not read backend labels aloud.

    Args:
        query: Main search string (preferred). Same meaning as q, topic, or question.
        q: Alternate name for the search string.
        search_query: Alternate name for the search string.
        topic: Alternate name for the search string.
        keywords: Alternate name for the search string.
        search: Alternate name for the search string.
        text: Alternate name for the search string.
        question: Alternate name for the search string.
    """
    args = {
        "query": query,
        "q": q,
        "search_query": search_query,
        "topic": topic,
        "keywords": keywords,
        "search": search,
        "text": text,
        "question": question,
    }
    raw_q = extract_search_query(args)
    if not raw_q:
        return "No search terms were provided. Call web_search again with a non-empty query."
    qn = raw_q
    if len(qn) > MAX_QUERY_LEN:
        qn = qn[:MAX_QUERY_LEN]

    parts: list[str] = []
    try:
        ddg = _duckduckgo_instant(qn)
        if ddg:
            parts.append("[Quick answer]\n" + ddg)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        parts.append(f"[Quick answer lookup failed: {e}]")

    combined = "\n\n".join(parts)
    if len(combined.strip()) < 160:
        try:
            html_hits = _duckduckgo_html(qn)
            if html_hits:
                parts.append("[Pages found: titles and links]\n" + html_hits)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            parts.append(f"[Page list lookup failed: {e}]")

    combined = "\n\n".join(parts)
    if len(combined.strip()) < 120:
        try:
            wiki = _wikipedia_snippets(qn)
            if wiki:
                parts.append("[Encyclopedia excerpt]\n" + wiki)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
            parts.append(f"[Encyclopedia lookup failed: {e}]")

    out = "\n\n".join(parts).strip()
    if not out:
        return "No useful results for that query. Try different keywords."
    if len(out) > MAX_COMBINED_SEARCH_CHARS:
        return out[: MAX_COMBINED_SEARCH_CHARS - 20] + "\n…[truncated]"
    return out


def _html_page_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]{1,400})</title>", html, re.I | re.DOTALL)
    if not m:
        return ""
    return _strip_tags(m.group(1))[:300]


def _strip_html_to_text(html: str, max_chars: int) -> str:
    t = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", html)
    t = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", t)
    t = re.sub(r"(?is)<noscript[^>]*>.*?</noscript>", " ", t)
    t = _strip_tags(t)
    if len(t) > max_chars:
        cut = t[: max_chars - 40].rsplit(" ", 1)[0]
        t = cut + "\n…[truncated]"
    return t


def read_web_page(
    url: str | None = None,
    link: str | None = None,
    href: str | None = None,
    page_url: str | None = None,
    address: str | None = None,
    website: str | None = None,
    page: str | None = None,
) -> str:
    """Fetch a public web page as read-only text so you can summarize it for the user.

    Use URLs from the latest search/link list or URLs the user gave. This only downloads
    HTML or plain text (no JavaScript execution). Prefer one page per call; call again for
    another link if needed.

    Args:
        url: Page address (https recommended).
        link, href, page_url, address, website, page: Alternate names for the same URL.
    """
    args = {
        "url": url,
        "link": link,
        "href": href,
        "page_url": page_url,
        "address": address,
        "website": website,
        "page": page,
    }
    raw = extract_page_url(args)
    if not raw:
        return "No URL was provided. Call read_web_page again with a full https link."
    u = normalize_user_url(raw)
    if len(u) > MAX_PAGE_URL_LEN:
        return "That URL is too long. Use a shorter link."
    if not url_is_allowed_for_fetch(u):
        return (
            "That link cannot be opened from here (blocked for safety). "
            "Use a normal public https page from your earlier results or the user."
        )
    try:
        req = urllib.request.Request(
            u,
            headers={
                "User-Agent": _USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
            },
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            body = resp.read(MAX_WEB_BODY_BYTES)
    except urllib.error.HTTPError as e:
        return f"The server returned HTTP {e.code} for that page. Try another link or summarize from snippets only."
    except urllib.error.URLError as e:
        return f"Could not load the page ({e.reason!r}). Check the URL or try a different result."
    except (TimeoutError, OSError) as e:
        return f"Loading the page failed ({e}). Try again or use another URL."

    try:
        text_body = body.decode("utf-8", errors="replace")
    except Exception:
        text_body = body.decode("latin-1", errors="replace")

    title = ""
    main: str
    if "json" in ctype:
        try:
            j = json.loads(text_body)
            main = json.dumps(j, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            main = text_body
    elif "html" in ctype or not ctype or ctype == "application/octet-stream":
        title = _html_page_title(text_body)
        main = _strip_html_to_text(text_body, MAX_READ_PAGE_CHARS)
    else:
        main = text_body

    main = main.strip()
    if len(main) > MAX_READ_PAGE_CHARS:
        main = main[: MAX_READ_PAGE_CHARS - 30] + "\n…[truncated]"

    if not main:
        return "The page had almost no readable text (maybe scripts-only or empty). Try another URL."

    head = (
        "Below is read-only text from the page—summarize in natural spoken language for a voice "
        "assistant. Do not quote the search tool by name; speak as if you read the page.\n"
        f"URL: {u}\n"
    )
    if title:
        head += f"Title: {title}\n"
    head += "\n---\n"
    return head + main


def get_current_utc_time() -> str:
    """Return the current date and time in UTC (ISO-style). Use when the user asks for the time or date."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
