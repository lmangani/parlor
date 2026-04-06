"""Allowlist and sanitization for LiteRT-LM tool calls."""

from __future__ import annotations

import copy
import json
from typing import Any

import litert_lm

from agent_tools import MAX_QUERY_LEN, extract_search_query

MAX_TOOL_RESPONSE_CHARS = 14000

_TRANSCRIPTION_KEYS = (
    "transcription",
    "transcript",
    "user_speech",
    "speech",
    "user_message",
    "what_the_user_said",
)
_RESPONSE_KEYS = (
    "response",
    "reply",
    "spoken_response",
    "voice_response",
    "answer",
    "spoken",
)
_DISPLAY_KEYS = (
    "display_context",
    "screen_text",
    "details",
    "notes",
    "context",
    "formatted_context",
)


def coalesce_respond_to_user_fields(args: dict[str, Any]) -> tuple[str, str, str]:
    """Map Gemma / LiteRT variants onto (transcription, response, display_context)."""

    def first_str(keys: tuple[str, ...]) -> str:
        for k in keys:
            v = args.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    return (
        first_str(_TRANSCRIPTION_KEYS),
        first_str(_RESPONSE_KEYS),
        first_str(_DISPLAY_KEYS),
    )


def extract_tool_name(tool_call: dict[str, Any]) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    if isinstance(tool_call.get("name"), str):
        return tool_call["name"]
    fn = tool_call.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str):
        return fn["name"]
    return None


def extract_tool_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        return {}
    merged: dict[str, Any] = {}

    top_args = tool_call.get("args")
    if isinstance(top_args, dict):
        merged.update(top_args)

    raw = tool_call.get("arguments")
    fn = tool_call.get("function")
    if raw is None and isinstance(fn, dict):
        raw = fn.get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                merged.update(parsed)
        except json.JSONDecodeError:
            pass
    elif isinstance(raw, dict):
        merged.update(raw)

    return merged


def _truncate_value(obj: Any, max_str: int) -> Any:
    if isinstance(obj, dict):
        return {k: _truncate_value(v, max_str) for k, v in list(obj.items())[:80]}
    if isinstance(obj, list):
        return [_truncate_value(v, max_str) for v in obj[:80]]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[: max_str - 24] + "\n…[truncated by server]"
    return obj


class ParlorToolPolicy(litert_lm.ToolEventHandler):
    """Approves only registered tool names and caps tool payloads sent back to the model."""

    def __init__(
        self,
        allowed_names: frozenset[str],
        trace: list[str],
    ) -> None:
        self._allowed = allowed_names
        self._trace = trace

    def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
        name = extract_tool_name(tool_call)
        if not name:
            print("Tool call denied: could not parse tool name", tool_call)
            return False
        if name not in self._allowed:
            print(f"Tool call denied (not allowlisted): {name!r}")
            return False

        args = extract_tool_arguments(tool_call)
        if name == "web_search":
            q = extract_search_query(args)
            if not q:
                print(
                    "web_search denied: no usable string in arguments; keys=",
                    list(args.keys()),
                )
                return False
            if len(q) > MAX_QUERY_LEN + 20:
                print("web_search denied: query too long")
                return False

        if name == "respond_to_user":
            _tr, r, d = coalesce_respond_to_user_fields(args)
            # Transcription is optional: Gemma often omits it or uses other keys; never block the turn.
            has_voice = bool(r)
            has_screen = bool(d)
            if not has_voice and not has_screen:
                print(
                    "respond_to_user denied: need non-empty response and/or display_context; keys=",
                    list(args.keys()),
                )
                return False

        self._trace.append(name)
        return True

    def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
        return _truncate_value(copy.deepcopy(tool_response), MAX_TOOL_RESPONSE_CHARS)


def build_optional_tools(
    *,
    enable_web_search: bool,
    enable_utc_time: bool,
    web_search_impl: Any | None = None,
) -> list[Any]:
    """Tool order: web_search before time so the schema highlights search for factual turns."""
    from agent_tools import get_current_utc_time, web_search

    ws = web_search_impl if web_search_impl is not None else web_search
    tools: list[Any] = []
    if enable_web_search:
        tools.append(ws)
    if enable_utc_time:
        tools.append(get_current_utc_time)
    return tools
