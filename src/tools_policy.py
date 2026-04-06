"""Allowlist and sanitization for LiteRT-LM tool calls."""

from __future__ import annotations

import copy
import json
from typing import Any

import litert_lm

from agent_tools import MAX_QUERY_LEN

MAX_TOOL_RESPONSE_CHARS = 8000


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
    raw = tool_call.get("arguments")
    if raw is None and isinstance(tool_call.get("function"), dict):
        raw = tool_call["function"].get("arguments")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    if isinstance(raw, dict):
        return raw
    return {}


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
            q = args.get("query", "")
            if not isinstance(q, str):
                print("web_search denied: query must be a string")
                return False
            if not q.strip():
                print("web_search denied: empty query")
                return False
            if len(q) > MAX_QUERY_LEN + 20:
                print("web_search denied: query too long")
                return False

        self._trace.append(name)
        return True

    def process_tool_response(self, tool_response: dict[str, Any]) -> dict[str, Any]:
        return _truncate_value(copy.deepcopy(tool_response), MAX_TOOL_RESPONSE_CHARS)


def build_optional_tools(
    *,
    enable_web_search: bool,
    enable_utc_time: bool,
) -> list[Any]:
    """Returns module-level tool callables to pass to create_conversation (order: utilities first)."""
    from agent_tools import get_current_utc_time, web_search

    tools: list[Any] = []
    if enable_utc_time:
        tools.append(get_current_utc_time)
    if enable_web_search:
        tools.append(web_search)
    return tools
