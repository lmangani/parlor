"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import functools
import json
from json.decoder import JSONDecoder
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import litert_lm
import tts
import agent_tools
from tools_policy import (
    ParlorToolPolicy,
    build_optional_tools,
    coalesce_respond_to_user_fields,
    merge_json_blob_arg,
    normalize_respond_to_user_merged_args,
)

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "")
    if path:
        return path
    from huggingface_hub import hf_hub_download
    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()


def _optional_tool_flags() -> tuple[bool, bool, bool]:
    """PARLOR_TOOLS: comma-separated subset, or 'none'. Default: lookup + read page + time."""
    raw = os.environ.get(
        "PARLOR_TOOLS", "web_search,read_web_page,get_current_utc_time"
    ).strip().lower()
    if raw == "none":
        return False, False, False
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return (
        "web_search" in parts,
        "read_web_page" in parts,
        "get_current_utc_time" in parts,
    )


ENABLE_WEB_SEARCH, ENABLE_READ_WEB_PAGE, ENABLE_UTC_TIME = _optional_tool_flags()

SEARCH_MEMORY_CAP = int(os.environ.get("PARLOR_SEARCH_MEMORY_CHARS", "9000"))


def _make_web_search_with_memory(memory: dict[str, str]):
    """Wrap web_search to retain the latest query + body for injected session context."""

    @functools.wraps(agent_tools.web_search)
    def web_search_with_memory(**kwargs):
        out = agent_tools.web_search(**kwargs)
        if isinstance(out, str) and out.strip():
            q = agent_tools.extract_search_query(kwargs)
            q = _strip_model_text(q) if q else ""
            memory["query"] = (q or "—")[: agent_tools.MAX_QUERY_LEN]
            memory["text"] = out[:SEARCH_MEMORY_CAP]
        return out

    return web_search_with_memory


def _user_text_search_memory_append(memory: dict[str, str]) -> str:
    """Re-inject last search results into every user turn (model does not read server logs)."""
    t = (memory.get("text") or "").strip()
    if not t:
        return ""
    q = (memory.get("query") or "").strip()
    return (
        "\n\n---\n[Latest lookup snapshot for this session — ground truth for follow-ups: "
        "numbered results, first vs second link, hostnames, paths, snippets.]\n"
        f"Query: {q}\n---\n{t}\n---\n"
        "If the user wants details from one of those links, call read_web_page with that https URL, "
        "then summarize in your own words (voice-first). "
        "If something is still missing or they want fresher facts, run another lookup, then "
        "respond_to_user. Never invent URLs.\n---"
    )


def build_system_prompt() -> str:
    lines = [
        "You are a capable voice assistant (like an IDE copilot, but for spoken conversation). "
        "The user talks through a microphone and may show their camera. They also read a text "
        "transcript on screen while they listen.",
        "",
        "Work like a good agent: use tools, read results internally, decide what matters, then "
        "choose how much to say aloud vs what to show as text. The user cannot see your raw tool "
        "logs—only what you put in respond_to_user.",
        "",
        "Voice UX: speak naturally (e.g. 'here is what I found'). Do not name search engines, "
        "vendors, or tools to the user unless they ask how you work.",
        "",
        "Tools:",
    ]
    if ENABLE_WEB_SEARCH:
        lines.extend(
            [
                "- web_search: Find fresh pointers and short snippets on the open web when the user "
                "asks about news, weather, events, places, products, or anything you should verify. "
                "Use short keyword queries; you may call more than once. Results are for you to read "
                "and paraphrase—never read tool labels aloud.",
                "",
            ]
        )
    if ENABLE_READ_WEB_PAGE:
        lines.extend(
            [
                "- read_web_page: Load one public https page as plain text (no scripts run) so you can "
                "answer from the actual article or listing. Use a full URL from the latest lookup "
                "snapshot or one the user gave you. Call once per page; then summarize in speech. "
                "If a URL is blocked or empty, say so briefly and offer alternatives.",
                "",
            ]
        )
    if ENABLE_UTC_TIME:
        lines.append(
            "- get_current_utc_time: use when the user asks for the current date, day, or time."
        )
    lines.extend(
        [
            "- respond_to_user: REQUIRED last step. Use THREE plain string fields (like web_search uses "
            "plain text in value—NOT JSON). LiteRT wraps each field in <|\"|>…<|\"|>; JSON inside a "
            "field breaks the parser because quote characters clash with the wrapper.",
            "  Fields (each optional except you must fill response and/or display_context):",
            "  • response — short text for TTS (1–4 sentences).",
            "  • display_context — bullets, URLs, search listings for the on-screen transcript.",
            "  • transcription — what the user said, if known; else leave empty.",
            "  SYNTAX (copy the shape exactly): after each field name there must be a colon, then "
            "<|\"|>, then your text, then <|\"|>, then a comma before the next field name:",
            "  call:respond_to_user{response:<|\"|>Spoken words here.<|\"|>,display_context:<|\"|>Screen text "
            "with URLs and bullets.<|\"|>,transcription:<|\"|><|\"|>}",
            "  Rules: (1) Never write JSON braces { } or JSON quotes around keys inside these fields. "
            "(2) After every comma the next token must be fieldname:<|\"|> — e.g. "
            ",display_context:<|\"|> not ,display_context<|\"|>. "
            "(3) Avoid the ASCII double-quote character inside values; use apostrophes for emphasis. "
            "(4) Exactly one respond_to_user per turn—no code block: prefix, no duplicated call.",
            "",
        ]
    )
    if ENABLE_WEB_SEARCH:
        lines.extend(
            [
                "Session memory: The user message may include a [Latest lookup snapshot] block from "
                "the most recent successful web_search. Prefer it for follow-ups (first link, exact "
                "URLs, order). For 'what does that page say?' use read_web_page on the chosen https "
                "URL, then respond_to_user.",
                "",
                "Follow-ups: Combine snapshot + chat. If you need newer or different pointers, "
                "web_search again; if you need body text from one link, read_web_page, then "
                "respond_to_user.",
                "",
                "Workflow: When you need pointers, web_search; when you need to read a specific page, "
                "read_web_page; then always respond_to_user. Small talk: respond_to_user only. "
                "Never output plain assistant text without respond_to_user.",
            ]
        )
    elif ENABLE_READ_WEB_PAGE:
        lines.extend(
            [
                "Workflow: When the user shares an https link or wants details from a specific page, "
                "call read_web_page, then respond_to_user. Otherwise respond_to_user only. "
                "Never output plain assistant text without respond_to_user.",
            ]
        )
    else:
        lines.append(
            "Workflow: Call respond_to_user with what they said and your answer. "
            "Never output a normal assistant message without using respond_to_user."
        )
    return "\n".join(lines)


SYSTEM_PROMPT = build_system_prompt()

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

engine = None
tts_backend = None


def load_models():
    global engine, tts_backend
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.GPU,
        vision_backend=litert_lm.Backend.GPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")
    print(
        f"Agent tools (env PARLOR_TOOLS): web_search={ENABLE_WEB_SEARCH}, "
        f"read_web_page={ENABLE_READ_WEB_PAGE}, get_current_utc_time={ENABLE_UTC_TIME}"
    )

    tts_backend = tts.load()


def unload_models():
    global engine
    if engine is not None:
        engine.__exit__(None, None, None)


@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, load_models)
    try:
        yield
    finally:
        await loop.run_in_executor(None, unload_models)


app = FastAPI(lifespan=lifespan)


def save_temp(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming TTS."""
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _strip_model_text(s: str | None) -> str:
    return str(s or "").replace('<|"|>', "").strip()


def _looks_like_leaked_tool_output(s: str | None) -> bool:
    if not s or not str(s).strip():
        return False
    t = str(s)
    needles = (
        "<|tool_call",
        "<tool_call",
        "call:respond_to_user",
        "<|channel|>",
        "display_context:<|",
        "<tool_response",
    )
    return any(n in t for n in needles)


def _sanitize_parlor_output(s: str | None) -> str:
    """Strip Gemma/LiteRT tool fragments so they are never shown or spoken."""
    if not s:
        return ""
    t = _strip_model_text(s)
    cut = t.find("<|tool_call")
    if cut != -1:
        t = t[:cut].rstrip()
    cut = t.find("call:respond_to_user{")
    if cut != -1:
        t = t[:cut].rstrip()
    t = re.sub(r"<\|[^|]*\|>", "", t)
    t = re.sub(r"<tool_call[^>]*>", "", t, flags=re.I)
    t = re.sub(r"</?tool_call\|?>", "", t, flags=re.I)
    return t.strip()


def _scrub_tracking_urls(s: str) -> str:
    """Remove ad/tracking URLs the model may have echoed (never speak or show these)."""

    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        u = raw.rstrip(").,;]\"'")
        return raw if agent_tools.url_is_safe_for_user(u) else ""

    return re.sub(r"https?://[^\s<>'\"]+", repl, s)


_CONVERSATION_RESET_MESSAGE = (
    "I hit a snag formatting that answer. Let's continue—what would you like to know?"
)

_EMPTY_ASSISTANT_RETRY_MESSAGE = (
    "I didn't quite catch that—could you say it again in a few words? "
    "For example, 'pick the first recipe you found,' or 'read out that link.'"
)

_LITERT_TOOL_PARSE_MESSAGE = (
    "Something went wrong formatting the reply. Say your question once more—I'll answer without "
    "repeating that step."
)


def _recover_tool_result_from_litert_error(err: str) -> dict[str, str] | None:
    """When LiteRT fails to parse FC, the error text often includes the raw call. Salvage JSON
    from broken `turn:<|\"|>{...}` output (model omits closing <|\"|> or lexer chokes on inner quotes).
    """
    if "respond_to_user" not in err or "turn:" not in err:
        return None
    markers = (
        "call:respond_to_user{turn:<|\"|>",
        "respond_to_user{turn:<|\"|>",
    )
    for m in markers:
        idx = err.find(m)
        if idx < 0:
            continue
        sub = err[idx + len(m) :]
        brace = sub.find("{")
        if brace < 0:
            continue
        dec = JSONDecoder()
        try:
            obj, _ = dec.raw_decode(sub, brace)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        flat = normalize_respond_to_user_merged_args(
            {k: v for k, v in obj.items() if isinstance(v, (str, int, float))}
        )
        tr, r, d = coalesce_respond_to_user_fields(flat)
        if not (r or d):
            continue
        return {"transcription": tr, "response": r, "display_context": d}
    return None


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}
    tool_trace: list[str] = []
    search_memory: dict[str, str] = {"query": "", "text": ""}

    def respond_to_user(
        response: str = "",
        display_context: str = "",
        transcription: str = "",
        turn: str = "",
        payload: str = "",
    ) -> str:
        """Plain string args; optional turn/payload JSON for legacy model output (expanded in policy)."""
        args = merge_json_blob_arg(
            {
                "response": response,
                "display_context": display_context,
                "transcription": transcription,
                "turn": turn,
                "payload": payload,
            },
            ("turn", "payload"),
        )
        args = normalize_respond_to_user_merged_args(args)
        tr, r, d = coalesce_respond_to_user_fields(args)
        tool_result["transcription"] = tr
        tool_result["response"] = r
        tool_result["display_context"] = d
        return "OK"

    web_search_impl = (
        _make_web_search_with_memory(search_memory) if ENABLE_WEB_SEARCH else None
    )
    optional = build_optional_tools(
        enable_web_search=ENABLE_WEB_SEARCH,
        enable_read_web_page=ENABLE_READ_WEB_PAGE,
        enable_utc_time=ENABLE_UTC_TIME,
        web_search_impl=web_search_impl,
    )
    tool_functions = [*optional, respond_to_user]
    allowed_names = frozenset({f.__name__ for f in tool_functions})
    tool_policy = ParlorToolPolicy(allowed_names, tool_trace)

    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=tool_functions,
        tool_event_handler=tool_policy,
    )
    conversation.__enter__()

    def reopen_conversation() -> None:
        nonlocal conversation
        try:
            conversation.__exit__(None, None, None)
        except Exception as ex:
            print(f"Parlor: conversation __exit__ during reset: {ex}")
        conversation = engine.create_conversation(
            messages=[{"role": "system", "content": SYSTEM_PROMPT}],
            tools=tool_functions,
            tool_event_handler=tool_policy,
        )
        conversation.__enter__()

    interrupted = asyncio.Event()
    msg_queue = asyncio.Queue()

    async def receiver():
        """Receive messages from WebSocket and route them."""
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "interrupt":
                    interrupted.set()
                    print("Client interrupted")
                else:
                    await msg_queue.put(msg)
        except WebSocketDisconnect:
            await msg_queue.put(None)

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            msg = await msg_queue.get()
            if msg is None:
                break

            audio_path = image_path = None
            interrupted.clear()

            try:
                if msg.get("audio"):
                    audio_path = save_temp(base64.b64decode(msg["audio"]), ".wav")
                if msg.get("image"):
                    image_path = save_temp(base64.b64decode(msg["image"]), ".jpg")

                # Build multimodal content
                content = []
                if audio_path:
                    content.append({"type": "audio", "path": os.path.abspath(audio_path)})
                if image_path:
                    content.append({"type": "image", "path": os.path.abspath(image_path)})

                memory_append = _user_text_search_memory_append(search_memory)
                search_hint = ""
                if ENABLE_WEB_SEARCH:
                    search_hint = (
                        " If this message includes [Latest lookup snapshot], use it for follow-ups "
                        "(first link, exact URLs). For what a specific page says, call read_web_page "
                        "with that https URL when available, then respond_to_user. "
                        "If you need new pointers, call web_search first. "
                        "Put URLs and bullets in display_context; keep voice concise and natural."
                    )
                elif ENABLE_READ_WEB_PAGE:
                    search_hint = (
                        " For page details, call read_web_page with a full https URL the user gave you, "
                        "then respond_to_user."
                    )
                tail = memory_append + search_hint

                if audio_path and image_path:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "The user just spoke to you (audio) while showing their camera (image). "
                                "Respond to what they said, referencing what you see if relevant."
                                + tail
                            ),
                        }
                    )
                elif audio_path:
                    content.append(
                        {
                            "type": "text",
                            "text": "The user just spoke to you. Respond to what they said." + tail,
                        }
                    )
                elif image_path:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "The user is showing you their camera. Describe what you see." + tail
                            ),
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "text",
                            "text": (msg.get("text", "Hello!") + tail),
                        }
                    )

                # LLM inference
                t0 = time.time()
                tool_result.clear()
                tool_trace.clear()
                response = None
                litert_error: str | None = None
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: conversation.send_message({"role": "user", "content": content})
                    )
                except RuntimeError as ex:
                    litert_error = str(ex)
                    print(f"Parlor: LiteRT send_message RuntimeError: {litert_error[:800]}")
                llm_time = time.time() - t0

                # Extract response from tool call or fallback to raw text
                skip_audio = False
                display_ctx = ""
                transcription = None
                needs_conversation_reset = False
                recovered_from_litert_parse = False

                if litert_error is not None:
                    recovered = _recover_tool_result_from_litert_error(litert_error)
                    if recovered:
                        tool_result.update(recovered)
                        tool_trace.append("respond_to_user(recovered)")
                        recovered_from_litert_parse = True
                        print(
                            f"LLM ({llm_time:.2f}s) [recovered] respond_to_user from LiteRT error text"
                        )
                        litert_error = None

                if litert_error is not None:
                    needs_conversation_reset = True
                    text_response = _LITERT_TOOL_PARSE_MESSAGE
                    print(f"LLM ({llm_time:.2f}s) [error] tool parse / runtime failure")
                elif tool_result:
                    transcription = _strip_model_text(tool_result.get("transcription", ""))
                    if _looks_like_leaked_tool_output(transcription):
                        transcription = ""

                    spoken_raw = _strip_model_text(tool_result.get("response", ""))
                    display_ctx = _strip_model_text(tool_result.get("display_context", ""))
                    if _looks_like_leaked_tool_output(spoken_raw):
                        spoken_raw = ""
                    if _looks_like_leaked_tool_output(display_ctx):
                        display_ctx = ""

                    text_response = _scrub_tracking_urls(_sanitize_parlor_output(spoken_raw))
                    display_ctx = _scrub_tracking_urls(_sanitize_parlor_output(display_ctx))

                    if not text_response and not display_ctx:
                        had_tool_leak = (
                            _looks_like_leaked_tool_output(tool_result.get("response"))
                            or _looks_like_leaked_tool_output(tool_result.get("display_context"))
                            or _looks_like_leaked_tool_output(tool_result.get("transcription"))
                        )
                        if had_tool_leak:
                            needs_conversation_reset = True
                            text_response = _CONVERSATION_RESET_MESSAGE
                        else:
                            text_response = _EMPTY_ASSISTANT_RETRY_MESSAGE
                        skip_audio = False
                    else:
                        skip_audio = not text_response and bool(display_ctx)

                    print(
                        f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → "
                        f"spoken={text_response[:120]!r} display_len={len(display_ctx)} "
                        f"skip_audio={skip_audio}"
                    )
                else:
                    if not response:
                        text_response = _EMPTY_ASSISTANT_RETRY_MESSAGE
                        print(f"LLM ({llm_time:.2f}s) [no response object]")
                    else:
                        raw_text = response["content"][0]["text"]
                        if _looks_like_leaked_tool_output(raw_text):
                            print(
                                "Parlor: leaked tool syntax in assistant text; resetting conversation"
                            )
                            needs_conversation_reset = True
                            text_response = _CONVERSATION_RESET_MESSAGE
                        else:
                            text_response = _scrub_tracking_urls(_sanitize_parlor_output(raw_text))
                            if not text_response.strip():
                                # Empty or token-only decode (e.g. <|"|>)—keep conversation history.
                                text_response = _EMPTY_ASSISTANT_RETRY_MESSAGE
                        print(f"LLM ({llm_time:.2f}s) [no tool] raw head: {raw_text[:160]!r}…")

                if needs_conversation_reset or recovered_from_litert_parse:
                    reopen_conversation()

                if interrupted.is_set():
                    print("Interrupted after LLM, skipping response")
                    continue

                reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
                if transcription:
                    reply["transcription"] = transcription
                if display_ctx:
                    reply["display_context"] = display_ctx
                if skip_audio:
                    reply["skip_audio"] = True
                if tool_trace:
                    reply["tools_used"] = list(tool_trace)
                await ws.send_text(json.dumps(reply))

                if interrupted.is_set():
                    print("Interrupted before TTS, skipping audio")
                    continue

                if skip_audio:
                    continue

                # Streaming TTS: split into sentences and send chunks progressively
                sentences = split_sentences(text_response)
                if not sentences:
                    sentences = [text_response]

                tts_start = time.time()

                # Signal start of audio stream
                await ws.send_text(json.dumps({
                    "type": "audio_start",
                    "sample_rate": tts_backend.sample_rate,
                    "sentence_count": len(sentences),
                }))

                for i, sentence in enumerate(sentences):
                    if interrupted.is_set():
                        print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                        break

                    # Generate audio for this sentence
                    pcm = await asyncio.get_event_loop().run_in_executor(
                        None, lambda s=sentence: tts_backend.generate(s)
                    )

                    if interrupted.is_set():
                        break

                    # Convert to 16-bit PCM and send as base64
                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(json.dumps({
                        "type": "audio_chunk",
                        "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                        "index": i,
                    }))

                tts_time = time.time() - tts_start
                print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

                if not interrupted.is_set():
                    await ws.send_text(json.dumps({
                        "type": "audio_end",
                        "tts_time": round(tts_time, 2),
                    }))

            finally:
                for p in [audio_path, image_path]:
                    if p and os.path.exists(p):
                        os.unlink(p)

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        recv_task.cancel()
        conversation.__exit__(None, None, None)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
