"""Parlor — on-device, real-time multimodal AI (voice + vision)."""

import asyncio
import base64
import json
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
from tools_policy import ParlorToolPolicy, build_optional_tools

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


def _optional_tool_flags() -> tuple[bool, bool]:
    """PARLOR_TOOLS: comma-separated subset, or 'none', default enables web + time."""
    raw = os.environ.get("PARLOR_TOOLS", "web_search,get_current_utc_time").strip().lower()
    if raw == "none":
        return False, False
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    return "web_search" in parts, "get_current_utc_time" in parts


ENABLE_WEB_SEARCH, ENABLE_UTC_TIME = _optional_tool_flags()


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
        "Tools:",
    ]
    if ENABLE_WEB_SEARCH:
        lines.extend(
            [
                "- web_search: Your main way to learn about the real world. Use it whenever the user "
                "asks about news, weather, sports, stocks, people, places, products, events, "
                '"what happened", "who won", definitions, or anything you are not 100% sure about. '
                "Prefer searching over inventing facts. Pass a short keyword query (you may call "
                "it more than once with different queries).",
                "",
            ]
        )
    if ENABLE_UTC_TIME:
        lines.append(
            "- get_current_utc_time: use when the user asks for the current date, day, or time."
        )
    lines.extend(
        [
            "- respond_to_user: REQUIRED last step every turn. Fields:",
            "  • transcription: exact words the user said in the audio.",
            "  • response: what you want spoken (TTS)—short, natural, 1–4 sentences for the ear. "
            "Leave empty when you prefer text-only for this turn: if display_context is non-empty, "
            "no audio is played and the user reads the transcript.",
            "  • display_context: optional longer text shown in the transcript in parallel with TTS. "
            "Put search summaries, bullet lists, URLs, numbered sources, step-by-step notes, or "
            "anything too dense for voice. The user reads this while or after hearing response.",
            "  Use display_context whenever tools returned detail worth keeping (especially web_search). "
            "You may speak a brief overview in response and put the full breakdown in display_context.",
            "",
        ]
    )
    if ENABLE_WEB_SEARCH:
        lines.append(
            "Workflow: If the user's request benefits from online facts, call web_search first, read "
            "the tool result, then call respond_to_user with what they said and your answer. "
            "For pure small talk with no factual lookup, you may call respond_to_user directly. "
            "Never output a normal assistant message without using respond_to_user."
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
        f"get_current_utc_time={ENABLE_UTC_TIME}"
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


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}
    tool_trace: list[str] = []

    def respond_to_user(
        transcription: str,
        response: str = "",
        display_context: str = "",
    ) -> str:
        """Deliver the turn: what to speak (TTS) and/or what to show on screen.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Spoken reply (1–4 short sentences for TTS). May be empty if you only
                want on-screen text; a short bridge line may be spoken instead.
            display_context: Longer text for the transcript: tool summaries, URLs, bullets,
                structured notes—shown in parallel with audio so the user can read details.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        tool_result["display_context"] = display_context
        return "OK"

    optional = build_optional_tools(
        enable_web_search=ENABLE_WEB_SEARCH,
        enable_utc_time=ENABLE_UTC_TIME,
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

                search_hint = ""
                if ENABLE_WEB_SEARCH:
                    search_hint = (
                        " If their question needs real-world facts (news, people, places, scores, "
                        "weather, dates, or anything to verify online), call web_search with a "
                        "short query before you call respond_to_user. Summarize what you learned in "
                        "display_context (links, bullets) and keep response short for voice."
                    )

                if audio_path and image_path:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "The user just spoke to you (audio) while showing their camera (image). "
                                "Respond to what they said, referencing what you see if relevant."
                                + search_hint
                            ),
                        }
                    )
                elif audio_path:
                    content.append(
                        {
                            "type": "text",
                            "text": "The user just spoke to you. Respond to what they said." + search_hint,
                        }
                    )
                elif image_path:
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "The user is showing you their camera. Describe what you see."
                                + search_hint
                            ),
                        }
                    )
                else:
                    content.append({"type": "text", "text": msg.get("text", "Hello!")})

                # LLM inference
                t0 = time.time()
                tool_result.clear()
                tool_trace.clear()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: conversation.send_message({"role": "user", "content": content})
                )
                llm_time = time.time() - t0

                # Extract response from tool call or fallback to raw text
                skip_audio = False
                if tool_result:
                    transcription = _strip_model_text(tool_result.get("transcription", ""))
                    spoken_raw = _strip_model_text(tool_result.get("response", ""))
                    display_ctx = _strip_model_text(tool_result.get("display_context", ""))
                    # Voice-only empty + on-screen detail => no TTS (read the transcript).
                    skip_audio = not spoken_raw and bool(display_ctx)
                    text_response = spoken_raw
                    print(
                        f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → "
                        f"spoken={text_response[:120]!r} display_len={len(display_ctx)} "
                        f"skip_audio={skip_audio}"
                    )
                else:
                    transcription = None
                    text_response = response["content"][0]["text"]
                    print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response}")

                if interrupted.is_set():
                    print("Interrupted after LLM, skipping response")
                    continue

                reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
                if transcription:
                    reply["transcription"] = transcription
                dc = _strip_model_text(tool_result.get("display_context", ""))
                if dc:
                    reply["display_context"] = dc
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
