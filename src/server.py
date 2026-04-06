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

LLM_TIMEOUT_SEC = float(os.environ.get("PARLOR_LLM_TIMEOUT_SEC", "120"))

SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. "
    "Each user turn includes only what the client sent—do not assume voice, text, or camera if missing. "
    "Turn shapes: (1) typed text only, (2) typed text plus one camera image, (3) voice audio only, "
    "(4) voice audio plus one camera image. Typed text and voice are never both present in the same turn. "
    "You MUST always use the respond_to_user tool to reply. "
    "For voice turns: set transcription to exactly what the user said, then response. "
    "For typed turns: set transcription to a short label such as \"typed message\" or the first few "
    "words of their text, then response."
)

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


def _strip_tool_str(s: str | None) -> str:
    return (s or "").replace('<|"|>', "").strip()


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Per-connection tool state captured via closure
    tool_result = {}

    def respond_to_user(transcription: str, response: str) -> str:
        """Respond to the user's voice message.

        Args:
            transcription: Exact transcription of what the user said in the audio.
            response: Your conversational response to the user. Keep it to 1-4 short sentences.
        """
        tool_result["transcription"] = transcription
        tool_result["response"] = response
        return "OK"

    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=[respond_to_user],
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
                user_text = (msg.get("text") or "").strip()
                has_audio = bool(msg.get("audio"))
                if user_text and has_audio:
                    print("Parlor: text and audio in one message; using typed text only")
                    has_audio = False

                if has_audio:
                    audio_path = save_temp(base64.b64decode(msg["audio"]), ".wav")
                if msg.get("image"):
                    image_path = save_temp(base64.b64decode(msg["image"]), ".jpg")

                content: list[dict] = []
                if user_text and not has_audio:
                    if image_path:
                        content.append({"type": "image", "path": os.path.abspath(image_path)})
                        content.append(
                            {
                                "type": "text",
                                "text": (
                                    "The user sent a typed message and included a current camera image "
                                    "(no voice in this turn). Answer using their text and what you see; "
                                    "do not invent spoken input.\n\n"
                                    + user_text
                                ),
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "text",
                                "text": (
                                    "The user sent a typed message only (no voice and no camera image in this turn). "
                                    "Answer their message; do not invent spoken input.\n\n"
                                    + user_text
                                ),
                            }
                        )
                elif has_audio:
                    content.append({"type": "audio", "path": os.path.abspath(audio_path)})
                    if image_path:
                        content.append({"type": "image", "path": os.path.abspath(image_path)})
                    if audio_path and image_path:
                        content.append(
                            {
                                "type": "text",
                                "text": (
                                    "The user just spoke to you (audio) while showing their camera (image). "
                                    "Respond to what they said, referencing what you see if relevant."
                                ),
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "text",
                                "text": (
                                    "The user just spoke to you (voice only; no camera image in this turn). "
                                    "Respond to what they said; do not describe a scene you were not shown."
                                ),
                            }
                        )
                else:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "text",
                                "text": "Send a voice message or type something in the text box.",
                                "llm_time": 0.0,
                                "error": True,
                            }
                        )
                    )
                    continue

                loop = asyncio.get_event_loop()
                t0 = time.time()
                tool_result.clear()
                try:
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: conversation.send_message({"role": "user", "content": content}),
                        ),
                        timeout=LLM_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    llm_time = time.time() - t0
                    print(f"Parlor: LLM timed out after {llm_time:.2f}s")
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "text",
                                "text": "That took too long and timed out. Try a shorter question.",
                                "llm_time": round(llm_time, 2),
                                "error": True,
                            }
                        )
                    )
                    continue
                except Exception as ex:
                    llm_time = time.time() - t0
                    print(f"Parlor: LLM error: {ex!r}")
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "text",
                                "text": "Something went wrong while thinking. Please try again.",
                                "llm_time": round(llm_time, 2),
                                "error": True,
                            }
                        )
                    )
                    continue

                llm_time = time.time() - t0

                if tool_result:
                    transcription = _strip_tool_str(tool_result.get("transcription"))
                    text_response = _strip_tool_str(tool_result.get("response"))
                    print(f"LLM ({llm_time:.2f}s) [tool] heard: {transcription!r} → {text_response!r}")
                else:
                    transcription = None
                    raw_blocks = response.get("content") or [{}]
                    text_response = _strip_tool_str(
                        raw_blocks[0].get("text") if raw_blocks else ""
                    )
                    print(f"LLM ({llm_time:.2f}s) [no tool]: {text_response!r}")

                if not text_response:
                    text_response = "I could not generate a reply. Please try again."

                if interrupted.is_set():
                    print("Interrupted after LLM, skipping response")
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "text",
                                "text": "",
                                "llm_time": round(llm_time, 2),
                                "cancelled": True,
                            }
                        )
                    )
                    continue

                reply = {"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}
                if transcription:
                    reply["transcription"] = transcription
                await ws.send_text(json.dumps(reply))

                if interrupted.is_set():
                    print("Interrupted before TTS, skipping audio")
                    await ws.send_text(
                        json.dumps({"type": "audio_end", "tts_time": 0.0, "cancelled": True})
                    )
                    continue

                sentences = split_sentences(text_response)
                if not sentences:
                    sentences = [text_response] if text_response else ["Sorry, I had nothing to say."]

                tts_start = time.time()

                await ws.send_text(
                    json.dumps(
                        {
                            "type": "audio_start",
                            "sample_rate": tts_backend.sample_rate,
                            "sentence_count": len(sentences),
                        }
                    )
                )

                for i, sentence in enumerate(sentences):
                    if interrupted.is_set():
                        print(f"Interrupted during TTS (sentence {i+1}/{len(sentences)})")
                        break

                    try:
                        pcm = await asyncio.wait_for(
                            loop.run_in_executor(
                                None, lambda s=sentence: tts_backend.generate(s)
                            ),
                            timeout=60.0,
                        )
                    except asyncio.TimeoutError:
                        print(f"Parlor: TTS timed out on sentence {i+1}")
                        interrupted.set()
                        break

                    if interrupted.is_set():
                        break

                    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_chunk",
                                "audio": base64.b64encode(pcm_int16.tobytes()).decode(),
                                "index": i,
                            }
                        )
                    )

                tts_time = time.time() - tts_start
                print(f"TTS ({tts_time:.2f}s): {len(sentences)} sentences")

                if interrupted.is_set():
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_end",
                                "tts_time": round(tts_time, 2),
                                "cancelled": True,
                            }
                        )
                    )
                else:
                    await ws.send_text(
                        json.dumps(
                            {
                                "type": "audio_end",
                                "tts_time": round(tts_time, 2),
                            }
                        )
                    )

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
