"""Real-time multimodal AI demo with Gemma 4 E2B + Kokoro TTS."""

import asyncio
import base64
import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

import kokoro_onnx
import litert_lm

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.expanduser("~/workspace/LiteRT-LM/run_dir/gemma-4-E2B-it.litertlm"),
)
SYSTEM_PROMPT = (
    "You are a friendly, conversational AI assistant. The user is talking to you "
    "through a microphone and showing you their camera. Respond naturally and "
    "concisely in 1-3 short sentences. Be direct and conversational."
)

app = FastAPI()
engine = None
tts = None


def load_models():
    global engine, tts
    print(f"Loading Gemma 4 E2B from {MODEL_PATH}...")
    engine = litert_lm.Engine(
        MODEL_PATH,
        backend=litert_lm.Backend.CPU,
        vision_backend=litert_lm.Backend.CPU,
        audio_backend=litert_lm.Backend.CPU,
    )
    engine.__enter__()
    print("Engine loaded.")

    tts_dir = Path(__file__).parent
    tts = kokoro_onnx.Kokoro(str(tts_dir / "kokoro-v1.0.onnx"), str(tts_dir / "voices-v1.0.bin"))
    print("Kokoro TTS loaded.")


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    pcm_int16 = (pcm * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


def save_temp(data: bytes, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


@app.on_event("startup")
async def startup():
    await asyncio.get_event_loop().run_in_executor(None, load_models)


@app.get("/")
async def root():
    return HTMLResponse(content=(Path(__file__).parent / "index.html").read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    conversation = engine.create_conversation(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}]
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

                if audio_path and image_path:
                    content.append({"type": "text", "text": "The user just spoke to you (audio) while showing their camera (image). Respond to what they said, referencing what you see if relevant."})
                elif audio_path:
                    content.append({"type": "text", "text": "The user just spoke to you. Respond to what they said."})
                elif image_path:
                    content.append({"type": "text", "text": "The user is showing you their camera. Describe what you see."})
                else:
                    content.append({"type": "text", "text": msg.get("text", "Hello!")})

                # LLM inference
                t0 = time.time()
                response = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: conversation.send_message({"role": "user", "content": content})
                )
                text_response = response["content"][0]["text"]
                llm_time = time.time() - t0
                print(f"LLM ({llm_time:.2f}s): {text_response}")

                if interrupted.is_set():
                    print("Interrupted after LLM, skipping response")
                    continue

                await ws.send_text(json.dumps({"type": "text", "text": text_response, "llm_time": round(llm_time, 2)}))

                if interrupted.is_set():
                    print("Interrupted before TTS, skipping audio")
                    continue

                # TTS
                t0 = time.time()
                wav_bytes = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: pcm_to_wav_bytes(*tts.create(text_response, voice="af_heart", speed=1.1))
                )
                tts_time = time.time() - t0
                print(f"TTS ({tts_time:.2f}s): {len(wav_bytes)} bytes")

                if interrupted.is_set():
                    print("Interrupted after TTS, skipping audio send")
                    continue

                await ws.send_text(json.dumps({"type": "audio", "audio": base64.b64encode(wav_bytes).decode(), "tts_time": round(tts_time, 2)}))

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
