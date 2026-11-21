#!/usr/bin/env python3
"""Minimal OpenAI-compatible TTS server for official Chatterbox Multilingual.

Used by the MoneyPrinterTurbo director fork. Loads Resemble AI's
ChatterboxMultilingualTTS (Spanish via language_id=es) and clones a voice
from a reference wav when audio_prompt_path is set.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/audio/speech
  POST /free
"""

from __future__ import annotations

import gc
import io
import json
import os
import subprocess
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = os.environ.get("CHATTERBOX_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHATTERBOX_PORT", "4123"))
KEEP_ALIVE = float(os.environ.get("CHATTERBOX_KEEP_ALIVE", "120"))
DEFAULT_LANGUAGE = os.environ.get("CHATTERBOX_LANGUAGE", "es")
VOICE_REF_FILE = Path(
    os.environ.get(
        "CHATTERBOX_VOICE_REF_FILE",
        str(Path.home() / "Videos/chatterbox/voice_ref.txt"),
    )
)

_lock = threading.Lock()
_model = None
_device = "cpu"
_idle_timer: threading.Timer | None = None


def _log(message: str) -> None:
    print(message, flush=True)


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _cancel_idle_timer() -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None


def unload_model() -> None:
    global _model
    with _lock:
        _cancel_idle_timer()
        if _model is None:
            return
        try:
            del _model
        except Exception:
            pass
        _model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
        _log("chatterbox model unloaded, GPU cache cleared")


def _schedule_unload() -> None:
    global _idle_timer
    _cancel_idle_timer()
    if KEEP_ALIVE <= 0:
        unload_model()
        return
    _idle_timer = threading.Timer(KEEP_ALIVE, unload_model)
    _idle_timer.daemon = True
    _idle_timer.start()


def load_model():
    global _model, _device
    with _lock:
        if _model is not None:
            return _model
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        _device = _pick_device()
        _log(f"loading Chatterbox Multilingual on {_device} (t3_model=v3)")
        try:
            _model = ChatterboxMultilingualTTS.from_pretrained(
                device=_device, t3_model="v3"
            )
        except TypeError:
            _log("t3_model=v3 not supported by this package, loading default multilingual")
            _model = ChatterboxMultilingualTTS.from_pretrained(device=_device)
        _log(f"model ready, sr={getattr(_model, 'sr', '?')}")
        return _model


def _current_voice_ref() -> str:
    if VOICE_REF_FILE.is_file():
        path = VOICE_REF_FILE.read_text(encoding="utf-8").strip()
        if path and Path(path).is_file():
            return path
    return ""


def _resolve_prompt_path(payload: dict) -> str | None:
    for key in ("audio_prompt_path", "audio_prompt", "reference_audio"):
        raw = str(payload.get(key) or "").strip()
        if raw and Path(raw).is_file():
            return raw
    voice = str(payload.get("voice") or "").strip()
    if voice and Path(voice).is_file():
        return voice
    if voice.lower() in {"cloned", "clonada", "default", "director"}:
        fallback = _current_voice_ref()
        return fallback or None
    fallback = _current_voice_ref()
    return fallback or None


def _wav_to_format(wav_path: Path, response_format: str) -> tuple[bytes, str]:
    fmt = (response_format or "mp3").lower().strip()
    if fmt in {"wav", "pcm"}:
        return wav_path.read_bytes(), "audio/wav"
    dest = wav_path.with_suffix(f".{fmt}")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-vn",
        "-ac",
        "1",
    ]
    if fmt == "mp3":
        command += ["-codec:a", "libmp3lame", "-q:a", "2", str(dest)]
        mime = "audio/mpeg"
    elif fmt == "opus":
        command += ["-c:a", "libopus", str(dest)]
        mime = "audio/opus"
    elif fmt == "flac":
        command += [str(dest)]
        mime = "audio/flac"
    else:
        command += ["-codec:a", "libmp3lame", "-q:a", "2", str(dest)]
        mime = "audio/mpeg"
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file():
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg failed")[-400:])
    data = dest.read_bytes()
    dest.unlink(missing_ok=True)
    return data, mime


def synthesize(payload: dict) -> tuple[bytes, str]:
    text = str(payload.get("input") or payload.get("text") or "").strip()
    if not text:
        raise ValueError("missing input text")
    language = str(payload.get("language") or payload.get("language_id") or DEFAULT_LANGUAGE).strip() or "es"
    if "-" in language:
        language = language.split("-", 1)[0].lower()
    prompt_path = _resolve_prompt_path(payload)
    model = load_model()
    kwargs = {"language_id": language}
    if prompt_path:
        kwargs["audio_prompt_path"] = prompt_path
        _log(f"synthesize lang={language} ref={prompt_path} chars={len(text)}")
    else:
        _log(f"synthesize lang={language} (no voice ref) chars={len(text)}")
    try:
        wav = model.generate(text, **kwargs)
    except TypeError:
        wav = model.generate(text, language_id=language)
    sr = int(getattr(model, "sr", 24000) or 24000)
    import torchaudio

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "out.wav"
        # generate may return a tensor (C, T) or (T,)
        tensor = wav.detach().cpu() if hasattr(wav, "detach") else wav
        if getattr(tensor, "dim", lambda: 1)() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(str(wav_path), tensor, sr)
        return _wav_to_format(wav_path, str(payload.get("response_format") or "mp3"))


class Handler(BaseHTTPRequestHandler):
    server_version = "ChatterboxDirector/1.0"

    def log_message(self, fmt: str, *args) -> None:
        _log("%s - %s" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"/health", "/v1/health"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "model_loaded": _model is not None,
                    "device": _device if _model is not None else _pick_device(),
                    "voice_ref": _current_voice_ref() or None,
                },
            )
            return
        if path in {"/v1/models", "/models"}:
            self._send_json(
                200,
                {
                    "data": [
                        {"id": "chatterbox", "object": "model"},
                        {"id": "chatterbox-multilingual-es", "object": "model"},
                    ]
                },
            )
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if path in {"/free", "/v1/free"}:
            unload_model()
            self._send_json(200, {"status": "freed"})
            return
        if path not in {"/v1/audio/speech", "/audio/speech"}:
            self._send_json(404, {"error": "not found"})
            return
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            audio, mime = synthesize(payload)
            self._send(200, audio, mime)
        except Exception as exc:
            _log("synthesis failed:\n" + traceback.format_exc())
            self._send_json(500, {"error": str(exc)})
        finally:
            _schedule_unload()


def main() -> None:
    VOICE_REF_FILE.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    _log(f"Chatterbox director server on http://{HOST}:{PORT} keep_alive={KEEP_ALIVE}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        unload_model()
        server.server_close()


if __name__ == "__main__":
    main()
