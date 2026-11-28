#!/usr/bin/env python3
"""Local MuseTalk 1.5 lip-sync HTTP server (same idea as Chatterbox)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOME = Path(os.environ.get("MUSETALK_HOME", Path.home() / "Videos/musetalk")).expanduser()
HOST = os.environ.get("MUSETALK_HOST", "127.0.0.1")
PORT = int(os.environ.get("MUSETALK_PORT", "4124"))
KEEP_ALIVE = int(os.environ.get("MUSETALK_KEEP_ALIVE", "120"))
PYTHON = HOME / ".venv" / "bin" / "python"
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "/usr/bin")

_lock = threading.Lock()
_last_used = time.time()
# This server only shells out; nothing stays on the GPU in this process.
# keep_alive is still honored so the parent can treat it like Chatterbox.


def _run_inference(video: Path, audio: Path, output: Path) -> Path:
    if not PYTHON.is_file():
        raise RuntimeError(f"MuseTalk venv not found: {PYTHON}")
    work = HOME / "work" / uuid.uuid4().hex[:10]
    work.mkdir(parents=True, exist_ok=True)
    config = work / "task.yaml"
    result_dir = work / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    # Official inference.ffmpeg commands do not quote paths; use this work dir.
    video_link = work / f"input{video.suffix.lower() or '.mp4'}"
    audio_link = work / f"audio{audio.suffix.lower() or '.wav'}"
    shutil.copy2(video, video_link)
    shutil.copy2(audio, audio_link)
    config.write_text(
        f'task_0:\n  video_path: "{video_link}"\n  audio_path: "{audio_link}"\n',
        encoding="utf-8",
    )
    log_path = work / "infer.log"
    cmd = [
        str(PYTHON),
        "-m",
        "scripts.inference",
        "--inference_config",
        str(config),
        "--result_dir",
        str(result_dir),
        "--unet_model_path",
        str(HOME / "models/musetalkV15/unet.pth"),
        "--unet_config",
        str(HOME / "models/musetalkV15/musetalk.json"),
        "--version",
        "v15",
        "--ffmpeg_path",
        FFMPEG_PATH,
        "--use_float16",
        "--batch_size",
        "4",
    ]
    try:
        with log_path.open("wb") as log:
            proc = subprocess.run(
                cmd,
                cwd=str(HOME),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-800:]
            raise RuntimeError(f"MuseTalk salió con código {proc.returncode}: {tail}")
        videos = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
        videos = [path for path in videos if path.is_file() and path.stat().st_size > 1000]
        if not videos:
            tail = log_path.read_text(encoding="utf-8", errors="ignore")[-800:]
            raise RuntimeError(f"MuseTalk no dejó un mp4: {tail}")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(videos[-1], output)
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[musetalk] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "home": str(HOME)})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        global _last_used
        path = self.path.rstrip("/")
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            payload = {}
        if path == "/free":
            self._json(200, {"freed": True})
            return
        if path != "/lipsync":
            self._json(404, {"error": "not found"})
            return
        video = Path(str(payload.get("video") or "")).expanduser()
        audio = Path(str(payload.get("audio") or "")).expanduser()
        output = Path(str(payload.get("output") or "")).expanduser()
        if not video.is_file() or not audio.is_file() or not str(output):
            self._json(400, {"error": "video, audio y output son obligatorios"})
            return
        with _lock:
            try:
                started = time.time()
                _run_inference(video, audio, output)
                _last_used = time.time()
                self._json(
                    200,
                    {
                        "ok": True,
                        "output": str(output),
                        "seconds": round(time.time() - started, 2),
                    },
                )
            except Exception as exc:
                traceback.print_exc()
                self._json(500, {"error": str(exc)})


def _idle_watcher():
    # Placeholder so keep_alive is configurable; the worker is a child process
    # that already frees the GPU when it exits.
    while True:
        time.sleep(max(30, KEEP_ALIVE))


def main():
    if not HOME.is_dir():
        raise SystemExit(f"MUSETALK_HOME no existe: {HOME}")
    thread = threading.Thread(target=_idle_watcher, daemon=True)
    thread.start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"MuseTalk server on http://{HOST}:{PORT} home={HOME}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
