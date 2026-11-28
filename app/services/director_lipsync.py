"""Lip-sync MiniMax clips to per-scene Chatterbox audio with MuseTalk 1.5."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger

from app.services.comfyui_director import (
    DirectorError,
    free_comfyui_memory,
    repo_root,
    unload_chatterbox,
    unload_ollama_models,
)

MUSETALK_URL_DEFAULT = "http://127.0.0.1:4124"
MUSETALK_SERVE_SCRIPT = repo_root() / "scripts" / "musetalk_serve.sh"
SEGMENTS_NAME = "audio_segments.json"


def _duration(path: str | Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        value = float((result.stdout or "").strip())
    except ValueError:
        value = 0.0
    return value if value > 0 else 0.0


def write_audio_segments(task_dir: str | Path, segments: list[dict[str, Any]]) -> Path:
    path = Path(task_dir) / SEGMENTS_NAME
    path.write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_audio_segments(task_dir: str | Path, audio_path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(task_dir) / SEGMENTS_NAME
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"could not read audio segments: {exc}")
    duration = _duration(audio_path)
    return [{"path": str(audio_path), "start": 0.0, "end": duration, "duration": duration}]


def extract_audio_slice(source: str | Path, dest: Path, start: float, duration: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.2, duration):.3f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file():
        raise DirectorError(
            "No pude recortar el audio de la escena: " + (result.stderr or "")[-300:]
        )
    return dest


def extend_video_to_duration(video: str | Path, dest: Path, seconds: float) -> Path:
    """Ping-pong (forward + reverse) the clip until it covers the audio length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    current = _duration(video)
    if current <= 0:
        raise DirectorError(f"No pude leer la duración del clip: {video}")
    if current >= seconds - 0.05:
        if dest.resolve() != Path(video).resolve():
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-t",
                    f"{seconds:.3f}",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(dest),
                ],
                check=False,
                capture_output=True,
            )
            if dest.is_file() and dest.stat().st_size > 1000:
                return dest
        return Path(video)
    # Build a forward+reverse cycle, then loop it.
    work = dest.parent / f"{dest.stem}_cycle.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-filter_complex",
            "[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(work),
        ],
        capture_output=True,
        text=True,
    )
    source = work if result.returncode == 0 and work.is_file() else Path(video)
    loops = max(1, int(seconds / max(_duration(source), 0.2) + 1))
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-stream_loop",
            str(loops),
            "-i",
            str(source),
            "-t",
            f"{seconds:.3f}",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(dest),
        ],
        capture_output=True,
        text=True,
    )
    if work.is_file() and work != dest:
        work.unlink(missing_ok=True)
    if result.returncode != 0 or not dest.is_file():
        raise DirectorError(
            "No pude alargar el clip hasta la duración del audio: "
            + (result.stderr or "")[-300:]
        )
    return dest


def unload_musetalk() -> None:
    try:
        response = requests.post(f"{MUSETALK_URL_DEFAULT}/free", timeout=10)
        if response.status_code >= 400:
            logger.warning(f"MuseTalk /free returned HTTP {response.status_code}")
    except requests.RequestException as exc:
        logger.debug(f"musetalk /free skipped: {exc}")


def ensure_musetalk_server(timeout: float = 20.0) -> str:
    try:
        response = requests.get(f"{MUSETALK_URL_DEFAULT}/health", timeout=2)
        if response.ok:
            return MUSETALK_URL_DEFAULT
    except requests.RequestException:
        pass
    if not MUSETALK_SERVE_SCRIPT.is_file():
        raise DirectorError(f"No encuentro el script de MuseTalk: {MUSETALK_SERVE_SCRIPT}")
    log_path = Path.home() / "Videos/musetalk/server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab")
    subprocess.Popen(
        ["sh", str(MUSETALK_SERVE_SCRIPT)],
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        cwd=str(Path.home() / "Videos/musetalk"),
    )
    deadline = time.time() + timeout
    last_error = "timeout"
    while time.time() < deadline:
        try:
            response = requests.get(f"{MUSETALK_URL_DEFAULT}/health", timeout=2)
            if response.ok:
                logger.info(f"MuseTalk server ready at {MUSETALK_URL_DEFAULT}")
                return MUSETALK_URL_DEFAULT
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise DirectorError(
        f"No pude levantar MuseTalk en {MUSETALK_URL_DEFAULT}: {last_error}. Revisá {log_path}"
    )


def _assign_segments_to_clips(
    video_paths: list[str],
    segments: list[dict[str, Any]],
    audio_path: str | Path,
) -> list[tuple[str, Path, float]]:
    """Return (video, audio_file_for_scene, target_seconds) for each clip."""
    pairs: list[tuple[str, Path, float]] = []
    usable = []
    for item in segments:
        duration = float(item.get("duration") or 0.0)
        if duration <= 0 and item.get("end") is not None:
            duration = max(0.0, float(item["end"]) - float(item.get("start") or 0.0))
        path = Path(str(item.get("path") or ""))
        if duration <= 0:
            continue
        usable.append({**item, "duration": duration, "path": path})
    if len(usable) == len(video_paths):
        for video, item in zip(video_paths, usable):
            audio = item["path"] if item["path"].is_file() else Path(audio_path)
            if audio == Path(audio_path) and (item.get("start") or 0) or item.get("end"):
                # slice later if we only have the master file
                pairs.append((video, audio, float(item["duration"])))
            else:
                pairs.append((video, audio, float(item["duration"])))
        return pairs
    # One audio for many clips (or vice versa): split the master sequentially.
    total_audio = _duration(audio_path)
    if total_audio <= 0:
        raise DirectorError("No pude leer la duración del audio para el lip-sync.")
    if len(video_paths) == 1:
        return [(video_paths[0], Path(audio_path), total_audio)]
    cursor = 0.0
    for index, video in enumerate(video_paths):
        remaining_clips = len(video_paths) - index
        remaining_audio = max(0.2, total_audio - cursor)
        if index == len(video_paths) - 1:
            length = remaining_audio
        else:
            # Prefer the rendered clip length, but leave audio for the rest.
            clip_len = _duration(video) or (remaining_audio / remaining_clips)
            length = min(clip_len, remaining_audio - 0.2 * (remaining_clips - 1))
            length = max(0.4, length)
        pairs.append((video, Path(audio_path), length))
        cursor += length
        if cursor >= total_audio - 0.05:
            # Drop leftover video clips; they would have no speech to sync.
            break
    return pairs


def lipsync_clips(
    video_paths: list[str],
    *,
    audio_path: str | Path,
    task_dir: str | Path,
    timeout: int = 600,
) -> list[str]:
    """Replace each MiniMax clip with a MuseTalk version matched to its audio slice."""
    if not video_paths:
        return video_paths
    audio = Path(audio_path)
    if not audio.is_file():
        raise DirectorError(f"No encuentro el audio para el lip-sync: {audio}")
    dest_dir = Path(task_dir)
    segments = load_audio_segments(dest_dir, audio)
    pairs = _assign_segments_to_clips(video_paths, segments, audio)
    unload_ollama_models()
    unload_chatterbox()
    free_comfyui_memory()
    ensure_musetalk_server()
    synced: list[str] = []
    try:
        cursor = 0.0
        for index, (video, segment_audio, seconds) in enumerate(pairs, start=1):
            work = dest_dir / f"lipsync_{index:02d}"
            work.mkdir(parents=True, exist_ok=True)
            slice_path = segment_audio
            if segment_audio.resolve() == audio.resolve() and len(pairs) != 1:
                slice_path = work / "line.wav"
                extract_audio_slice(audio, slice_path, cursor, seconds)
                cursor += seconds
            elif _duration(segment_audio) <= 0:
                slice_path = work / "line.wav"
                extract_audio_slice(audio, slice_path, 0.0, seconds)
            target = max(seconds, _duration(slice_path) or seconds)
            extended = work / "extended.mp4"
            extend_video_to_duration(video, extended, target)
            output = dest_dir / f"{Path(video).stem}_lipsync.mp4"
            logger.info(f"MuseTalk lip-sync clip {index}: {video} + {slice_path} -> {output}")
            response = requests.post(
                f"{MUSETALK_URL_DEFAULT}/lipsync",
                json={
                    "video": str(extended if extended.is_file() else video),
                    "audio": str(slice_path),
                    "output": str(output),
                },
                timeout=timeout,
            )
            if response.status_code >= 400:
                raise DirectorError(
                    f"MuseTalk falló (HTTP {response.status_code}): {response.text[:500]}"
                )
            payload = response.json() if response.content else {}
            if not output.is_file() or output.stat().st_size < 1000:
                raise DirectorError(f"MuseTalk no escribió el clip: {payload}")
            # Trim to the audio length if MuseTalk ran a bit long.
            trimmed = dest_dir / f"{Path(video).stem}_lipsync_trim.mp4"
            audio_len = _duration(slice_path) or target
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(output),
                    "-i",
                    str(slice_path),
                    "-t",
                    f"{audio_len:.3f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(trimmed),
                ],
                capture_output=True,
                text=True,
            )
            final = trimmed if result.returncode == 0 and trimmed.is_file() else output
            synced.append(str(final))
            logger.info(f"lip-sync clip {index} ready in {final} ({_duration(final):.2f}s)")
    finally:
        unload_musetalk()
        free_comfyui_memory()
    return synced
