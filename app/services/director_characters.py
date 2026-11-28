"""Character catalog, lightweight speaker diarization, and protagonist helpers."""

from __future__ import annotations

import math
import re
import shutil
import subprocess
import tomllib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from loguru import logger

from app.services.comfyui_director import (
    DirectorError,
    extract_voice_reference,
    repo_root,
    set_active_voice_ref,
)

LOCAL_TZ = ZoneInfo("America/Buenos_Aires")
SPEAKER_MODEL = repo_root() / "models" / "speaker" / "campplus_sv_en_voxceleb_16k.onnx"
SIMILARITY_THRESHOLD = 0.40
DIALOGUE_LINE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]+)\]|(?P<name>[^:]{1,40}))\s*:\s*(?P<text>.+?)\s*$"
)
SCENE_OWNER = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(?P<rest>.*)$")


def characters_dir() -> Path:
    path = repo_root() / "bibles" / "characters"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "personaje"


def _unique_slug(name: str) -> str:
    base = _slugify(name)
    slug = base
    index = 2
    while (characters_dir() / slug / "character.toml").is_file():
        slug = f"{base}-{index}"
        index += 1
    return slug


def _esc(value: str) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class Character:
    name: str
    slug: str
    face_ref: str = ""
    voice_ref: str = ""
    created: str = ""
    source: str = ""
    extra_faces: list[str] = field(default_factory=list)
    path: str = ""

    def thumb(self) -> str:
        if self.face_ref and Path(self.face_ref).is_file():
            return self.face_ref
        return ""


@dataclass
class SpeakerCandidate:
    speaker_id: str
    wav_path: str
    count: int
    duration: float
    segments: list[tuple[float, float]] = field(default_factory=list)


def _dump_character(char: Character) -> str:
    extras = ", ".join(f'"{_esc(item)}"' for item in char.extra_faces if item)
    return (
        f'name = "{_esc(char.name)}"\n'
        f'slug = "{_esc(char.slug)}"\n'
        f'face_ref = "{_esc(char.face_ref)}"\n'
        f'voice_ref = "{_esc(char.voice_ref)}"\n'
        f'created = "{_esc(char.created)}"\n'
        f'source = "{_esc(char.source)}"\n'
        f"extra_faces = [{extras}]\n"
    )


def _load_one(folder: Path) -> Character | None:
    toml_path = folder / "character.toml"
    if not toml_path.is_file():
        return None
    try:
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"could not read {toml_path}: {exc}")
        return None

    def _abs(raw: str) -> str:
        raw = str(raw or "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = folder / path
        return str(path)

    extras = data.get("extra_faces") or []
    if not isinstance(extras, list):
        extras = []
    char = Character(
        name=str(data.get("name") or folder.name).strip() or folder.name,
        slug=str(data.get("slug") or folder.name).strip() or folder.name,
        face_ref=_abs(str(data.get("face_ref") or "")),
        voice_ref=_abs(str(data.get("voice_ref") or "")),
        created=str(data.get("created") or ""),
        source=str(data.get("source") or ""),
        extra_faces=[_abs(str(item)) for item in extras],
        path=str(toml_path),
    )
    return char


def list_characters() -> list[Character]:
    items = []
    root = characters_dir()
    for folder in sorted(root.iterdir(), key=lambda path: path.name):
        if folder.is_dir():
            char = _load_one(folder)
            if char:
                items.append(char)
    return items


def load_character(slug: str) -> Character | None:
    slug = _slugify(slug)
    return _load_one(characters_dir() / slug)


def save_character(
    *,
    name: str,
    face_path: str | Path,
    voice_path: str | Path | None = None,
    source: str = "",
    extra_faces: list[str] | None = None,
    slug: str | None = None,
) -> Character:
    name = (name or "").strip()
    if not name:
        raise DirectorError("Ponéle un nombre al personaje.")
    face = Path(face_path)
    if not face.is_file():
        raise DirectorError(f"No encuentro el recorte de cara: {face}")
    slug = slug or _unique_slug(name)
    dest = characters_dir() / slug
    dest.mkdir(parents=True, exist_ok=True)
    face_dest = dest / f"face{face.suffix.lower() or '.png'}"
    if Path(face).resolve() != face_dest.resolve():
        shutil.copy2(face, face_dest)
    voice_dest = ""
    if voice_path and Path(voice_path).is_file():
        src_voice = Path(voice_path)
        voice_dest_path = dest / f"voice{src_voice.suffix.lower() or '.wav'}"
        if src_voice.resolve() != voice_dest_path.resolve():
            shutil.copy2(src_voice, voice_dest_path)
        voice_dest = str(voice_dest_path)
    copied_extras = []
    for index, extra in enumerate(extra_faces or [], start=2):
        src = Path(extra)
        if not src.is_file():
            continue
        extra_dest = dest / f"face_{index}{src.suffix.lower() or '.png'}"
        if src.resolve() != extra_dest.resolve():
            shutil.copy2(src, extra_dest)
        copied_extras.append(str(extra_dest))
    existing = _load_one(dest)
    created = (existing.created if existing and existing.created else datetime.now(LOCAL_TZ).isoformat())
    char = Character(
        name=name,
        slug=slug,
        face_ref=str(face_dest),
        voice_ref=voice_dest,
        created=created,
        source=source,
        extra_faces=copied_extras,
        path=str(dest / "character.toml"),
    )
    Path(char.path).write_text(_dump_character(char), encoding="utf-8")
    return char


def rename_character(slug: str, new_name: str) -> Character:
    char = load_character(slug)
    if not char:
        raise DirectorError(f"No encuentro el personaje '{slug}'.")
    new_name = (new_name or "").strip()
    if not new_name:
        raise DirectorError("El nombre no puede estar vacío.")
    char.name = new_name
    Path(char.path).write_text(_dump_character(char), encoding="utf-8")
    return char


def delete_character(slug: str) -> None:
    slug = _slugify(slug)
    folder = characters_dir() / slug
    if not folder.is_dir():
        raise DirectorError(f"No encuentro el personaje '{slug}'.")
    shutil.rmtree(folder)


def protagonists_from_settings(settings: dict[str, Any] | None = None) -> list[Character]:
    if settings is None:
        from app.services.comfyui_director import comfyui_settings

        settings = comfyui_settings()
    raw = settings.get("protagonists") or ""
    if isinstance(raw, list):
        slugs = [str(item).strip() for item in raw]
    else:
        slugs = [part.strip() for part in str(raw).split(",")]
    characters = []
    for slug in slugs:
        if not slug:
            continue
        char = load_character(slug)
        if char:
            characters.append(char)
    return characters


def apply_character_to_bible(bible: dict[str, Any], char: Character) -> dict[str, Any]:
    bible = dict(bible)
    if char.name:
        bible["character_name"] = char.name
    if char.face_ref and Path(char.face_ref).is_file():
        bible["face_ref"] = char.face_ref
    if char.voice_ref and Path(char.voice_ref).is_file():
        bible["voice_ref"] = char.voice_ref
        set_active_voice_ref(char.voice_ref)
    return bible


# ----- speaker diarization (CAM++ ONNX, no gated token, CPU) -----

_SESSION = None


def _speaker_session():
    global _SESSION
    if _SESSION is None:
        if not SPEAKER_MODEL.is_file():
            raise DirectorError(
                "No encuentro el modelo de locutor CAM++ en "
                f"{SPEAKER_MODEL}. Es un ONNX chico, sin token de Hugging Face."
            )
        import onnxruntime as ort

        _SESSION = ort.InferenceSession(str(SPEAKER_MODEL), providers=["CPUExecutionProvider"])
    return _SESSION


def _load_mono16k(path: Path, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error"]
    if start > 0:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", str(path)]
    if duration and duration > 0:
        command += ["-t", f"{duration:.3f}"]
    command += ["-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise DirectorError(
            "No pude leer el audio para separar locutores: "
            + (result.stderr or b"").decode("utf-8", "ignore")[-300:]
        )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def _fbank(wav: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, wav.astype(np.float32))
    fbank.input_finished()
    if fbank.num_frames_ready <= 0:
        return np.zeros((1, 80), dtype=np.float32)
    feats = np.stack([fbank.get_frame(i) for i in range(fbank.num_frames_ready)]).astype(np.float32)
    feats -= feats.mean(axis=0, keepdims=True)
    return feats


def _embed(wav: np.ndarray) -> np.ndarray:
    if wav.size < 16000 * 0.4:
        raise ValueError("audio too short")
    feats = _fbank(wav)
    session = _speaker_session()
    name = session.get_inputs()[0].name
    out = session.run(None, {name: feats[None, ...]})[0]
    vector = np.asarray(out[0], dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _vad_segments(wav: np.ndarray, sample_rate: int = 16000) -> list[tuple[float, float]]:
    frame = int(sample_rate * 0.03)
    hop = frame
    if wav.size < frame * 4:
        return [(0.0, wav.size / sample_rate)] if wav.size > sample_rate * 0.4 else []
    energies = []
    for start in range(0, wav.size - frame + 1, hop):
        chunk = wav[start : start + frame]
        energies.append(float(np.sqrt(np.mean(chunk * chunk) + 1e-12)))
    threshold = max(0.008, float(np.median(energies)) * 0.6)
    voiced = [energy > threshold for energy in energies]
    # fill 2-frame holes
    for i in range(1, len(voiced) - 1):
        if voiced[i - 1] and voiced[i + 1]:
            voiced[i] = True
    segments: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(voiced):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            segments.append((start * hop / sample_rate, i * hop / sample_rate))
            start = None
    if start is not None:
        segments.append((start * hop / sample_rate, len(voiced) * hop / sample_rate))
    # merge gaps < 0.35s, drop < 0.6s
    merged: list[tuple[float, float]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= 0.35:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    return [(a, b) for a, b in merged if (b - a) >= 0.6]


def _cut_wav(wav: np.ndarray, start: float, end: float, sample_rate: int = 16000) -> np.ndarray:
    i0 = max(0, int(start * sample_rate))
    i1 = min(wav.size, int(end * sample_rate))
    return wav[i0:i1]


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(wav, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-i",
            "pipe:0",
            str(path),
        ],
        input=pcm.tobytes(),
        capture_output=True,
    )
    if result.returncode != 0 or not path.is_file():
        raise DirectorError(
            "No pude guardar la muestra de locutor: "
            + (result.stderr or b"").decode("utf-8", "ignore")[-300:]
        )
    return path


def diarize_speakers(
    source: str | Path,
    dest_dir: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    max_speakers: int = 6,
    face_count: int | None = None,
) -> list[SpeakerCandidate]:
    """Cluster speaker embeddings (CAM++ ONNX) over energy-VAD segments."""
    src = Path(source)
    if src.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return []
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    wav = _load_mono16k(src, start=max(0.0, start_seconds), duration=duration_seconds)
    offset = max(0.0, start_seconds)
    segments = _vad_segments(wav)
    if not segments:
        return []
    chunks: list[dict[str, Any]] = []
    for seg_start, seg_end in segments:
        t = seg_start
        while t < seg_end - 0.45:
            end = min(seg_end, t + 1.5)
            piece = _cut_wav(wav, t, end)
            try:
                embedding = _embed(piece)
            except Exception:
                t += 1.1
                continue
            chunks.append(
                {
                    "start": offset + t,
                    "end": offset + end,
                    "embedding": embedding,
                    "local_start": t,
                    "local_end": end,
                }
            )
            t += 1.1
    if not chunks:
        return []
    clusters: list[list[dict[str, Any]]] = []
    for item in chunks:
        assigned = False
        for cluster in clusters:
            if _cosine(item["embedding"], cluster[0]["embedding"]) >= SIMILARITY_THRESHOLD:
                cluster.append(item)
                assigned = True
                break
        if not assigned:
            clusters.append([item])
    clusters.sort(key=len, reverse=True)
    # One face on screen almost always means one speaker; merge leftovers.
    if face_count == 1 and len(clusters) > 1:
        dominant = clusters[0]
        for extra in clusters[1:]:
            dominant.extend(extra)
        clusters = [dominant]
    else:
        # Merge tiny leftover clusters into the closest larger one.
        if len(clusters) > 1:
            merged = [clusters[0]]
            for extra in clusters[1:]:
                if len(extra) <= 2 and len(merged[0]) >= 3:
                    # attach to the most similar already-accepted cluster
                    best = max(
                        merged,
                        key=lambda cluster: _cosine(extra[0]["embedding"], cluster[0]["embedding"]),
                    )
                    if _cosine(extra[0]["embedding"], best[0]["embedding"]) >= 0.30:
                        best.extend(extra)
                        continue
                merged.append(extra)
            clusters = merged
    clusters = clusters[: max(1, max_speakers)]
    speakers: list[SpeakerCandidate] = []
    for index, cluster in enumerate(clusters, start=1):
        # Representative: longest chunk closest to the centroid.
        centroid = np.mean([item["embedding"] for item in cluster], axis=0)
        best = max(
            cluster,
            key=lambda item: (item["end"] - item["start"]) + _cosine(item["embedding"], centroid),
        )
        speaker_id = chr(ord("A") + index - 1) if index <= 26 else str(index)
        wav_path = dest / f"speaker_{speaker_id}.wav"
        snippet = _cut_wav(wav, best["local_start"], best["local_end"])
        _write_wav(wav_path, snippet)
        speakers.append(
            SpeakerCandidate(
                speaker_id=speaker_id,
                wav_path=str(wav_path.resolve()),
                count=len(cluster),
                duration=sum(item["end"] - item["start"] for item in cluster),
                segments=[(item["start"], item["end"]) for item in cluster],
            )
        )
    logger.info(f"diarized {len(chunks)} chunks into {len(speakers)} speakers from {src}")
    return speakers


def map_faces_to_speakers(
    faces,
    speakers: list[SpeakerCandidate],
) -> dict[int, str]:
    """Majority vote: which speaker is talking when that face is on screen."""
    mapping: dict[int, str] = {}
    if not faces or not speakers:
        return mapping
    if len(faces) == 1 and len(speakers) == 1:
        return {faces[0].cluster_id: speakers[0].speaker_id}
    for face in faces:
        votes: dict[str, int] = {speaker.speaker_id: 0 for speaker in speakers}
        times = list(getattr(face, "times", None) or [])
        for time_sec in times:
            for speaker in speakers:
                if any(start <= time_sec <= end for start, end in speaker.segments):
                    votes[speaker.speaker_id] += 1
                    break
        if not times or sum(votes.values()) == 0:
            continue
        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        best_id, best_votes = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if best_votes >= 1 and best_votes > second:
            mapping[face.cluster_id] = best_id
    return mapping


# ----- script / generation helpers -----

def parse_dialogue_line(line: str) -> tuple[str, str] | None:
    match = DIALOGUE_LINE.match(line or "")
    if not match:
        return None
    name = (match.group("bracket") or match.group("name") or "").strip()
    text = (match.group("text") or "").strip()
    if not name or not text:
        return None
    return name, text


def parse_scene_owner(term: str) -> tuple[str, str]:
    match = SCENE_OWNER.match(term or "")
    if not match:
        return "", (term or "").strip()
    return match.group("name").strip(), (match.group("rest") or "").strip()


def _name_matches(label: str, char: Character) -> bool:
    label = _slugify(label)
    return label in {_slugify(char.name), _slugify(char.slug)}


def character_for_label(label: str, characters: list[Character]) -> Character | None:
    for char in characters:
        if _name_matches(label, char):
            return char
    return None


def character_for_scene(term: str, index: int, characters: list[Character]) -> Character | None:
    if not characters:
        return None
    label, _ = parse_scene_owner(term)
    if label:
        found = character_for_label(label, characters)
        if found:
            return found
    # Alternate if the term has no tag.
    return characters[(max(1, index) - 1) % len(characters)]


def dialogue_script_instruction(characters: list[Character], paragraph_number: int) -> str:
    names = ", ".join(char.name for char in characters)
    lines = "\n".join(f"{char.name}: ..." for char in characters)
    count = max(paragraph_number, len(characters) * 2)
    return (
        f"\n\nThis is a short dialogue among these characters: {names}. "
        f"Write about {count} spoken lines, alternating speakers, each line starting "
        f"exactly with the character name and a colon, like:\n{lines}\n"
        "Do not put two people in the same shot. No stage directions in the spoken text."
    )


def terms_from_dialogue(script: str, characters: list[Character]) -> list[str]:
    terms = []
    for line in (script or "").splitlines():
        parsed = parse_dialogue_line(line)
        if not parsed:
            continue
        name, text = parsed
        terms.append(
            f"[{name}] medium shot of {name} speaking, cinematic, one person only: {text}"
        )
    if terms:
        return terms
    # Fallback: alternate empty visual beats.
    return [f"[{char.name}] medium shot of {char.name}, cinematic, one person only" for char in characters]


def tts_dialogue(
    *,
    script: str,
    characters: list[Character],
    voice_name: str,
    voice_rate: float,
    output_path: str | Path,
) -> Any:
    """Synthesize tagged dialogue lines, each with that character's voice_ref."""
    from app.services import voice as voice_service

    lines = []
    for raw in (script or "").splitlines():
        parsed = parse_dialogue_line(raw)
        if parsed:
            lines.append(parsed)
    if not lines:
        return None
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output.parent / "dialogue_lines"
    temp_dir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    spoken: list[str] = []
    segments: list[dict] = []
    cursor = 0.0
    try:
        for index, (name, text) in enumerate(lines, start=1):
            char = character_for_label(name, characters)
            if char and char.voice_ref and Path(char.voice_ref).is_file():
                set_active_voice_ref(char.voice_ref)
            part = temp_dir / f"line_{index:02d}.mp3"
            sub = voice_service.tts(
                text=text,
                voice_name=voice_name,
                voice_rate=voice_rate,
                voice_file=str(part),
            )
            if sub is None or not part.is_file():
                raise DirectorError(f"No pude sintetizar la réplica de {name}.")
            duration = float(voice_service.get_audio_duration(str(part)) or 0.0)
            parts.append(part)
            spoken.append(text)
            segments.append(
                {
                    "path": str(part),
                    "start": cursor,
                    "end": cursor + duration,
                    "duration": duration,
                    "name": name,
                    "text": text,
                }
            )
            cursor += duration
        concat_list = temp_dir / "concat.txt"
        concat_list.write_text(
            "".join(f"file '{part.resolve()}'\n" for part in parts),
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list),
                "-c",
                "copy",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output.is_file():
            # re-encode if stream copy fails
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat_list),
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
        if result.returncode != 0 or not output.is_file():
            raise DirectorError("No pude unir las réplicas de voz: " + (result.stderr or "")[-300:])
        duration = voice_service.get_audio_duration(str(output))
        try:
            from app.services.director_lipsync import write_audio_segments

            write_audio_segments(output.parent, segments)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"could not write audio segments: {exc}")
        sub_maker = voice_service.ensure_legacy_submaker_fields(voice_service.SubMaker())
        return voice_service.populate_legacy_submaker_with_full_text(
            sub_maker=sub_maker,
            text=" ".join(spoken),
            audio_duration_seconds=duration,
        )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

def concat_audio_files(sources: list[str | Path], destination: str | Path) -> Path:
    """Concatenate wav/mp3 samples into one voice_ref."""
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    files = [Path(src) for src in sources if src and Path(src).is_file()]
    if not files:
        raise DirectorError("No hay muestras de voz para unir.")
    if len(files) == 1:
        if files[0].resolve() != dest.resolve():
            shutil.copy2(files[0], dest)
        return dest
    list_file = dest.parent / f"{dest.stem}_concat.txt"
    # Re-encode to wav first so mixed formats can be joined.
    wavs = []
    for index, src in enumerate(files, start=1):
        wav = dest.parent / f"{dest.stem}_part{index:02d}.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and wav.is_file():
            wavs.append(wav)
    if not wavs:
        raise DirectorError("No pude preparar las muestras para unirlas.")
    list_file.write_text("".join(f"file '{w.resolve()}'\n" for w in wavs), encoding="utf-8")
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(list_file), str(dest)],
        capture_output=True,
        text=True,
    )
    for wav in wavs:
        if wav.resolve() != dest.resolve():
            wav.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    if result.returncode != 0 or not dest.is_file():
        raise DirectorError("No pude unir las muestras de voz: " + (result.stderr or "")[-300:])
    return dest


def extract_full_voice_reference(
    source: str | Path,
    destination: str | Path,
    *,
    max_seconds: float = 60.0,
) -> Path:
    """Extract the whole spoken audio (silences trimmed), capped at max_seconds."""
    src = Path(source)
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    filt = (
        "silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB:"
        "detection=peak,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.2:start_threshold=-40dB:"
        "detection=peak,"
        "areverse"
    )
    command = [
        "ffmpeg", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", filt,
        "-t", str(max(8.0, min(max_seconds, float(max_seconds)))),
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1000:
        command = [
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000",
            "-t", str(max(8.0, min(60.0, float(max_seconds)))),
            str(dest),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1000:
        raise DirectorError(
            "No pude extraer el recorte total de audio: "
            + (result.stderr or result.stdout or "")[-400:]
        )
    return dest

