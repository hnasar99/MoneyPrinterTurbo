"""Face reference extraction and PuLID first-frame generation for the director fork."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from loguru import logger

from app.services.comfyui_director import (
    DirectorError,
    comfy_url,
    free_comfyui_memory,
    load_bible,
    repo_root,
    resolve_bible_path,
    save_bible,
    unload_chatterbox,
    unload_ollama_models,
    upload_image,
)

INSIGHTFACE_ROOT = Path.home() / "Videos/ComfyUI/models/insightface"
PULID_CHECKPOINT = "sdxl_lightning_4step.safetensors"
PULID_ADAPTER = "ip-adapter_pulid_sdxl_fp16.safetensors"
SIMILARITY_THRESHOLD = 0.5


def faces_dir() -> Path:
    path = repo_root() / "bibles" / "faces"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _sharpening(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class FaceCandidate:
    cluster_id: int
    count: int
    crop_path: str
    thumb_path: str
    score: float
    extra_paths: list[str] = field(default_factory=list)


def _face_analyzer():
    from insightface.app import FaceAnalysis

    if not (INSIGHTFACE_ROOT / "models" / "antelopev2").is_dir():
        raise DirectorError(
            "No encuentro los modelos AntelopeV2 de InsightFace en "
            f"{INSIGHTFACE_ROOT}/models/antelopev2."
        )
    analyzer = FaceAnalysis(name="antelopev2", root=str(INSIGHTFACE_ROOT), providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return analyzer


def _iter_sample_frames(source: Path, max_frames: int = 24):
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        image = cv2.imread(str(source))
        if image is None:
            raise DirectorError(f"No pude leer la imagen: {source}")
        yield 0, image
        return
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise DirectorError(f"No pude abrir el video para buscar caras: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 24.0
        if total <= 0:
            index = 0
            kept = 0
            while kept < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % max(1, int(fps // 2) or 1) == 0:
                    yield index, frame
                    kept += 1
                index += 1
            return
        duration = total / fps
        count = min(max_frames, max(6, int(duration * 2)))
        start = min(0.3, max(0.0, duration * 0.02))
        end = max(start + 0.1, duration - 0.2)
        for i in range(count):
            t = start + (end - start) * (i / max(1, count - 1))
            frame_index = min(total - 1, max(0, int(t * fps)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                yield frame_index, frame
    finally:
        capture.release()


def _crop_with_margin(image: np.ndarray, bbox, margin: float = 0.35) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    x1 -= w * margin
    x2 += w * margin
    y1 -= h * margin
    y2 += h * margin
    height, width = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def extract_face_candidates(source: str | Path, dest_dir: str | Path | None = None) -> list[FaceCandidate]:
    """Detect faces, group them by person, and write one thumbnail per cluster."""
    src = Path(source)
    dest = Path(dest_dir) if dest_dir else faces_dir() / (src.stem or "personaje")
    dest.mkdir(parents=True, exist_ok=True)
    analyzer = _face_analyzer()
    detections: list[dict[str, Any]] = []
    for frame_index, frame in _iter_sample_frames(src):
        faces = analyzer.get(frame) or []
        for face in faces:
            bbox = getattr(face, "bbox", None)
            embedding = getattr(face, "embedding", None)
            if bbox is None or embedding is None:
                continue
            crop = _crop_with_margin(frame, bbox)
            if crop.size == 0:
                continue
            area = max(1.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
            score = float(getattr(face, "det_score", 0.5)) * math.sqrt(area) * (1.0 + min(_sharpening(crop) / 400.0, 1.0))
            detections.append(
                {
                    "embedding": np.asarray(embedding, dtype=np.float32),
                    "crop": crop,
                    "score": score,
                    "frame": frame_index,
                }
            )
    if not detections:
        raise DirectorError(
            "No encontré caras en ese material. Probá con un tramo más cercano o una foto más nítida."
        )

    clusters: list[list[dict[str, Any]]] = []
    for item in detections:
        assigned = False
        for cluster in clusters:
            if _cosine(item["embedding"], cluster[0]["embedding"]) >= SIMILARITY_THRESHOLD:
                cluster.append(item)
                assigned = True
                break
        if not assigned:
            clusters.append([item])
    clusters.sort(key=len, reverse=True)

    candidates: list[FaceCandidate] = []
    for index, cluster in enumerate(clusters, start=1):
        best = max(cluster, key=lambda item: item["score"])
        others = sorted(cluster, key=lambda item: item["score"], reverse=True)[1:4]
        crop_path = dest / f"person_{index:02d}.png"
        thumb_path = dest / f"person_{index:02d}_thumb.jpg"
        cv2.imwrite(str(crop_path), best["crop"])
        thumb = best["crop"]
        height, width = thumb.shape[:2]
        scale = 160 / max(height, width)
        if scale < 1:
            thumb = cv2.resize(thumb, (max(1, int(width * scale)), max(1, int(height * scale))))
        cv2.imwrite(str(thumb_path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        extras = []
        for extra_index, extra in enumerate(others, start=2):
            extra_path = dest / f"person_{index:02d}_{extra_index}.png"
            cv2.imwrite(str(extra_path), extra["crop"])
            extras.append(str(extra_path))
        candidates.append(
            FaceCandidate(
                cluster_id=index,
                count=len(cluster),
                crop_path=str(crop_path.resolve()),
                thumb_path=str(thumb_path.resolve()),
                score=float(best["score"]),
                extra_paths=extras,
            )
        )
    logger.info(f"grouped {len(detections)} face detections into {len(candidates)} people from {src}")
    return candidates


def _snap_sdxl(width: int, height: int) -> tuple[int, int]:
    # SDXL likes multiples of 64; keep the aspect and stay modest for 12 GB.
    width = max(64, int(width))
    height = max(64, int(height))
    long_side = max(width, height)
    scale = 768 / long_side if long_side > 0 else 1
    width = int(round(width * scale / 64.0) * 64)
    height = int(round(height * scale / 64.0) * 64)
    return max(64, width), max(64, height)


def _pulid_workflow(
    *,
    prompt: str,
    negative: str,
    width: int,
    height: int,
    seed: int,
    image_name: str,
) -> dict[str, Any]:
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": PULID_CHECKPOINT},
        },
        "16": {
            "class_type": "PulidModelLoader",
            "inputs": {"pulid_file": PULID_ADAPTER},
        },
        "17": {
            "class_type": "PulidInsightFaceLoader",
            "inputs": {"provider": "CPU"},
        },
        "19": {"class_type": "PulidEvaClipLoader", "inputs": {}},
        "12": {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        },
        "22": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": prompt},
        },
        "23": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["4", 1], "text": negative},
        },
        "33": {
            "class_type": "ApplyPulid",
            "inputs": {
                "model": ["4", 0],
                "pulid": ["16", 0],
                "eva_clip": ["19", 0],
                "face_analysis": ["17", 0],
                "image": ["12", 0],
                "method": "fidelity",
                "weight": 0.8,
                "start_at": 0.0,
                "end_at": 1.0,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["33", 0],
                "positive": ["22", 0],
                "negative": ["23", 0],
                "latent_image": ["5", 0],
                "seed": int(seed),
                "steps": 4,
                "cfg": 1.2,
                "sampler_name": "dpmpp_2m",
                "scheduler": "sgm_uniform",
                "denoise": 1.0,
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "director_pulid"},
        },
    }


def build_pulid_prompt(bible: dict[str, Any], scene: str = "") -> tuple[str, str]:
    parts = [
        "photorealistic cinematic still, natural skin texture, sharp eyes",
        bible.get("style") or "",
        f"the same person: {bible.get('character')}" if bible.get("character") else "one person, consistent identity",
        bible.get("camera") or "",
        f"scene: {scene}" if scene else "medium shot, looking toward camera",
    ]
    prompt = ", ".join(part.rstrip(".") for part in parts if part)
    negative = (
        bible.get("negative_prompt")
        or "flaws in the eyes, deformed face, extra fingers, watermark, logo, subtitles, lowres, blurry"
    )
    return prompt, negative


def generate_pulid_first_frame(
    *,
    face_path: str | Path,
    output_path: str | Path,
    bible: dict[str, Any],
    scene: str = "",
    timeout: int = 420,
) -> Path:
    """Generate a PuLID-SDXL first frame and save it to output_path."""
    face = Path(face_path)
    if not face.is_file():
        raise DirectorError(f"No encuentro la cara de referencia: {face}")
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prompt, negative = build_pulid_prompt(bible, scene)
    width, height = _snap_sdxl(int(bible.get("width") or 640), int(bible.get("height") or 352))
    seed = int(bible.get("seed") or 12345)
    base = comfy_url()
    unload_ollama_models()
    unload_chatterbox()
    free_comfyui_memory()
    image_name = upload_image(str(face), base)
    workflow = _pulid_workflow(
        prompt=prompt,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        image_name=image_name,
    )
    started = time.time()
    response = requests.post(
        f"{base}/prompt",
        json={"prompt": workflow, "client_id": f"mpt-pulid-{uuid.uuid4().hex[:8]}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise DirectorError(
            f"ComfyUI rechazó el workflow de PuLID (HTTP {response.status_code}): {response.text[:500]}"
        )
    payload = response.json() if response.content else {}
    if payload.get("node_errors"):
        raise DirectorError(
            "ComfyUI rechazó el workflow de PuLID: "
            + str(payload["node_errors"])[:800]
        )
    prompt_id = str(payload.get("prompt_id") or "").strip()
    if not prompt_id:
        raise DirectorError("ComfyUI no devolvió prompt_id para PuLID.")
    logger.info(f"PuLID prompt submitted: {prompt_id}")
    deadline = time.time() + max(30, timeout)
    record = None
    while time.time() < deadline:
        history = requests.get(f"{base}/history/{prompt_id}", timeout=30)
        history.raise_for_status()
        data = history.json() if history.content else {}
        record = data.get(prompt_id) or (data if data.get("outputs") else None)
        if record:
            status = (record.get("status") or {}).get("status_str") or ""
            messages = record.get("status", {}).get("messages") or []
            blob = status + " " + str(messages)
            if status == "error" or record.get("node_errors"):
                if "out of memory" in blob.lower() or "oom" in blob.lower():
                    raise DirectorError(f"PuLID se quedó sin memoria de video (OOM): {blob[:500]}")
                raise DirectorError(f"PuLID falló: {blob[:800]}")
            if record.get("outputs"):
                break
        time.sleep(2)
    else:
        raise DirectorError(f"PuLID no terminó en {timeout}s (prompt_id={prompt_id}).")
    info = None
    for node_out in (record.get("outputs") or {}).values():
        if not isinstance(node_out, dict):
            continue
        for item in node_out.get("images") or []:
            if isinstance(item, dict) and item.get("filename"):
                info = {
                    "filename": str(item["filename"]),
                    "subfolder": str(item.get("subfolder") or ""),
                    "type": str(item.get("type") or "output"),
                }
                break
        if info:
            break
    if not info:
        raise DirectorError("PuLID terminó sin una imagen de salida.")
    params = urlencode(info)
    image = requests.get(f"{base}/view?{params}", timeout=120)
    if image.status_code >= 400:
        raise DirectorError(f"No pude descargar el primer fotograma de PuLID: HTTP {image.status_code}")
    dest.write_bytes(image.content)
    logger.info(f"PuLID first frame saved in {time.time() - started:.1f}s: {dest}")
    # Free SDXL/PuLID weights before MiniMax H3 takes the GPU.
    free_comfyui_memory()
    return dest


def current_face_ref(bible: dict | None = None) -> str:
    if bible and bible.get("face_ref") and Path(str(bible["face_ref"])).is_file():
        return str(Path(bible["face_ref"]).expanduser())
    try:
        loaded = load_bible()
        if loaded.get("face_ref") and Path(loaded["face_ref"]).is_file():
            return str(loaded["face_ref"])
    except Exception:
        pass
    return ""


def save_face_ref_in_bible(face_path: str | Path, bible_path: str | Path | None = None) -> Path:
    path = resolve_bible_path(str(bible_path) if bible_path else None)
    bible = load_bible(path)
    resolved = Path(face_path).expanduser().resolve()
    bible["face_ref"] = str(resolved)
    return save_bible(path, bible)
