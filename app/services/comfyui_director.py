"""Director fork: video bible, local/cloud LLMs, and ComfyUI MiniMax H3 clips."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
import tomllib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, List
from urllib.parse import urlencode

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MINIMAX_OPENAI_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT = 900

# Local defaults: models already installed on this machine (no pull).
OLLAMA_BIBLE_MODEL = "qwen3-vl:8b"
OLLAMA_PROMPT_MODEL = "Qwen2.5:3b"
# Verified 2026-11-21 against integrate.api.nvidia.com/v1/models
NVIDIA_BIBLE_MODEL = "google/gemma-4-31b-it"
NVIDIA_PROMPT_MODEL = "google/gemma-3-4b-it"
# Verified 2026-11-21 against openrouter.ai/api/v1/models (:free variants)
OPENROUTER_BIBLE_MODEL = "google/gemma-4-31b-it:free"
OPENROUTER_PROMPT_MODEL = "nvidia/nemotron-3.5-lightning:free"

UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE_NAME = "minimax_h3_video_vae_int8_convrot.safetensors"
AUDIO_VAE_NAME = "minimax_h3_audio_vae_fp32.safetensors"
LORA_NAME = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"

PROVIDERS = ("ollama", "nvidia", "openrouter", "configured")
ROLE_LABELS = {"bible": "biblia/guion", "prompt": "prompts visuales"}


class DirectorError(Exception):
    """Error visible de la fuente ComfyUI / biblia."""


def repo_root() -> Path:
    return Path(utils.root_dir())


def bibles_dir() -> Path:
    path = repo_root() / "bibles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def comfyui_settings() -> dict[str, Any]:
    section = dict(config.comfyui or {})
    section.setdefault("url", DEFAULT_COMFY_URL)
    section.setdefault("workflow", "")
    section.setdefault("timeout", DEFAULT_TIMEOUT)
    section.setdefault("bible", "bibles/example.toml")
    section.setdefault("chain_scenes", True)
    section.setdefault("bible_llm_provider", "ollama")
    section.setdefault("bible_llm_model", OLLAMA_BIBLE_MODEL)
    section.setdefault("prompt_llm_provider", "ollama")
    section.setdefault("prompt_llm_model", OLLAMA_PROMPT_MODEL)
    return section


def recommended_models() -> dict[str, dict[str, str]]:
    return {
        "bible": {
            "ollama": OLLAMA_BIBLE_MODEL,
            "nvidia": NVIDIA_BIBLE_MODEL,
            "openrouter": OPENROUTER_BIBLE_MODEL,
        },
        "prompt": {
            "ollama": OLLAMA_PROMPT_MODEL,
            "nvidia": NVIDIA_PROMPT_MODEL,
            "openrouter": OPENROUTER_PROMPT_MODEL,
        },
    }


def list_bible_files() -> list[str]:
    files = sorted(path.name for path in bibles_dir().glob("*.toml"))
    if "example.toml" in files:
        files.remove("example.toml")
        files.insert(0, "example.toml")
    return files


def resolve_bible_path(value: str | None) -> Path:
    raw = str(value or comfyui_settings().get("bible") or "bibles/example.toml").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root() / path
    return path


def load_bible(path: str | Path | None = None) -> dict[str, Any]:
    bible_path = resolve_bible_path(str(path) if path else None)
    if not bible_path.is_file():
        raise DirectorError(f"No encuentro la biblia: {bible_path}")
    with bible_path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise DirectorError("La biblia no es un objeto TOML válido.")
    return _normalize_bible(data, source=bible_path)


def _normalize_bible(data: dict[str, Any], source: Path | None = None) -> dict[str, Any]:
    character = data.get("character") if isinstance(data.get("character"), dict) else {}
    style = data.get("style") if isinstance(data.get("style"), dict) else {}
    camera = data.get("camera") if isinstance(data.get("camera"), dict) else {}
    generation = data.get("generation") if isinstance(data.get("generation"), dict) else {}
    reference = str(character.get("reference_image") or "").strip()
    if reference:
        ref_path = Path(reference).expanduser()
        if not ref_path.is_absolute():
            base = source.parent if source else repo_root()
            ref_path = base / ref_path
        reference = str(ref_path)
    voice_ref = str(character.get("voice_ref") or "").strip()
    if voice_ref:
        voice_path = Path(voice_ref).expanduser()
        if not voice_path.is_absolute():
            base = source.parent if source else repo_root()
            voice_path = base / voice_path
        voice_ref = str(voice_path)
    width = _as_int(generation.get("width"), 640)
    height = _as_int(generation.get("height"), 352)
    width = max(32, min(16384, width - (width % 32)))
    height = max(32, min(16384, height - (height % 32)))
    clip_seconds = _as_float(generation.get("clip_seconds"), 3.0)
    clip_seconds = min(5.0, max(3.0, clip_seconds))
    use_lora = bool(generation.get("use_turbo_lora", True))
    steps = _as_int(generation.get("steps"), 8 if use_lora else 20)
    steps = max(1, min(40, steps))
    return {
        "title": str(data.get("title") or "Proyecto").strip() or "Proyecto",
        "strategy": str(data.get("strategy") or "").strip(),
        "character_name": str(character.get("name") or "").strip(),
        "character": str(character.get("description") or "").strip(),
        "reference_image": reference,
        "voice_ref": voice_ref,
        "style": str(style.get("sentence") or "").strip(),
        "camera": str(camera.get("rules") or "").strip(),
        "negative_prompt": str(generation.get("negative_prompt") or "").strip(),
        "seed": _as_int(generation.get("seed"), 12345),
        "width": width,
        "height": height,
        "clip_seconds": clip_seconds,
        "steps": steps,
        "use_turbo_lora": use_lora,
        "path": str(source) if source else "",
    }


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dump_bible_toml(bible: dict[str, Any]) -> str:
    def esc(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    reference = str(bible.get("reference_image") or "")
    return f"""title = "{esc(str(bible.get("title") or "Proyecto"))}"
strategy = \"\"\"{esc(str(bible.get("strategy") or ""))}\"\"\"

[character]
name = "{esc(str(bible.get("character_name") or ""))}"
description = \"\"\"{esc(str(bible.get("character") or ""))}\"\"\"
reference_image = "{esc(reference)}"
voice_ref = "{esc(str(bible.get("voice_ref") or ""))}"

[style]
sentence = \"\"\"{esc(str(bible.get("style") or ""))}\"\"\"

[camera]
rules = \"\"\"{esc(str(bible.get("camera") or ""))}\"\"\"

[generation]
negative_prompt = \"\"\"{esc(str(bible.get("negative_prompt") or ""))}\"\"\"
seed = {_as_int(bible.get("seed"), 12345)}
width = {_as_int(bible.get("width"), 640)}
height = {_as_int(bible.get("height"), 352)}
clip_seconds = {_as_float(bible.get("clip_seconds"), 3.0)}
steps = {_as_int(bible.get("steps"), 8)}
use_turbo_lora = {str(bool(bible.get("use_turbo_lora", True))).lower()}
"""


def save_bible(path: str | Path, content: str | dict[str, Any]) -> Path:
    bible_path = Path(path)
    if not bible_path.is_absolute():
        bible_path = repo_root() / bible_path
    bible_path.parent.mkdir(parents=True, exist_ok=True)
    text = content if isinstance(content, str) else dump_bible_toml(content)
    bible_path.write_text(text, encoding="utf-8")
    # Validate after write.
    load_bible(bible_path)
    return bible_path


def frame_length(seconds: float) -> int:
    frames = max(5, int(round(float(seconds) * 24)))
    frames += (5 - (frames % 17)) % 17
    return max(5, min(frames, 3600))


def comfy_url() -> str:
    return str(comfyui_settings().get("url") or DEFAULT_COMFY_URL).rstrip("/")


def ollama_native_url() -> str:
    raw = str(config.app.get("ollama_base_url") or "").strip()
    if raw:
        return raw.rstrip("/").removesuffix("/v1")
    default = config.get_default_ollama_base_url().rstrip("/")
    return default.removesuffix("/v1")


def free_comfyui_memory(base_url: str | None = None) -> None:
    url = (base_url or comfy_url()).rstrip("/")
    try:
        response = requests.post(
            f"{url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=30,
        )
        if response.status_code >= 400:
            logger.warning(f"ComfyUI /free returned HTTP {response.status_code}")
        else:
            logger.info("freed ComfyUI models and memory")
    except requests.RequestException as exc:
        logger.warning(f"could not free ComfyUI memory: {exc}")


def unload_ollama_models() -> None:
    base = ollama_native_url()
    names: list[str] = []
    try:
        response = requests.get(f"{base}/api/ps", timeout=10)
        if response.ok:
            payload = response.json() if response.content else {}
            names = [
                str(item.get("name") or item.get("model") or "").strip()
                for item in payload.get("models") or []
                if str(item.get("name") or item.get("model") or "").strip()
            ]
    except requests.RequestException as exc:
        logger.warning(f"could not list loaded Ollama models: {exc}")
        names = [OLLAMA_BIBLE_MODEL, OLLAMA_PROMPT_MODEL]
    if not names:
        return
    for name in names:
        try:
            requests.post(
                f"{base}/api/generate",
                json={"model": name, "keep_alive": 0, "prompt": ""},
                timeout=60,
            )
            logger.info(f"unloaded Ollama model: {name}")
        except requests.RequestException as exc:
            logger.warning(f"could not unload Ollama model {name}: {exc}")


def _strip_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    cleaned = re.sub(r"<\|?/?think\|>", "", cleaned, flags=re.I)
    return cleaned.strip()


def _extract_message_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    if payload.get("message") and isinstance(payload["message"], dict):
        return str(payload["message"].get("content") or "").strip()
    if payload.get("response"):
        return str(payload.get("response") or "").strip()
    choices = payload.get("choices") or []
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
        if content:
            return str(content).strip()
        if message.get("reasoning_content") and not content:
            return ""
    return ""


def _ollama_chat(model: str, prompt: str, keep_alive: Any = 0) -> str:
    response = requests.post(
        f"{ollama_native_url()}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
        },
        timeout=180,
    )
    if response.status_code >= 400:
        raise DirectorError(
            f"Ollama respondió HTTP {response.status_code}: {response.text[:300]}"
        )
    text = _strip_think(_extract_message_text(response.json()))
    if not text:
        raise DirectorError(f"Ollama devolvió una respuesta vacía (modelo {model}).")
    return text


def _openai_chat(base_url: str, api_key: str, model: str, prompt: str, extra_headers: dict | None = None) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        },
        timeout=180,
    )
    if response.status_code >= 400:
        raise DirectorError(
            f"El proveedor {base_url} respondió HTTP {response.status_code}: {response.text[:300]}"
        )
    text = _strip_think(_extract_message_text(response.json()))
    if not text:
        raise DirectorError(f"El modelo {model} devolvió una respuesta vacía.")
    return text


def _configured_provider_chat(prompt: str, app_config=None) -> str:
    from app.services import llm

    text = _strip_think(str(llm._generate_response(prompt, app_config=app_config) or "").strip())
    if not text:
        raise DirectorError("El proveedor LLM configurado devolvió una respuesta vacía.")
    return text


def _nvidia_key() -> str:
    return str(config.app.get("nvidia_api_key") or "").strip()


def _openrouter_key() -> str:
    return str(config.app.get("openrouter_api_key") or "").strip()


def complete_text(
    prompt: str,
    role: str = "bible",
    provider: str | None = None,
    model: str | None = None,
    keep_ollama_loaded: bool = False,
    app_config=None,
) -> tuple[str, str, str]:
    """Return (text, provider_used, model_used). Frees ComfyUI VRAM first."""
    settings = comfyui_settings()
    role_key = "bible" if role == "bible" else "prompt"
    preferred_provider = str(
        provider or settings.get(f"{role_key}_llm_provider") or "ollama"
    ).strip().lower()
    if preferred_provider not in PROVIDERS:
        preferred_provider = "ollama"
    preferred_model = str(
        model or settings.get(f"{role_key}_llm_model") or recommended_models()[role_key]["ollama"]
    ).strip()
    rec = recommended_models()[role_key]

    chain: list[tuple[str, str]] = []

    def add(name: str, mid: str) -> None:
        if mid and (name, mid) not in chain:
            chain.append((name, mid))

    add(preferred_provider, preferred_model or rec.get(preferred_provider, ""))
    add("ollama", rec["ollama"])
    add("nvidia", rec["nvidia"])
    add("openrouter", rec["openrouter"])
    add("configured", "")

    free_comfyui_memory()
    errors: list[str] = []
    for name, mid in chain:
        try:
            if name == "ollama":
                text = _ollama_chat(
                    mid or rec["ollama"],
                    prompt,
                    keep_alive="5m" if keep_ollama_loaded else 0,
                )
                return text, "ollama", mid or rec["ollama"]
            if name == "nvidia":
                key = _nvidia_key()
                if not key:
                    errors.append("NVIDIA: falta nvidia_api_key en config.toml")
                    continue
                text = _openai_chat(
                    NVIDIA_BASE_URL, key, mid or rec["nvidia"], prompt
                )
                return text, "nvidia", mid or rec["nvidia"]
            if name == "openrouter":
                key = _openrouter_key()
                if not key:
                    errors.append("OpenRouter: falta openrouter_api_key en config.toml")
                    continue
                text = _openai_chat(
                    str(config.app.get("openrouter_base_url") or OPENROUTER_BASE_URL),
                    key,
                    mid or rec["openrouter"],
                    prompt,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/harry0703/MoneyPrinterTurbo",
                        "X-Title": "MoneyPrinterTurbo director fork",
                    },
                )
                return text, "openrouter", mid or rec["openrouter"]
            if name == "configured":
                text = _configured_provider_chat(prompt, app_config=app_config)
                used = str(
                    (app_config or config.app).get("llm_provider") or "configured"
                )
                used_model = str(
                    (app_config or config.app).get(f"{used}_model_name") or ""
                )
                return text, used, used_model
        except Exception as exc:  # noqa: BLE001 — fallback chain must keep going
            errors.append(f"{name}: {exc}")
            logger.warning(f"director LLM {name} failed: {exc}")
    raise DirectorError(
        "No pude completar el texto con Ollama, NVIDIA, OpenRouter ni el proveedor configurado. "
        + " | ".join(errors)
    )


def _extract_toml(text: str) -> str:
    fenced = re.search(r"```(?:toml)?\s*(.*?)```", text, flags=re.S | re.I)
    body = fenced.group(1).strip() if fenced else text.strip()
    if "title" not in body and "[" not in body:
        raise DirectorError("El LLM no devolvió una biblia en formato TOML.")
    return body


def generate_bible_from_idea(
    idea: str,
    destination: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> tuple[Path, dict[str, Any], str, str]:
    idea = (idea or "").strip()
    if not idea:
        raise DirectorError("Escribí una idea o estrategia para generar la biblia.")
    example = (bibles_dir() / "example.toml").read_text(encoding="utf-8")
    prompt = f"""You are a film director writing a reusable "video bible" for an AI video pipeline.

Return ONLY valid TOML, no markdown, matching this exact schema and keys:
{example}

Fill every field from the user's idea/strategy. Keep visual descriptions in English
(character.description, style.sentence, camera.rules, generation.negative_prompt).
title may be in Spanish. strategy should restate the idea clearly (Spanish is OK).
Use clip_seconds between 3 and 5, width/height multiples of 32, prefer 640x352 landscape
or 352x640 portrait for a 12 GB GPU. steps=8 and use_turbo_lora=true.
reference_image and voice_ref should stay empty unless the user named a file path.
Leave seed as a positive integer.

User idea/strategy:
{idea}
"""
    text, used_provider, used_model = complete_text(
        prompt, role="bible", provider=provider, model=model, keep_ollama_loaded=False
    )
    toml_text = _extract_toml(text)
    parsed = _normalize_bible(tomllib.loads(toml_text))
    parsed["strategy"] = parsed["strategy"] or idea
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", parsed["title"].lower()).strip("-") or "biblia"
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = Path(destination) if destination else bibles_dir() / f"{slug}-{stamp}.toml"
    if not path.is_absolute():
        path = repo_root() / path
    saved = save_bible(path, parsed)
    return saved, load_bible(saved), used_provider, used_model


def build_visual_prompt(bible: dict[str, Any], scene: str) -> str:
    parts = [
        bible.get("style") or "",
        f"Character (keep identity identical in every shot): {bible.get('character')}"
        if bible.get("character")
        else "",
        f"Camera: {bible.get('camera')}" if bible.get("camera") else "",
        f"Action: {scene.strip()}",
    ]
    prompt = ". ".join(part.rstrip(".") for part in parts if part)
    if bible.get("negative_prompt"):
        prompt = f"{prompt}. Avoid: {bible['negative_prompt']}"
    return prompt


def scene_prompt_via_llm(
    bible: dict[str, Any],
    scene: str,
    provider: str | None = None,
    model: str | None = None,
    keep_ollama_loaded: bool = True,
    app_config=None,
) -> str:
    instruction = f"""Rewrite the scene into ONE English visual prompt for MiniMax H3 (ComfyUI).
Keep the character identity, wardrobe, lighting, color palette and camera language identical
to the bible. Only the action may change. Do not add dialogue captions or watermarks.
Return only the prompt, no quotes, no preamble.

BIBLE
Character: {bible.get('character') or '(unspecified)'}
Style: {bible.get('style') or '(unspecified)'}
Camera: {bible.get('camera') or '(unspecified)'}
Avoid: {bible.get('negative_prompt') or 'none'}

SCENE / SEARCH TERM
{scene}
"""
    try:
        text, _, _ = complete_text(
            instruction,
            role="prompt",
            provider=provider,
            model=model,
            keep_ollama_loaded=keep_ollama_loaded,
            app_config=app_config,
        )
        text = text.strip().strip('"')
        if text:
            if bible.get("negative_prompt") and "Avoid:" not in text:
                text = f"{text}. Avoid: {bible['negative_prompt']}"
            return text
    except DirectorError as exc:
        logger.warning(f"falling back to deterministic visual prompt: {exc}")
    return build_visual_prompt(bible, scene)


def generate_script_and_terms(
    video_subject: str,
    language: str = "",
    paragraph_number: int = 1,
    video_script_prompt: str = "",
    custom_system_prompt: str = "",
    match_script_order: bool = False,
    app_config=None,
) -> tuple[str, list[str]]:
    from app.services import llm

    script_prompt = llm.build_script_prompt(
        video_subject=video_subject,
        language=language,
        paragraph_number=paragraph_number,
        video_script_prompt=video_script_prompt,
        custom_system_prompt=custom_system_prompt,
    )
    script, _, _ = complete_text(
        script_prompt, role="bible", keep_ollama_loaded=True, app_config=app_config
    )
    script = script.replace("*", "").strip()
    if not script:
        return "", []
    try:
        terms_prompt = (
            "Return ONLY a JSON array of 5 to 8 short English visual scene phrases "
            "covering the script in narrative order. No markdown.\n\n"
            f"Subject: {video_subject}\n\nScript:\n{script}"
        )
        raw, _, _ = complete_text(
            terms_prompt, role="prompt", keep_ollama_loaded=False, app_config=app_config
        )
        match = re.search(r"\[.*\]", raw, flags=re.S)
        terms = json.loads(match.group(0) if match else raw)
        terms = [str(item).strip() for item in terms if str(item).strip()]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"director terms failed, continuing without them: {exc}")
        terms = []
    unload_ollama_models()
    return script, terms


def validate_ready() -> str | None:
    settings = comfyui_settings()
    try:
        load_bible(settings.get("bible"))
    except DirectorError as exc:
        return str(exc)
    url = comfy_url()
    try:
        response = requests.get(url, timeout=5)
        if response.status_code >= 400:
            return f"ComfyUI no responde en {url} (HTTP {response.status_code})."
    except requests.RequestException:
        return f"No pude conectar con ComfyUI en {url}. ¿Está corriendo en el puerto 8188?"
    return None


def _snap_aspect(bible: dict[str, Any], video_aspect: VideoAspect | None) -> tuple[int, int]:
    width, height = bible["width"], bible["height"]
    aspect = getattr(video_aspect, "value", video_aspect)
    if aspect == VideoAspect.portrait.value or aspect == VideoAspect.portrait:
        if width > height:
            width, height = height, width
    elif aspect == VideoAspect.landscape.value or aspect == VideoAspect.landscape:
        if height > width:
            width, height = height, width
    return width, height


def _build_minimax_workflow(
    *,
    prompt: str,
    width: int,
    height: int,
    length: int,
    seed: int,
    steps: int,
    use_lora: bool,
    first_image_name: str | None,
) -> dict[str, Any]:
    model_node = "20" if use_lora else "1"
    workflow: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE_NAME},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": AUDIO_VAE_NAME},
        },
        "5": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "6": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed)},
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [model_node, 0],
                "scheduler": "simple",
                "steps": int(steps),
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {"model": [model_node, 0], "conditioning": ["5", 0]},
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["6", 0],
                "guider": ["9", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 1],
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
        },
        "12": {
            "class_type": "VAEDecodeAudio",
            "inputs": {"samples": ["10", 0], "vae": ["4", 0]},
        },
        "13": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["11", 0], "fps": 24.0, "audio": ["12", 0]},
        },
        "14": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["13", 0],
                "filename_prefix": "video/mpt_director",
                "format": "auto",
                "codec": "auto",
            },
        },
    }
    if use_lora:
        workflow["20"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_NAME,
                "strength_model": 1.0,
            },
        }
    if first_image_name:
        workflow["30"] = {
            "class_type": "LoadImage",
            "inputs": {"image": first_image_name},
        }
        workflow["5"]["inputs"]["first_frame"] = ["30", 0]
    return workflow


def _load_custom_workflow(path: str, prompt: str, first_image_name: str | None) -> dict[str, Any]:
    workflow_path = Path(path)
    if not workflow_path.is_absolute():
        workflow_path = repo_root() / workflow_path
    data = json.loads(workflow_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DirectorError("El workflow personalizado no es un objeto JSON.")
    # If this is UI format, reject with a clear message.
    if "nodes" in data and "links" in data and "class_type" not in next(iter(data.values()), {}):
        raise DirectorError(
            "El workflow tiene formato de interfaz de ComfyUI. Exportalo en formato API (File → Export (API))."
        )
    # Patch the first MiniMax / text prompt field and first LoadImage.
    patched_prompt = False
    for node in data.values():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        inputs = node.setdefault("inputs", {})
        if not patched_prompt and "prompt" in inputs and isinstance(inputs["prompt"], str):
            inputs["prompt"] = prompt
            patched_prompt = True
        if first_image_name and node.get("class_type") == "LoadImage":
            inputs["image"] = first_image_name
    return data


def upload_image(image_path: str, base_url: str | None = None) -> str:
    path = Path(image_path)
    if not path.is_file():
        raise DirectorError(f"No encuentro la imagen de referencia: {image_path}")
    url = (base_url or comfy_url()).rstrip("/")
    with path.open("rb") as handle:
        response = requests.post(
            f"{url}/upload/image",
            files={"image": (path.name, handle, "image/png")},
            data={"overwrite": "true"},
            timeout=60,
        )
    if response.status_code >= 400:
        raise DirectorError(f"No pude subir la imagen a ComfyUI: HTTP {response.status_code}")
    payload = response.json() if response.content else {}
    name = str(payload.get("name") or path.name)
    subfolder = str(payload.get("subfolder") or "").strip()
    return f"{subfolder}/{name}" if subfolder else name


def extract_last_frame(video_path: str, destination: str | Path) -> Path:
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-sseof",
        "-0.2",
        "-i",
        str(video_path),
        "-update",
        "1",
        "-q:v",
        "2",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file():
        raise DirectorError(
            "No pude extraer el último fotograma con ffmpeg: "
            + (result.stderr or result.stdout or "")[-400:]
        )
    return dest


def _history_errors(record: dict[str, Any]) -> str:
    status = record.get("status") or {}
    messages = []
    for item in status.get("messages") or []:
        if isinstance(item, (list, tuple)) and item and item[0] == "execution_error":
            messages.append(json.dumps(item[1], ensure_ascii=False)[:800])
        elif isinstance(item, str):
            messages.append(item)
    if record.get("node_errors"):
        messages.append(json.dumps(record["node_errors"], ensure_ascii=False)[:800])
    return "\n".join(messages)


def _pick_output_file(record: dict[str, Any]) -> dict[str, str]:
    outputs = record.get("outputs") or {}
    for key in ("videos", "gifs", "images", "files"):
        for node_out in outputs.values():
            if not isinstance(node_out, dict):
                continue
            for item in node_out.get(key) or []:
                if isinstance(item, dict) and item.get("filename"):
                    return {
                        "filename": str(item["filename"]),
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    }
    raise DirectorError("ComfyUI terminó sin un archivo de video en /history.")


def _download_view(base_url: str, info: dict[str, str], destination: Path) -> Path:
    params = urlencode(
        {
            "filename": info["filename"],
            "subfolder": info.get("subfolder") or "",
            "type": info.get("type") or "output",
        }
    )
    response = requests.get(f"{base_url}/view?{params}", timeout=120)
    if response.status_code >= 400:
        raise DirectorError(f"No pude descargar el clip de ComfyUI: HTTP {response.status_code}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    if destination.stat().st_size < 1000:
        raise DirectorError("El archivo descargado de ComfyUI está vacío o es demasiado chico.")
    return destination


def submit_minimax_clip(
    *,
    prompt: str,
    bible: dict[str, Any],
    output_path: Path,
    first_frame_path: str | None = None,
    video_aspect: VideoAspect | None = None,
    timeout: int | None = None,
) -> tuple[Path, float]:
    settings = comfyui_settings()
    base = comfy_url()
    width, height = _snap_aspect(bible, video_aspect)
    length = frame_length(bible["clip_seconds"])
    first_name = upload_image(first_frame_path, base) if first_frame_path else None
    custom = str(settings.get("workflow") or "").strip()
    if custom:
        workflow = _load_custom_workflow(custom, prompt, first_name)
    else:
        workflow = _build_minimax_workflow(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            seed=bible["seed"],
            steps=bible["steps"],
            use_lora=bible["use_turbo_lora"],
            first_image_name=first_name,
        )
    unload_ollama_models()
    unload_chatterbox()
    started = time.time()
    response = requests.post(
        f"{base}/prompt",
        json={"prompt": workflow, "client_id": f"mpt-director-{uuid.uuid4().hex[:8]}"},
        timeout=60,
    )
    if response.status_code >= 400:
        raise DirectorError(
            f"ComfyUI rechazó el workflow (HTTP {response.status_code}): {response.text[:500]}"
        )
    payload = response.json() if response.content else {}
    if payload.get("node_errors"):
        raise DirectorError(
            "ComfyUI rechazó el workflow: "
            + json.dumps(payload["node_errors"], ensure_ascii=False)[:800]
        )
    prompt_id = str(payload.get("prompt_id") or "").strip()
    if not prompt_id:
        raise DirectorError("ComfyUI no devolvió prompt_id.")
    logger.info(f"ComfyUI prompt submitted: {prompt_id}")
    limit = int(timeout if timeout is not None else settings.get("timeout") or DEFAULT_TIMEOUT)
    deadline = time.time() + max(30, limit)
    record = None
    while time.time() < deadline:
        history = requests.get(f"{base}/history/{prompt_id}", timeout=30)
        history.raise_for_status()
        data = history.json() if history.content else {}
        record = data.get(prompt_id) or (data if data.get("outputs") else None)
        if record:
            errors = _history_errors(record)
            status = (record.get("status") or {}).get("status_str") or ""
            completed = bool((record.get("status") or {}).get("completed")) or bool(
                record.get("outputs")
            )
            if errors or status == "error":
                blob = errors or status
                if re.search(r"out of memory|oom|CUDA", blob, flags=re.I):
                    raise DirectorError(
                        "ComfyUI se quedó sin memoria de video (OOM) al generar el clip. "
                        f"Detalle: {blob[:500]}"
                    )
                raise DirectorError(f"ComfyUI falló al generar el clip: {blob[:800]}")
            if completed and record.get("outputs"):
                break
        time.sleep(3)
    else:
        raise DirectorError(
            f"ComfyUI no terminó el clip en {limit}s (prompt_id={prompt_id}). Lo detuve para no seguir esperando."
        )
    info = _pick_output_file(record)
    saved = _download_view(base, info, output_path)
    duration = _as_float(bible["clip_seconds"], 3.0)
    try:
        from moviepy.video.io.VideoFileClip import VideoFileClip

        with VideoFileClip(str(saved)) as clip:
            if clip.duration and math.isfinite(clip.duration) and clip.duration > 0:
                duration = float(clip.duration)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"could not read generated clip duration: {exc}")
    logger.info(
        f"director clip saved in {time.time() - started:.1f}s: {saved} ({duration:.2f}s)"
    )
    return saved, duration


def download_videos_on_demand(
    *,
    task_id: str,
    search_terms: List[str],
    video_aspect: VideoAspect,
    audio_duration: float,
    max_clip_duration: int,
    material_directory: str,
    **_: Any,
) -> List[str]:
    from app.services import material as material_service

    settings = comfyui_settings()
    bible = load_bible(settings.get("bible"))
    if max_clip_duration:
        bible["clip_seconds"] = min(5.0, max(3.0, float(max_clip_duration)))
    video_paths: list[str] = []
    material_sources: list[dict[str, Any]] = []
    try:
        required = float(audio_duration)
    except (TypeError, ValueError):
        required = 0.0
    if not math.isfinite(required) or required <= 0:
        logger.warning("skip ComfyUI generation because audio duration is not positive")
        material_service._persist_material_sources(task_id, material_sources)
        return video_paths

    save_dir = material_directory or utils.task_dir(task_id)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    chain = bool(settings.get("chain_scenes", True))
    previous_frame: str | None = bible.get("reference_image") or None
    if previous_frame and not Path(previous_frame).is_file():
        logger.warning(f"bible reference image is missing, first scene will be text-only: {previous_frame}")
        previous_frame = None

    total = 0.0
    scenes = [str(term).strip() for term in search_terms if str(term).strip()]
    if not scenes:
        raise DirectorError("No hay escenas o palabras clave para generar en ComfyUI.")

    try:
        prompts: list[str] = []
        for index, scene in enumerate(scenes):
            prompts.append(
                scene_prompt_via_llm(
                    bible,
                    scene,
                    keep_ollama_loaded=index < len(scenes) - 1,
                )
            )
        unload_ollama_models()

        for index, (scene, prompt) in enumerate(zip(scenes, prompts), start=1):
            output = Path(save_dir) / f"comfyui_{task_id}_{index:02d}.mp4"
            first_frame = previous_frame if (chain or index == 1) else None
            saved, duration = submit_minimax_clip(
                prompt=prompt,
                bible=bible,
                output_path=output,
                first_frame_path=first_frame,
                video_aspect=video_aspect,
            )
            video_paths.append(str(saved))
            item = MaterialInfo(
                provider="comfyui",
                url=str(saved),
                duration=int(max(1, round(duration))),
                source_info={
                    "provider": "comfyui",
                    "scene": scene,
                    "prompt": prompt,
                    "bible": bible.get("path") or "",
                },
            )
            try:
                material_sources.append(
                    material_service._material_source_record(item, str(saved))
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"failed to record comfyui source: {exc}")
            if chain:
                frame_path = Path(save_dir) / f"comfyui_{task_id}_{index:02d}_last.png"
                previous_frame = str(extract_last_frame(str(saved), frame_path))
            total += min(float(bible["clip_seconds"]), duration)
            if total >= required:
                break
    finally:
        material_service._persist_material_sources(task_id, material_sources)
        unload_ollama_models()
    logger.success(f"generated {len(video_paths)} ComfyUI clips")
    return video_paths


CLONED_VOICE_NAME = "chatterbox:cloned-Female"
CHATTERBOX_URL_DEFAULT = "http://127.0.0.1:4123"
CHATTERBOX_VOICE_REF_POINTER = Path.home() / "Videos/chatterbox/voice_ref.txt"
CHATTERBOX_SERVE_SCRIPT = repo_root() / "scripts" / "chatterbox_serve.sh"


def voices_dir() -> Path:
    path = bibles_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_active_voice_ref(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    CHATTERBOX_VOICE_REF_POINTER.parent.mkdir(parents=True, exist_ok=True)
    CHATTERBOX_VOICE_REF_POINTER.write_text(str(resolved), encoding="utf-8")
    return resolved


def current_voice_ref(bible: dict | None = None) -> str:
    if bible and bible.get("voice_ref") and Path(str(bible["voice_ref"])).is_file():
        return str(Path(bible["voice_ref"]).expanduser())
    try:
        loaded = load_bible()
        if loaded.get("voice_ref") and Path(loaded["voice_ref"]).is_file():
            return str(loaded["voice_ref"])
    except Exception:
        pass
    if CHATTERBOX_VOICE_REF_POINTER.is_file():
        raw = CHATTERBOX_VOICE_REF_POINTER.read_text(encoding="utf-8").strip()
        if raw and Path(raw).is_file():
            return raw
    return ""


def extract_voice_reference(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    max_seconds: float = 25.0,
    start_seconds: float = 0.5,
) -> Path:
    """Extract a clean mono voice sample with ffmpeg (no Demucs)."""
    src = Path(source).expanduser()
    if not src.is_file():
        raise DirectorError(f"No encuentro el archivo de voz: {src}")
    if destination is None:
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", src.stem.lower()).strip("-") or "voz"
        destination = voices_dir() / f"{slug}.wav"
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Trim leading/trailing silence and cap at ~10-30 seconds, mono 16 kHz.
    filt = (
        "silenceremove=start_periods=1:start_silence=0.25:start_threshold=-40dB:"
        "detection=peak,"
        "areverse,"
        "silenceremove=start_periods=1:start_silence=0.25:start_threshold=-40dB:"
        "detection=peak,"
        "areverse"
    )
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(max(0.0, float(start_seconds))),
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filt,
        "-t",
        str(max(10.0, min(30.0, float(max_seconds)))),
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1000:
        # Retry without silenceremove in case the filter failed on this ffmpeg build.
        command = [
            "ffmpeg", "-y", "-ss", "0.3", "-i", str(src),
            "-vn", "-ac", "1", "-ar", "16000",
            "-t", str(max(10.0, min(30.0, float(max_seconds)))),
            str(dest),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size < 1000:
        raise DirectorError(
            "No pude extraer el audio de referencia con ffmpeg: "
            + (result.stderr or result.stdout or "")[-400:]
        )
    set_active_voice_ref(dest)
    return dest


def save_voice_ref_in_bible(voice_path: str | Path, bible_path: str | Path | None = None) -> Path:
    path = resolve_bible_path(str(bible_path) if bible_path else None)
    bible = load_bible(path)
    bible["voice_ref"] = str(Path(voice_path).expanduser().resolve())
    set_active_voice_ref(bible["voice_ref"])
    return save_bible(path, bible)


def unload_chatterbox() -> None:
    url = str(config.chatterbox.get("base_url") or CHATTERBOX_URL_DEFAULT).rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    try:
        response = requests.post(f"{url}/free", timeout=10)
        if response.status_code >= 400:
            logger.warning(f"Chatterbox /free returned HTTP {response.status_code}")
        else:
            logger.info("freed Chatterbox GPU memory")
    except requests.RequestException as exc:
        logger.debug(f"chatterbox /free skipped: {exc}")


def ensure_chatterbox_server(timeout: float = 20.0) -> str:
    """Start the local Chatterbox server if it is down. Returns the base URL without /v1."""
    raw = str(config.chatterbox.get("base_url") or "http://127.0.0.1:4123/v1").strip().rstrip("/")
    health_base = raw[:-3] if raw.endswith("/v1") else raw
    try:
        response = requests.get(f"{health_base}/health", timeout=2)
        if response.ok:
            return health_base
    except requests.RequestException:
        pass
    script = CHATTERBOX_SERVE_SCRIPT
    if not script.is_file():
        raise DirectorError(f"No encuentro el script de Chatterbox: {script}")
    log_path = Path.home() / "Videos/chatterbox/server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "ab")
    subprocess.Popen(
        ["sh", str(script)],
        stdout=log_handle,
        stderr=log_handle,
        start_new_session=True,
        cwd=str(Path.home() / "Videos/chatterbox"),
    )
    deadline = time.time() + timeout
    last_error = "timeout"
    while time.time() < deadline:
        try:
            response = requests.get(f"{health_base}/health", timeout=2)
            if response.ok:
                logger.info(f"Chatterbox server ready at {health_base}")
                return health_base
            last_error = f"HTTP {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise DirectorError(
        f"No pude levantar el servidor de Chatterbox en {health_base}: {last_error}. "
        f"Revisá {log_path}"
    )


def _parse_clip_window(start_seconds: float | None, end_seconds: float | None) -> tuple[float, float]:
    start = max(0.0, float(start_seconds or 0.0))
    end = float(end_seconds or 0.0)
    if end > start:
        duration = min(30.0, max(3.0, end - start))
    else:
        duration = 25.0
    return start, duration


def download_source_from_url(
    url: str,
    dest_dir: str | Path | None = None,
    *,
    save_video: bool = False,
) -> tuple[Path, Path | None]:
    """Download audio (and optionally a modest video) with yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise DirectorError(
            "Falta yt-dlp en el entorno de MoneyPrinterTurbo. "
            "Instalálo con: uv pip install --python .venv/bin/python yt-dlp"
        ) from exc

    source = (url or "").strip()
    if not re.match(r"https?://", source, flags=re.I):
        raise DirectorError("Pegá una URL válida (YouTube u otro sitio que soporte yt-dlp).")

    dest = Path(dest_dir) if dest_dir else voices_dir() / "descarga"
    dest.mkdir(parents=True, exist_ok=True)

    def _run(opts: dict) -> Path:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(source, download=True)
            prepared = Path(ydl.prepare_filename(info))
        if prepared.is_file():
            return prepared
        # prepare_filename may not match after format merge; take newest file in dest.
        files = [path for path in dest.iterdir() if path.is_file()]
        if not files:
            raise DirectorError("yt-dlp terminó sin dejar un archivo.")
        return max(files, key=lambda path: path.stat().st_mtime)

    common = {
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
        "overwrites": True,
        "no_warnings": True,
        "restrictfilenames": True,
    }
    audio_opts = {
        **common,
        "format": "bestaudio/best",
        "outtmpl": str(dest / "source_audio.%(ext)s"),
    }
    try:
        audio_path = _run(audio_opts)
    except Exception as exc:
        raise DirectorError(f"No pude descargar el audio: {exc}") from exc

    video_path = None
    if save_video:
        video_opts = {
            **common,
            "format": "bv*[height<=720]+ba/b[height<=720]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(dest / "source_video.%(ext)s"),
        }
        try:
            video_path = _run(video_opts)
        except Exception as exc:
            logger.warning(f"optional video download failed: {exc}")
            video_path = None
    return audio_path, video_path


