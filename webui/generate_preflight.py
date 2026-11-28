"""Pre-flight checks and field highlights for Generar video."""

from __future__ import annotations

import html
import re
from typing import Any

import requests
import streamlit as st

from app.config import config
from app.services import comfyui_director
from app.services import director_characters
from app.services import voice

SESSION_KEY = "director_field_errors"

# field -> Streamlit widget keys to outline in red
FIELD_KEYS: dict[str, list[str]] = {
    "video_subject": ["video_subject", "video_subject_field"],
    "video_script": ["video_script"],
    "video_terms": ["video_terms", "auto_generate_script", "auto_generate_terms"],
    "video_source": ["video_source_select"],
    "voice": ["tts_server_select"],
    "llm": ["llm_provider_select"],
    "bible": ["director_bible_select"],
    "protagonists": ["director_protagonists"],
    "lip_sync": ["director_lip_sync"],
}

_RUNTIME_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"empty text content|failed to generate video search terms", re.I),
        "llm",
        "El modelo de Ollama devolvió una respuesta vacía al armar las palabras clave. "
        "Suele pasar con modelos que razonan (como qwen3-vl). Escribí las palabras clave "
        "a mano, reintentá, o cambiá el modelo en Ajustes.",
    ),
    (
        re.compile(r"failed to generate video script", re.I),
        "video_subject",
        "No pude generar el guion con el LLM configurado. Revisá el proveedor y el modelo "
        "en Ajustes, o escribí el guion a mano.",
    ),
    (
        re.compile(r"ComfyUI|no pude conectar con ComfyUI|:8188", re.I),
        "video_source",
        "Elegiste ComfyUI como fuente, pero no responde en :8188. Arrancá ComfyUI o cambiá la fuente.",
    ),
    (
        re.compile(r"biblia|bible", re.I),
        "bible",
        "No encuentro la biblia de video. Elegí un archivo de biblia o creá uno en este panel.",
    ),
    (
        re.compile(r"MuseTalk|lip-?sync", re.I),
        "lip_sync",
        "Falló el lip-sync (MuseTalk). Podés apagar 🎬 Lip-sync e intentar de nuevo, o revisá que el servidor :4124 esté bien.",
    ),
    (
        re.compile(r"Chatterbox|voice_ref|voz clonada", re.I),
        "voice",
        "Falló Chatterbox. Confirmá que el servidor en :4123 esté arriba y que el protagonista tenga una muestra de voz.",
    ),
    (
        re.compile(r"pexels", re.I),
        "video_source",
        "La fuente Pexels necesita una API key en Ajustes.",
    ),
    (
        re.compile(r"pixabay", re.I),
        "video_source",
        "La fuente Pixabay necesita una API key en Ajustes.",
    ),
]


def _error(field: str, message: str, extra_keys: list[str] | None = None) -> dict[str, Any]:
    keys = list(FIELD_KEYS.get(field, [field]))
    if field == "llm":
        provider = str(config.app.get("llm_provider") or "ollama").lower()
        keys.extend(
            [
                f"{provider}_model_name_input",
                f"{provider}_api_key_input",
            ]
        )
    if field == "voice":
        server = str(
            st.session_state.get("tts_server_select")
            or config.ui.get("tts_server")
            or "chatterbox"
        )
        keys.append(f"speech_synthesis_select_{server}")
    if extra_keys:
        keys.extend(extra_keys)
    # Unique, CSS-safe keys only.
    clean: list[str] = []
    for key in keys:
        safe = re.sub(r"[^A-Za-z0-9_-]", "", key)
        if safe and safe not in clean:
            clean.append(safe)
    return {"field": field, "message": message, "css_keys": clean}


def get_errors() -> list[dict[str, Any]]:
    raw = st.session_state.get(SESSION_KEY) or []
    return [item for item in raw if isinstance(item, dict) and item.get("message")]


def set_errors(errors: list[dict[str, Any]] | None) -> None:
    if errors:
        st.session_state[SESSION_KEY] = errors
    else:
        st.session_state.pop(SESSION_KEY, None)


def clear_errors() -> None:
    st.session_state.pop(SESSION_KEY, None)


def inject_styles() -> None:
    errors = get_errors()
    if not errors:
        return
    keys: list[str] = []
    for item in errors:
        for key in item.get("css_keys") or []:
            if key not in keys:
                keys.append(key)
    if not keys:
        return
    selectors = ", ".join(f'div[class*="st-key-{key}"]' for key in keys)
    st.markdown(
        f"""
<style>
{selectors} {{
    outline: 2px solid #DC2626 !important;
    border-radius: 10px !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def render_summary() -> None:
    errors = get_errors()
    if not errors:
        return
    items = "".join(
        f"<li>{html.escape(str(item['message']))}</li>" for item in errors
    )
    st.markdown(
        f"""
<div class="director-preflight-summary">
  <strong>No pude generar el video. Revisá lo marcado en rojo:</strong>
  <ul>{items}</ul>
</div>
""",
        unsafe_allow_html=True,
    )


def field_caption(field: str) -> None:
    messages = [item["message"] for item in get_errors() if item.get("field") == field]
    if not messages:
        return
    text = html.escape(messages[0])
    st.markdown(
        f'<p class="director-field-error-msg">{text}</p>',
        unsafe_allow_html=True,
    )


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code < 500
    except requests.RequestException:
        return False


def _ollama_problem() -> str | None:
    provider = str(config.app.get("llm_provider") or "").lower()
    if provider != "ollama":
        return None
    from app.services.llm import get_llm_provider

    spec = get_llm_provider("ollama")
    configured = str(config.app.get("ollama_base_url") or "").strip()
    base = configured or (spec.resolve_base_url(configured) if spec else "")
    if not base:
        base = config.get_default_ollama_base_url()
    root = base.rstrip("/").removesuffix("/v1")
    model = str(config.app.get("ollama_model_name") or "").strip()
    try:
        response = requests.get(f"{root}/api/tags", timeout=2)
        if response.status_code >= 400:
            return (
                f"Ollama respondió HTTP {response.status_code} en {root}. "
                "Revisá el proveedor LLM en Ajustes."
            )
        names = [
            str(item.get("name") or item.get("model") or "")
            for item in (response.json().get("models") or [])
        ]
        if model and not any(model == name or model in name or name.startswith(model) for name in names):
            return (
                f"Ollama está arriba, pero no encuentro el modelo «{model}». "
                "Cargalo o elegí otro en Ajustes."
            )
    except requests.RequestException:
        return (
            "No pude conectar con Ollama. ¿Está corriendo? Revisá el proveedor LLM en Ajustes."
        )
    return None


def _other_llm_problem() -> str | None:
    provider = str(config.app.get("llm_provider") or "openai").lower()
    if provider in {"", "ollama"}:
        return None
    from app.services.llm import get_llm_provider

    spec = get_llm_provider(provider)
    if spec is None:
        return f"El proveedor LLM «{provider}» no está soportado. Elegí otro en Ajustes."
    if spec.requires_api_key and not str(
        config.app.get(spec.config_key("api_key"), "") or ""
    ).strip():
        return f"Falta la API key de {provider}. Cargala en Ajustes antes de generar."
    if spec.requires_model_name and not str(
        config.app.get(spec.config_key("model_name"), "") or ""
    ).strip():
        return f"Falta el nombre del modelo de {provider}. Completalo en Ajustes."
    return None


def _chatterbox_health_url() -> str:
    base = str(config.chatterbox.get("base_url") or "http://127.0.0.1:4123/v1").strip()
    root = base.rstrip("/").removesuffix("/v1")
    return f"{root}/health"


def validate_generation(params: Any, *, voice_mode: str | None = None) -> list[dict[str, Any]]:
    """Return field errors; an empty list means the task may start."""
    errors: list[dict[str, Any]] = []
    subject = str(getattr(params, "video_subject", "") or "").strip()
    script = str(getattr(params, "video_script", "") or "").strip()
    source = str(getattr(params, "video_source", "") or "").strip()
    voice_name = str(getattr(params, "voice_name", "") or "").strip()
    settings = comfyui_director.comfyui_settings()
    protagonists = director_characters.protagonists_from_settings(settings)
    lip_sync = bool(settings.get("lip_sync", False)) or bool(
        st.session_state.get("director_lip_sync")
    )

    if not subject and not script:
        errors.append(
            _error(
                "video_subject",
                "Falta el tema o el guion. Escribí uno de los dos para poder generar.",
            )
        )
        errors.append(
            _error(
                "video_script",
                "Falta el tema o el guion. Escribí uno de los dos para poder generar.",
            )
        )

    llm_error = _ollama_problem() or _other_llm_problem()
    if llm_error:
        errors.append(_error("llm", llm_error))

    if source == "comfyui":
        ready = comfyui_director.validate_ready()
        if ready:
            field = "bible" if "biblia" in ready.lower() or "bible" in ready.lower() else "video_source"
            errors.append(_error(field, ready))
    elif source == "pexels" and not str(config.app.get("pexels_api_keys") or "").strip():
        errors.append(
            _error(
                "video_source",
                "Elegiste Pexels, pero falta la API key en Ajustes.",
            )
        )
    elif source == "pixabay" and not str(config.app.get("pixabay_api_keys") or "").strip():
        errors.append(
            _error(
                "video_source",
                "Elegiste Pixabay, pero falta la API key en Ajustes.",
            )
        )
    elif source == "coverr" and not str(config.app.get("coverr_api_keys") or "").strip():
        errors.append(
            _error(
                "video_source",
                "Elegiste Coverr, pero falta la API key en Ajustes.",
            )
        )

    if protagonists and source != "comfyui":
        errors.append(
            _error(
                "video_source",
                "Hay protagonistas elegidos, pero la fuente no es ComfyUI. "
                "Cambiá la fuente a ComfyUI o sacá los protagonistas.",
            )
        )
        errors.append(
            _error(
                "protagonists",
                "Los protagonistas solo se usan cuando la fuente de video es ComfyUI.",
            )
        )

    if voice.is_chatterbox_voice(voice_name) or (
        protagonists and source == "comfyui"
    ):
        if not _http_ok(_chatterbox_health_url()):
            errors.append(
                _error(
                    "voice",
                    "Elegiste Chatterbox, pero no responde en :4123. Arrancá el servidor de voz.",
                )
            )
        needs_ref = voice.is_chatterbox_voice(voice_name) and "cloned" in voice_name.lower()
        if needs_ref and not comfyui_director.current_voice_ref():
            if not any(char.voice_ref for char in protagonists):
                errors.append(
                    _error(
                        "voice",
                        "La voz clonada de Chatterbox necesita una muestra (voice_ref) "
                        "en el protagonista o en la biblia.",
                    )
                )
                if protagonists:
                    errors.append(
                        _error(
                            "protagonists",
                            "El personaje elegido no tiene muestra de voz. Cargala en 🎬 Personajes.",
                        )
                    )

    if lip_sync:
        if source != "comfyui":
            errors.append(
                _error(
                    "lip_sync",
                    "El lip-sync solo corre con fuente ComfyUI. Apagalo o cambiá la fuente.",
                )
            )
        if not protagonists:
            errors.append(
                _error(
                    "lip_sync",
                    "El lip-sync está encendido, pero no hay un protagonista. Elegí uno o apagalo.",
                )
            )
        if not (
            voice.is_chatterbox_voice(voice_name)
            or any(char.voice_ref for char in protagonists)
        ):
            errors.append(
                _error(
                    "lip_sync",
                    "El lip-sync necesita la voz de Chatterbox del protagonista. "
                    "Elegí Chatterbox o apagá el interruptor.",
                )
            )

    if len(protagonists) >= 2 and script:
        tagged = []
        for line in script.splitlines():
            parsed = director_characters.parse_dialogue_line(line)
            if parsed:
                tagged.append(parsed[0])
        if not tagged:
            names = ", ".join(char.name for char in protagonists)
            errors.append(
                _error(
                    "video_script",
                    f"Hay varios protagonistas ({names}), pero el guion no tiene réplicas "
                    f"`Nombre:`. Escribí cada línea como `{protagonists[0].name}: …`.",
                )
            )
        else:
            missing = [
                char.name
                for char in protagonists
                if director_characters.character_for_label(char.name, protagonists)
                and not any(
                    director_characters.character_for_label(label, protagonists) is char
                    or director_characters._name_matches(label, char)
                    for label in tagged
                )
            ]
            if missing:
                errors.append(
                    _error(
                        "video_script",
                        "El guion en formato diálogo no incluye a: "
                        + ", ".join(missing)
                        + ". Agregá réplicas `Nombre:` para cada protagonista.",
                    )
                )

    # Deduplicate identical messages.
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in errors:
        mark = (item["field"], item["message"])
        if mark in seen:
            continue
        seen.add(mark)
        unique.append(item)
    return unique


def map_runtime_error(error: str, stage: str | None = None) -> list[dict[str, Any]]:
    text = str(error or "").strip()
    if not text:
        return []
    # Never show a raw traceback as the user-facing message.
    brief = text.split("Traceback", 1)[0].strip() or text
    for pattern, field, message in _RUNTIME_RULES:
        if pattern.search(text):
            return [_error(field, message)]
    if stage in {"script", "terms"}:
        return [
            _error(
                "llm",
                "Falló el LLM al preparar el guion o las palabras clave. "
                "Revisá el proveedor en Ajustes, o completalos a mano.",
            )
        ]
    if stage in {"audio"}:
        return [_error("voice", f"Falló el audio: {brief[:240]}")]
    if stage in {"materials"}:
        return [_error("video_source", f"Falló la fuente de video: {brief[:240]}")]
    return [
        _error(
            "video_source",
            f"La generación falló: {brief[:240]}",
        )
    ]


def remember_runtime_failure(error: str, stage: str | None = None) -> list[dict[str, Any]]:
    mapped = map_runtime_error(error, stage)
    if mapped:
        set_errors(mapped)
    return mapped
