"""WebUI del fork director: sección marcada con 🎬 y borde violeta."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

from app.config import config
from app.services import comfyui_director, director_face


def _persist(key: str, value) -> None:
    config.update_config_nonblocking(config.comfyui, key, value)


def _provider_options() -> dict[str, str]:
    return {
        "ollama": "Ollama (local)",
        "nvidia": "NVIDIA (build.nvidia.com)",
        "openrouter": "OpenRouter (gratis)",
        "configured": "Proveedor ya configurado",
    }


def render(params) -> None:
    settings = comfyui_director.comfyui_settings()
    rec = comfyui_director.recommended_models()
    providers = _provider_options()

    st.markdown('<div class="director-fork-marker"></div>', unsafe_allow_html=True)
    st.caption("Nuevo (fork director)")
    st.markdown("**🎬 Director ComfyUI + MiniMax H3**")
    st.caption(
        "Recomendado en esta máquina: Ollama local. "
        f"Biblia/guion → `{rec['bible']['ollama']}` (8.8B, ya instalado). "
        f"Prompts visuales → `{rec['prompt']['ollama']}` (3B, rápido, ya instalado). "
        "Si Ollama falla: NVIDIA (clave `nvidia_api_key`) y después OpenRouter free. "
        "No se puede tener el LLM y MiniMax H3 juntos en la 3060: el fork descarga Ollama "
        "(`keep_alive=0`) antes de ComfyUI y libera ComfyUI (`/free`) antes de usar el LLM."
    )

    bible_files = comfyui_director.list_bible_files()
    current_bible = Path(str(settings.get("bible") or "bibles/example.toml")).name
    if current_bible not in bible_files and bible_files:
        bible_files.append(current_bible)
    selected_bible = st.selectbox(
        "🎬 Archivo de biblia",
        options=bible_files or ["example.toml"],
        index=max(0, (bible_files or ["example.toml"]).index(current_bible))
        if current_bible in (bible_files or ["example.toml"])
        else 0,
        key="director_bible_select",
        help="Plantilla de estilo y continuidad que se aplica a todas las escenas.",
    )
    bible_rel = f"bibles/{selected_bible}"
    _persist("bible", bible_rel)

    idea_default = "\n\n".join(
        part
        for part in (
            str(getattr(params, "video_subject", "") or "").strip(),
            str(getattr(params, "video_script", "") or "").strip(),
        )
        if part
    )
    if "director_idea_text" not in st.session_state:
        st.session_state["director_idea_text"] = idea_default
    st.text_area(
        "🎬 Idea o estrategia para la biblia",
        key="director_idea_text",
        height=100,
        help="El botón de abajo usa este texto y el LLM de biblia/guion.",
    )
    if st.button("🎬 Generar biblia con IA", key="director_generate_bible", type="secondary"):
        try:
            with st.spinner("Generando la biblia (libero ComfyUI de la VRAM antes)..."):
                path, bible, used_provider, used_model = (
                    comfyui_director.generate_bible_from_idea(
                        st.session_state.get("director_idea_text") or "",
                        provider=st.session_state.get("director_bible_provider"),
                        model=st.session_state.get("director_bible_model"),
                    )
                )
            rel = str(path.relative_to(comfyui_director.repo_root()))
            _persist("bible", rel)
            st.session_state["director_bible_text"] = comfyui_director.dump_bible_toml(bible)
            st.session_state["director_bible_edit_path"] = rel
            st.success(
                f"Biblia guardada en {rel} (vía {used_provider} · {used_model}). "
                "Podés editarla abajo."
            )
            st.rerun()
        except comfyui_director.DirectorError as exc:
            st.error(str(exc))

    edit_path = st.session_state.get("director_bible_edit_path") or bible_rel
    try:
        current_text = comfyui_director.resolve_bible_path(edit_path).read_text(
            encoding="utf-8"
        )
    except OSError:
        current_text = ""
    if (
        "director_bible_editor" not in st.session_state
        or st.session_state.get("director_bible_loaded_path") != edit_path
        or "director_bible_text" in st.session_state
    ):
        st.session_state["director_bible_editor"] = st.session_state.pop(
            "director_bible_text", current_text
        )
        st.session_state["director_bible_loaded_path"] = edit_path
    text = st.text_area(
        "🎬 Biblia (editable)",
        height=220,
        key="director_bible_editor",
    )
    if st.button("🎬 Guardar biblia", key="director_save_bible"):
        try:
            saved = comfyui_director.save_bible(edit_path, text)
            rel = str(saved.relative_to(comfyui_director.repo_root()))
            _persist("bible", rel)
            st.session_state.pop("director_bible_text", None)
            st.success(f"Guardé la biblia en {rel}.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"No pude guardar la biblia: {exc}")

    url = st.text_input(
        "🎬 URL de ComfyUI",
        value=str(settings.get("url") or comfyui_director.DEFAULT_COMFY_URL),
        key="director_comfy_url",
    )
    _persist("url", url.strip() or comfyui_director.DEFAULT_COMFY_URL)

    chain = st.checkbox(
        "🎬 Encadenar escenas (último fotograma → siguiente clip)",
        value=bool(settings.get("chain_scenes", True)),
        key="director_chain_scenes",
        help="Usa el modo first-frame de MiniMax H3 para sostener continuidad visual.",
    )
    _persist("chain_scenes", chain)

    st.markdown("**🎬 Modelos por tarea**")
    col_a, col_b = st.columns(2)
    with col_a:
        bible_provider = st.selectbox(
            "🎬 LLM biblia y guion",
            options=list(providers.keys()),
            format_func=lambda key: providers[key],
            index=list(providers.keys()).index(settings.get("bible_llm_provider") or "ollama")
            if (settings.get("bible_llm_provider") or "ollama") in providers
            else 0,
            key="director_bible_provider",
        )
        _persist("bible_llm_provider", bible_provider)
        default_bible_model = rec["bible"].get(bible_provider, rec["bible"]["ollama"])
        stored_bible_model = str(settings.get("bible_llm_model") or default_bible_model)
        if st.session_state.get("director_bible_provider_prev") != bible_provider:
            stored_bible_model = default_bible_model
            st.session_state["director_bible_provider_prev"] = bible_provider
        bible_model = st.text_input(
            "🎬 Modelo biblia/guion",
            value=stored_bible_model,
            key="director_bible_model",
            help=(
                f"Recomendado local: {rec['bible']['ollama']}. "
                f"NVIDIA: {rec['bible']['nvidia']}. "
                f"OpenRouter: {rec['bible']['openrouter']}."
            ),
        )
        _persist("bible_llm_model", bible_model.strip() or default_bible_model)
    with col_b:
        prompt_provider = st.selectbox(
            "🎬 LLM prompts visuales",
            options=list(providers.keys()),
            format_func=lambda key: providers[key],
            index=list(providers.keys()).index(settings.get("prompt_llm_provider") or "ollama")
            if (settings.get("prompt_llm_provider") or "ollama") in providers
            else 0,
            key="director_prompt_provider",
        )
        _persist("prompt_llm_provider", prompt_provider)
        default_prompt_model = rec["prompt"].get(prompt_provider, rec["prompt"]["ollama"])
        stored_prompt_model = str(settings.get("prompt_llm_model") or default_prompt_model)
        if st.session_state.get("director_prompt_provider_prev") != prompt_provider:
            stored_prompt_model = default_prompt_model
            st.session_state["director_prompt_provider_prev"] = prompt_provider
        prompt_model = st.text_input(
            "🎬 Modelo prompts visuales",
            value=stored_prompt_model,
            key="director_prompt_model",
            help=(
                f"Recomendado local: {rec['prompt']['ollama']}. "
                f"NVIDIA: {rec['prompt']['nvidia']}. "
                f"OpenRouter: {rec['prompt']['openrouter']}."
            ),
        )
        _persist("prompt_llm_model", prompt_model.strip() or default_prompt_model)

    st.caption(
        "Si el modelo preferido falla, el orden de respaldo es: "
        "Ollama local → NVIDIA free → OpenRouter free → el proveedor que ya tengas configurado "
        "(OpenRouter, Ollama u OpenAI-compatible, por ejemplo MiniMax)."
    )

    st.markdown("**🎬 Clonar personaje**")
    st.caption(
        "Solo usá voz y cara de personas que te hayan dado consentimiento. "
        "El mismo video, foto o URL sirve para extraer las dos referencias. "
        "Si en el audio hablan varias personas, elegí un inicio y un fin donde "
        "hable solo quien querés clonar (por ahora no hay diarización de voces)."
    )
    uploaded_voice = st.file_uploader(
        "🎬 Video, audio o foto para clonar voz y cara",
        type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mkv", "webm", "mov", "jpg", "jpeg", "png", "webp", "bmp"],
        key="director_voice_uploader",
    )
    voice_url = st.text_input(
        "🎬 URL de YouTube u otro video",
        key="director_voice_url",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Se descarga el audio y un video corto (hasta 720p) con yt-dlp para extraer voz y cara.",
    )
    start_col, end_col, video_col = st.columns(3)
    voice_start = start_col.number_input(
        "🎬 Inicio (s)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="director_voice_start",
        help="Segundo desde el que cortar la muestra de voz (y el tramo a muestrear para caras).",
    )
    voice_end = end_col.number_input(
        "🎬 Fin (s)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="director_voice_end",
        help="0 = tomar unos 25 s desde el inicio indicado.",
    )
    save_source_video = video_col.checkbox(
        "🎬 Conservar el video descargado",
        key="director_save_source_video",
        help="Si está apagado, igual bajo un video temporal para detectar caras y después lo puedo borrar.",
    )
    if st.button("🎬 Extraer voz y caras", key="director_extract_voice"):
        if uploaded_voice is None and not str(voice_url or "").strip():
            st.error("Subí un archivo o pegá una URL de YouTube (u otro video).")
        else:
            try:
                start_s, duration_s = comfyui_director._parse_clip_window(
                    voice_start, voice_end
                )
                slug = re.sub(
                    r"[^a-zA-Z0-9_-]+",
                    "-",
                    (Path(str(settings.get("bible") or "personaje")).stem or "personaje").lower(),
                ).strip("-") or "personaje"
                dest_dir = comfyui_director.voices_dir() / slug
                image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                video_suffixes = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
                notes = []
                visual_source = None
                video_path = None
                with st.spinner("Extrayendo voz y detectando caras..."):
                    if str(voice_url or "").strip():
                        source, video_path = comfyui_director.download_source_from_url(
                            voice_url,
                            dest_dir,
                            save_video=True,
                        )
                    else:
                        suffix = Path(uploaded_voice.name or "fuente.bin").suffix.lower() or ".bin"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        source = dest_dir / f"upload{suffix}"
                        source.write_bytes(uploaded_voice.getbuffer())
                        video_path = source if suffix in video_suffixes else None
                    is_image = source.suffix.lower() in image_suffixes
                    if is_image:
                        notes.append("El archivo es una foto: no hay audio para clonar la voz.")
                    else:
                        dest = comfyui_director.extract_voice_reference(
                            source,
                            dest_dir / "voice_ref.wav",
                            max_seconds=duration_s,
                            start_seconds=start_s,
                        )
                        comfyui_director.save_voice_ref_in_bible(dest, bible_rel)
                        st.session_state["tts_server_select"] = "chatterbox"
                        st.session_state["speech_synthesis_select_chatterbox"] = (
                            comfyui_director.CLONED_VOICE_NAME
                        )
                        notes.append(f"Voz: {dest}")
                        st.audio(str(dest))
                    if video_path and Path(video_path).is_file():
                        visual_source = Path(video_path)
                    elif is_image:
                        visual_source = Path(source)
                    elif source.suffix.lower() in video_suffixes:
                        visual_source = Path(source)
                    if visual_source is not None:
                        face_dir = director_face.faces_dir() / slug
                        candidates = director_face.extract_face_candidates(
                            visual_source, face_dir
                        )
                        st.session_state["director_face_candidates"] = [
                            {
                                "cluster_id": c.cluster_id,
                                "count": c.count,
                                "crop_path": c.crop_path,
                                "thumb_path": c.thumb_path,
                            }
                            for c in candidates
                        ]
                        if len(candidates) == 1:
                            saved = director_face.save_face_ref_in_bible(
                                candidates[0].crop_path, bible_rel
                            )
                            notes.append(
                                f"Había una sola persona: guardé su cara en {candidates[0].crop_path}."
                            )
                        else:
                            notes.append(
                                f"Encontré {len(candidates)} personas. Elegí cuál clonar."
                            )
                    else:
                        st.session_state.pop("director_face_candidates", None)
                        notes.append(
                            "No había un video o una foto para buscar caras. "
                            "Si pegaste una URL, revisá que yt-dlp haya podido bajar el video."
                        )
                    if video_path and not save_source_video and str(voice_url or "").strip():
                        try:
                            Path(video_path).unlink(missing_ok=True)
                        except OSError:
                            pass
                extra = " ".join(notes)
                st.success(extra or "Listo.")
            except Exception as exc:
                st.error(str(exc))

    candidates = st.session_state.get("director_face_candidates") or []
    if len(candidates) > 1:
        st.markdown("**🎬 ¿A quién querés clonar?**")
        st.caption("Una miniatura por persona. Elegí la cara que va a la biblia.")
        cols = st.columns(min(4, len(candidates)))
        for index, cand in enumerate(candidates):
            with cols[index % len(cols)]:
                if Path(cand["thumb_path"]).is_file():
                    st.image(cand["thumb_path"])
                st.caption(f"Persona {cand['cluster_id']} · {cand['count']} tomas")
                if st.button(
                    f"🎬 Usar persona {cand['cluster_id']}",
                    key=f"director_pick_face_{cand['cluster_id']}",
                ):
                    try:
                        director_face.save_face_ref_in_bible(cand["crop_path"], bible_rel)
                        st.success(f"Guardé la cara en {cand['crop_path']}.")
                    except Exception as exc:
                        st.error(str(exc))

    existing = comfyui_director.current_voice_ref()
    if existing and Path(existing).is_file():
        st.caption(f"Referencia de voz actual: {existing}")
        st.audio(existing)
    existing_face = director_face.current_face_ref()
    if existing_face and Path(existing_face).is_file():
        st.caption(f"Referencia de cara actual: {existing_face}")
        st.image(existing_face, width=160)
