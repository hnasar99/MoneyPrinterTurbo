"""WebUI del fork director: sección marcada con 🎬 y borde violeta."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import streamlit as st

from webui import generate_preflight

from app.config import config
from app.services import comfyui_director, director_characters, director_face


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
    generate_preflight.field_caption("bible")
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

    st.markdown("**🎬 Protagonistas**")
    characters = director_characters.list_characters()
    if not characters:
        st.caption(
            "Todavía no hay personajes en el catálogo. Crealos en la pestaña "
            "🎬 Personajes y después elegilos acá."
        )
        _persist("protagonists", "")
        _persist("lip_sync", False)
    else:
        labels = {char.slug: char.name for char in characters}
        saved = [
            part.strip()
            for part in str(settings.get("protagonists") or "").split(",")
            if part.strip()
        ]
        default = [slug for slug in saved if slug in labels]
        selected = st.multiselect(
            "🎬 Protagonistas del video",
            options=[char.slug for char in characters],
            default=default,
            format_func=lambda slug: labels.get(slug, slug),
            key="director_protagonists",
            help=(
                "Opcional. Con uno, se usa su cara (PuLID) y su voz (Chatterbox). "
                "Con dos o más, se alternan los planos y las réplicas (estilo diálogo). "
                "No hay dos caras en el mismo cuadro. El lip-sync se aplica por escena."
            ),
        )
        _persist("protagonists", ",".join(selected))
        if selected:
            st.session_state["tts_server_select"] = "chatterbox"
            st.session_state["speech_synthesis_select_chatterbox"] = (
                comfyui_director.CLONED_VOICE_NAME
            )
            cols = st.columns(min(4, len(selected)))
            for index, slug in enumerate(selected):
                char = director_characters.load_character(slug)
                if not char:
                    continue
                with cols[index % len(cols)]:
                    if char.thumb():
                        st.image(char.thumb(), width=120)
                    st.caption(char.name)
                    if not char.voice_ref:
                        st.caption("Sin voz todavía.")
            if len(selected) >= 2:
                st.caption(
                    "Varios protagonistas: el guion va en réplicas `Nombre:` y cada "
                    "escena muestra a una sola persona, alternando."
                )
        generate_preflight.field_caption("protagonists")
        # Lip-sync defaults to on when there is a protagonist (Chatterbox voice).
        if selected and not st.session_state.get("director_lip_sync_touched"):
            st.session_state["director_lip_sync"] = True
        lip_sync = st.checkbox(
            "🎬 Lip-sync",
            value=bool(st.session_state.get("director_lip_sync", bool(selected))),
            key="director_lip_sync",
            help=(
                "Después de MiniMax, MuseTalk 1.5 sincroniza la boca con el audio "
                "de Chatterbox de cada escena. Si el clip es más corto que la línea, "
                "se alarga con un ping-pong (ida y vuelta). Se activa solo con "
                "protagonista y voz clonada; podés apagarlo."
            ),
            on_change=lambda: st.session_state.update({"director_lip_sync_touched": True}),
        )
        _persist("lip_sync", bool(lip_sync))
        generate_preflight.field_caption("lip_sync")

    has_selection = bool(str(settings.get("protagonists") or "").strip())
    existing = comfyui_director.current_voice_ref()
    if existing and Path(existing).is_file() and not has_selection:
        st.caption(f"Referencia de voz de la biblia: {existing}")
        st.audio(existing)
    existing_face = director_face.current_face_ref()
    if existing_face and Path(existing_face).is_file() and not has_selection:
        st.caption(f"Referencia de cara de la biblia: {existing_face}")
        st.image(existing_face, width=160)


def render_placeholder_scenes() -> None:
    st.markdown('<div class="director-fork-marker"></div>', unsafe_allow_html=True)
    st.caption("Nuevo (fork director)")
    st.markdown("**🎬 Escenas**")
    st.info(
        "Próximamente: foto de una escena y asignación de varias personas a ese plano. "
        "Por ahora los personajes se arman en 🎬 Personajes y se eligen como protagonistas "
        "en 🎬 Generar video."
    )


def render_placeholder_improve() -> None:
    st.markdown('<div class="director-fork-marker"></div>', unsafe_allow_html=True)
    st.caption("Nuevo (fork director)")
    st.markdown("**🎬 Mejorar**")
    st.info(
        "Próximamente: upscale, interpolación de fotogramas y restauración de caras. "
        "Todavía no está en esta etapa."
    )


def render_characters() -> None:
    st.markdown('<div class="director-fork-marker"></div>', unsafe_allow_html=True)
    st.caption("Nuevo (fork director)")
    st.markdown("**🎬 Personajes**")
    st.caption(
        "Solo usá voz y cara de personas que te hayan dado consentimiento. "
        "Subí un video, una foto o pegá una URL: detecto las caras, separo locutores "
        "si hay audio, y armás un personaje por persona (estilo Google Fotos). "
        "Si el diarizado no alcanza, recortá un inicio y un fin donde hable solo "
        "quien querés clonar, o asigná a mano cada muestra de voz."
    )

    uploaded = st.file_uploader(
        "🎬 Video, audio o foto",
        type=["wav", "mp3", "m4a", "flac", "ogg", "mp4", "mkv", "webm", "mov", "jpg", "jpeg", "png", "webp", "bmp"],
        key="director_cast_uploader",
    )
    source_url = st.text_input(
        "🎬 URL de YouTube u otro video",
        key="director_cast_url",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    start_col, end_col, keep_col = st.columns(3)
    start_s = start_col.number_input(
        "🎬 Inicio (s)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="director_cast_start",
        help="Recorte opcional. También sirve como respaldo si hablan varios y el diarizado falla.",
    )
    end_s = end_col.number_input(
        "🎬 Fin (s)",
        min_value=0.0,
        value=0.0,
        step=1.0,
        key="director_cast_end",
        help="0 = usar el tramo por defecto (unos 25 s) o el video completo para caras.",
    )
    keep_video = keep_col.checkbox(
        "🎬 Conservar el video descargado",
        key="director_cast_keep_video",
    )
    if st.button("🎬 Detectar personas", key="director_detect_cast"):
        if uploaded is None and not str(source_url or "").strip():
            st.error("Subí un archivo o pegá una URL.")
        else:
            try:
                _extract_cast_to_session(
                    uploaded=uploaded,
                    source_url=str(source_url or "").strip(),
                    start_s=start_s,
                    end_s=end_s,
                    keep_video=bool(keep_video),
                )
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    _render_cast_draft()
    _render_character_catalog()


def _extract_cast_to_session(*, uploaded, source_url: str, start_s: float, end_s: float, keep_video: bool) -> None:
    start_clip, duration_s = comfyui_director._parse_clip_window(start_s, end_s)
    work_dir = director_characters.characters_dir() / "_inbox"
    work_dir.mkdir(parents=True, exist_ok=True)
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_suffixes = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
    video_path = None
    with st.spinner("Detectando caras y locutores..."):
        if source_url:
            source, video_path = comfyui_director.download_source_from_url(
                source_url, work_dir, save_video=True
            )
        else:
            suffix = Path(uploaded.name or "fuente.bin").suffix.lower() or ".bin"
            source = work_dir / f"upload{suffix}"
            source.write_bytes(uploaded.getbuffer())
            video_path = source if suffix in video_suffixes else None
        is_image = source.suffix.lower() in image_suffixes
        visual = None
        if video_path and Path(video_path).is_file():
            visual = Path(video_path)
        elif is_image or source.suffix.lower() in video_suffixes:
            visual = Path(source)

        faces = []
        if visual is not None:
            faces = director_face.extract_face_candidates(visual, work_dir / "faces")
        speakers = []
        audio_source = None if is_image else Path(source)
        if audio_source and audio_source.is_file():
            try:
                speakers = director_characters.diarize_speakers(
                    audio_source,
                    work_dir / "speakers",
                    start_seconds=start_clip,
                    duration_seconds=duration_s if end_s or start_s else None,
                    face_count=len(faces),
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"No pude separar locutores automáticamente: {exc}")
                speakers = []
            # Always keep the start/end (or default) crop as a manual fallback sample.
            try:
                manual = comfyui_director.extract_voice_reference(
                    audio_source,
                    work_dir / "speakers" / "recorte_manual.wav",
                    max_seconds=duration_s,
                    start_seconds=start_clip,
                )
                speakers.append(
                    director_characters.SpeakerCandidate(
                        speaker_id="recorte",
                        wav_path=str(manual),
                        count=1,
                        duration=0.0,
                        segments=[],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if not speakers:
                    st.warning(f"No pude recortar una muestra de voz: {exc}")
            try:
                total = director_characters.extract_full_voice_reference(
                    audio_source,
                    work_dir / "speakers" / "recorte_total.wav",
                    max_seconds=60.0,
                )
                speakers.append(
                    director_characters.SpeakerCandidate(
                        speaker_id="total",
                        wav_path=str(total),
                        count=1,
                        duration=0.0,
                        segments=[],
                    )
                )
            except Exception as exc:  # noqa: BLE001
                st.warning(f"No pude armar el recorte total del audio: {exc}")
        mapping = director_characters.map_faces_to_speakers(faces, speakers)
        st.session_state["director_cast_faces"] = [
            {
                "cluster_id": face.cluster_id,
                "count": face.count,
                "crop_path": face.crop_path,
                "thumb_path": face.thumb_path,
                "extra_paths": face.extra_paths,
                "name": f"Persona {face.cluster_id}",
                "speaker_id": mapping.get(face.cluster_id, ""),
            }
            for face in faces
        ]
        st.session_state["director_cast_speakers"] = [
            {
                "speaker_id": speaker.speaker_id,
                "wav_path": speaker.wav_path,
                "count": speaker.count,
            }
            for speaker in speakers
        ]
        st.session_state["director_cast_source"] = source_url or str(source)
        if video_path and not keep_video and source_url:
            try:
                Path(video_path).unlink(missing_ok=True)
            except OSError:
                pass
    n_faces = len(st.session_state.get("director_cast_faces") or [])
    n_spk = len(st.session_state.get("director_cast_speakers") or [])
    if n_faces == 0 and n_spk == 0:
        raise comfyui_director.DirectorError(
            "No encontré caras ni locutores en ese material."
        )
    st.success(
        f"Encontré {n_faces} persona(s) y {n_spk} muestra(s) de voz. "
        "Ponéles un nombre y guardalas en el catálogo."
    )


def _render_cast_draft() -> None:
    faces = st.session_state.get("director_cast_faces") or []
    speakers = st.session_state.get("director_cast_speakers") or []
    if not faces and not speakers:
        return
    st.markdown("**🎬 Personas detectadas**")
    speaker_options = [""] + [item["speaker_id"] for item in speakers]
    speaker_labels = {"": "(sin voz)"}
    for item in speakers:
        sid = item["speaker_id"]
        if sid == "recorte":
            speaker_labels[sid] = "Recorte manual (solo el tramo inicio/fin que marcaste)"
        elif sid == "total":
            speaker_labels[sid] = "Recorte total del audio (todo el habla del video, hasta 60 s)"
        else:
            speaker_labels[sid] = f"Locutor {sid} (diarizado)"
    if speakers:
        st.caption(
            "Escuchá las muestras y asigná una a cada persona. "
            "Si el diarizado partió de más a un solo hablante, marcá "
            "«usar todas como una sola voz» o elegí el recorte total."
        )
        merge_all = st.checkbox(
            "🎬 Usar todas como una sola voz",
            key="director_merge_all_voices",
            help=(
                "Une las muestras diarizadas (A, B, C…) en un solo voice_ref. "
                "No incluye el recorte manual ni el recorte total, salvo que no haya otras."
            ),
        )
        cols = st.columns(min(4, len(speakers)))
        for index, speaker in enumerate(speakers):
            with cols[index % len(cols)]:
                st.caption(speaker_labels.get(speaker["speaker_id"], speaker["speaker_id"]))
                if Path(speaker["wav_path"]).is_file():
                    st.audio(speaker["wav_path"])

    source = str(st.session_state.get("director_cast_source") or "")
    if not faces and speakers:
        st.caption("No hubo caras, pero sí audio. Podés guardar un personaje solo con voz.")
        name = st.text_input("🎬 Nombre", key="director_voice_only_name", value="Personaje")
        choice = st.selectbox(
            "🎬 Muestra de voz",
            options=[item["speaker_id"] for item in speakers],
            format_func=lambda key: speaker_labels.get(key, key),
            key="director_voice_only_spk",
        )
        if st.button("🎬 Guardar personaje (solo voz)", key="director_save_voice_only"):
            wav = next((item["wav_path"] for item in speakers if item["speaker_id"] == choice), "")
            # A 1x1 placeholder is not useful; require a later face. Allow voice-only.
            try:
                work = director_characters.characters_dir() / "_inbox"
                # Use a tiny solid image? Better reject without face for PuLID.
                st.warning("Para PuLID hace falta una cara. Subí también una foto o un video con el rostro.")
            except Exception as exc:
                st.error(str(exc))
        return

    for face in faces:
        cluster = face["cluster_id"]
        cols = st.columns([1, 2, 2])
        with cols[0]:
            if Path(face["thumb_path"]).is_file():
                st.image(face["thumb_path"])
            st.caption(f"{face['count']} tomas")
        with cols[1]:
            face["name"] = st.text_input(
                "🎬 Nombre",
                value=face.get("name") or f"Persona {cluster}",
                key=f"director_cast_name_{cluster}",
            )
        with cols[2]:
            default_index = 0
            if face.get("speaker_id") in speaker_options:
                default_index = speaker_options.index(face["speaker_id"])
            chosen = st.selectbox(
                "🎬 Voz",
                options=speaker_options,
                index=default_index,
                format_func=lambda key: speaker_labels.get(key, key or "(sin voz)"),
                key=f"director_cast_voice_{cluster}",
            )
            face["speaker_id"] = chosen
        if st.button(f"🎬 Guardar {face.get('name') or cluster} en el catálogo", key=f"director_save_char_{cluster}"):
            try:
                wav = ""
                if st.session_state.get("director_merge_all_voices"):
                    auto = [
                        item["wav_path"]
                        for item in speakers
                        if item["speaker_id"] not in {"", "recorte", "total"}
                    ]
                    if not auto:
                        auto = [item["wav_path"] for item in speakers if item.get("wav_path")]
                    if auto:
                        merged = director_characters.characters_dir() / "_inbox" / "voz_unida.wav"
                        director_characters.concat_audio_files(auto, merged)
                        wav = str(merged)
                elif face.get("speaker_id"):
                    wav = next(
                        (
                            item["wav_path"]
                            for item in speakers
                            if item["speaker_id"] == face["speaker_id"]
                        ),
                        "",
                    )
                char = director_characters.save_character(
                    name=face.get("name") or f"Persona {cluster}",
                    face_path=face["crop_path"],
                    voice_path=wav or None,
                    source=source,
                    extra_faces=face.get("extra_paths") or [],
                )
                st.success(f"Guardé a {char.name} en bibles/characters/{char.slug}/")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


def _render_character_catalog() -> None:
    characters = director_characters.list_characters()
    st.markdown("**🎬 Catálogo**")
    if not characters:
        st.caption("El catálogo está vacío.")
        return
    cols = st.columns(min(4, len(characters)))
    for index, char in enumerate(characters):
        with cols[index % len(cols)]:
            if char.thumb():
                st.image(char.thumb())
            st.markdown(f"**{char.name}**")
            st.caption(char.slug)
            if char.voice_ref and Path(char.voice_ref).is_file():
                st.audio(char.voice_ref)
            else:
                st.caption("Sin voz")
            new_name = st.text_input(
                "Renombrar",
                value=char.name,
                key=f"director_rename_{char.slug}",
                label_visibility="collapsed",
            )
            b1, b2 = st.columns(2)
            if b1.button("🎬 Renombrar", key=f"director_do_rename_{char.slug}"):
                try:
                    director_characters.rename_character(char.slug, new_name)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
            if b2.button("🎬 Borrar", key=f"director_do_delete_{char.slug}"):
                try:
                    director_characters.delete_character(char.slug)
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
