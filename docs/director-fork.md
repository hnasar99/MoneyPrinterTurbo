# Fork director (rama `director`)

Esta rama agrega una fuente de video **🎬 ComfyUI MiniMax (fork director)** a MoneyPrinterTurbo. No es un generador de clips de stock: escribe un guion a partir de tu estrategia, convierte cada escena en un prompt visual y le pide el clip a ComfyUI con MiniMax H3, respetando una **biblia** de estilo y continuidad.

## Cómo usarlo

1. Levantá ComfyUI (en esta máquina suele estar en `http://127.0.0.1:8188`) y el WebUI: `cd ~/Videos/MoneyPrinterTurbo && sh webui.sh`.
2. Arriba hay pestañas grandes: **🎬 Generar video**, **🎬 Personajes**, **🎬 Escenas** y **🎬 Mejorar** (estas dos últimas todavía son un placeholder). Están marcadas con 🎬, borde violeta y *Nuevo (fork director)*.
3. En **Fuente de video** elegí **🎬 ComfyUI MiniMax (fork director)**.
4. Completá o generá una biblia (`bibles/example.toml` es la plantilla). El botón **🎬 Generar biblia con IA** toma la idea/estrategia y guarda un archivo editable en `bibles/`.
5. Dejá encendido **🎬 Encadenar escenas** para que el último fotograma de un clip sea el primer fotograma del siguiente (modo first-frame de MiniMax H3).
6. Generá el video como siempre. MoneyPrinterTurbo pone voz, subtítulos y música; ComfyUI solo fabrica los clips.

## Biblia

Un archivo TOML por proyecto, en `bibles/`. Fija personaje, frase de estilo, reglas de cámara, negative prompt, semilla, resolución, duración (3–5 s) y pasos. Lo único que debería cambiar entre escenas es la acción.

## Modelos de texto (local primero)

En esta máquina **Ollama ya está instalado** (0.32.5) con, entre otros:

- `qwen3-vl:8b` (8.8B) — recomendado para biblia y guion.
- `Qwen2.5:3b` (3.1B) — recomendado para pasar cada escena a prompt visual (más rápido).
- También están `deepseek-r1:14b` (justo para 12 GB) y un distill de 1.2B; no son el default.

No se bajó ningún modelo nuevo. Si en otra máquina no hubiera Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b
# o, si preferís visión + texto:
# ollama pull qwen3-vl:8b
```

Elegí modelos de 7–8B (Q4) para una RTX 3060 12 GB. No cargues el LLM y MiniMax H3 al mismo tiempo.

### VRAM

El fork libera memoria solo:

- Antes de una llamada al LLM: `POST /free` en ComfyUI (`unload_models` + `free_memory`).
- Antes de mandar un clip a ComfyUI: descarga los modelos de Ollama con `keep_alive=0`.

### Respaldo (si Ollama falla)

1. **NVIDIA free** (build.nvidia.com), OpenAI-compatible:
   - `base_url`: `https://integrate.api.nvidia.com/v1`
   - Clave: `nvidia_api_key` en la sección `[app]` de `config.toml` (vacía; la cargás vos).
   - Modelos verificados el 2026-11-21 en `/v1/models`: biblia/guion `google/gemma-4-31b-it`; prompts `google/gemma-3-4b-it`.
2. **OpenRouter free**, modelos verificados el 2026-11-21: biblia/guion `google/gemma-4-31b-it:free`; prompts `nvidia/nemotron-3.5-lightning:free`. Usa `openrouter_api_key`.
3. El proveedor que ya tengas configurado en MoneyPrinterTurbo.

En la sección violeta podés cambiar proveedor y modelo por tarea.

## MiniMax como LLM de texto (guion y biblia)

No hace falta un proveedor nuevo. Usá el que ya es compatible con OpenAI:

- **OpenRouter:** `llm_provider = "openrouter"` y un id actual, por ejemplo `minimax/minimax-m3.1-flash`, `minimax/minimax-m3` o `minimax/minimax-m2.5`. El id viejo `minimax/minimax-m2` fue dado de baja el 5 de noviembre de 2026.
- **API oficial de MiniMax:** `llm_provider = "openai"` (u otro OpenAI-compatible), `openai_base_url = "https://api.minimax.io/v1"`, modelo `MiniMax-M3.1-Flash` o `MiniMax-M3`. Documentación: https://platform.minimax.io/docs/api-reference/text-openai-api

No se agregó ninguna clave en este fork.

## Configuración

Sección `[comfyui]` en `config.toml` (url, ruta de workflow API opcional, timeout, biblia, encadenado y modelos por tarea). Ver `config.example.toml`.

## Límites

- MiniMax H3 es pesado para 16 GB de RAM y 12 GB de VRAM. El default de la plantilla es 640×352, 3 s y 8 pasos con LoRA turbo. Si aparece OOM, bajá resolución o duración; no conviene tener ComfyUI y Ollama cargados juntos.
- Cada clip se genera por separado. El encadenado por último fotograma ayuda, pero no reemplaza un LoRA de personaje.
- MoneyPrinterTurbo sigue armando el video final con esos clips, más voz y subtítulos.

## Catálogo de personajes

En la pestaña **🎬 Personajes**:

1. Leé la nota de consentimiento. Subí un video, una foto o pegá una URL (inicio y fin opcionales).
2. Pulsá **🎬 Detectar personas**. InsightFace agrupa una miniatura por persona. Si hay audio, un diarizado liviano (CAM++ ONNX, ~29 MB, sin token de Hugging Face) arma una muestra por locutor y trata de asignarla a la cara según quién está en cuadro. Si no alcanza, queda el recorte manual inicio/fin y podés asignar la voz a mano, con previa.
3. Ponéle un nombre a cada persona y guardala. Cada una vive en `bibles/characters/<slug>/` con un `character.toml` (nombre, `face_ref`, `voice_ref`, fecha, origen). El catálogo se ve como una grilla de íconos; se puede renombrar o borrar.
4. En **🎬 Generar video**, con la fuente ComfyUI, elegí **🎬 Protagonistas**. Con uno, se usan su cara (PuLID) y su voz (Chatterbox). Con dos o más, el guion va en réplicas `Nombre:` y se alternan planos y voces. No hay dos caras en el mismo cuadro ni lip-sync.

Chatterbox Multilingual (español) sigue en un venv aparte (`~/Videos/chatterbox/.venv`) y un servidor local en `http://127.0.0.1:4123`. Si elegís protagonistas, el WebUI selecciona **🎬 Chatterbox (voz clonada)**. También podés seguir usando `voice_ref` / `face_ref` en la biblia si no hay protagonistas.

Solo usá voces y caras de personas que te hayan dado consentimiento.

Arranque manual del servidor: `sh ~/Videos/MoneyPrinterTurbo/scripts/chatterbox_serve.sh`. MoneyPrinterTurbo lo levanta solo si no está en marcha. Tras un rato inactivo libera la GPU (`/free`, keep_alive 120 s) para no pelear con MiniMax H3.

Los subtítulos de esta voz usan el recorte del texto sobre la duración del audio (como ElevenLabs). Para un ajuste más fino, `subtitle_provider = "whisper"`.

## Clonar cara (PuLID)

PuLID no puede condicionar MiniMax H3 directamente. El camino práctico en 12 GB de VRAM es:

1. El mismo flujo de **🎬 Extraer voz y caras** muestrea fotogramas (o usa la foto), detecta caras con InsightFace (AntelopeV2) y agrupa por persona.
2. Si hay más de una persona, elegí la miniatura. El recorte queda en `bibles/faces/` y en la biblia (`[character] face_ref`).
3. Al generar, si `face_ref` está definido, ComfyUI fabrica el primer fotograma con PuLID sobre **SDXL Lightning de 4 pasos** (`sdxl_lightning_4step.safetensors`) a resolución modesta (lado largo 768, múltiplo de 64). InsightFace corre en CPU para ahorrar VRAM.
4. Ese still entra como first-frame del clip MiniMax H3. En escenas siguientes se reusa el último fotograma de la cadena. Si la cadena se corta (o está apagada), se vuelve a generar un still con PuLID.
5. Entre PuLID y MiniMax el fork hace `POST /free` para no dejar los pesos de SDXL ocupando la GPU.

Nodos: `~/Videos/ComfyUI/custom_nodes/PuLID_ComfyUI` (cubiq). Modelos: `models/pulid/ip-adapter_pulid_sdxl_fp16.safetensors`, `models/checkpoints/sdxl_lightning_4step.safetensors`, `models/insightface/models/antelopev2/`. EVA-CLIP se descarga solo la primera vez.

No hace falta una sola persona en todo el video: el catálogo muestra una cara por persona. El diarizado intenta emparejar locutor y cara; si no, asignás la muestra a mano o recortás inicio/fin a un solo hablante.

## Lip-sync (MuseTalk 1.5)

Después de que MiniMax renderiza cada clip, y antes de que MoneyPrinterTurbo los pegue, el fork puede sincronizar la boca con el audio de Chatterbox de esa escena.

Se eligió **MuseTalk 1.5** (no LatentSync 1.6) porque entra cómodo en una RTX 3060 12 GB (~7.5 GB de VRAM en FP16). LatentSync 1.6 pide unos 18 GB a 512 px. MuseTalk recicla el clip en ida y vuelta si el audio es más largo, y el fork además alarga el video con un ping-pong para que la boca cubra toda la línea.

Corre en un venv aparte (`~/Videos/musetalk/.venv`) y un servidor local en `http://127.0.0.1:4124`, igual que Chatterbox. El fork libera ComfyUI (`/free`), Chatterbox y Ollama antes de usarlo, y lo descarga al terminar.

En **🎬 Generar video**, con un protagonista elegido, el interruptor **🎬 Lip-sync** queda encendido por defecto (se puede apagar). Arranque manual: `sh ~/Videos/MoneyPrinterTurbo/scripts/musetalk_serve.sh`.

El venv usa Python 3.10, torch 2.0.1+cu118, mmcv 2.0.1 y mmpose 1.1.0, con `--version v15 --use_float16 --batch_size 4`. Dejá `setuptools==70.3.0`: desde setuptools 82 ya no está `pkg_resources` y mmpose no arranca. Los pesos viven en `~/Videos/musetalk/models` (no se commitean).

## Errores al generar

Si falta un dato o algo no encaja (Ollama caído, ComfyUI en :8188 apagado, Chatterbox sin muestra, lip-sync sin protagonista, Pexels sin API key, guion vacío, etc.), **Generar video** no arranca: aparece un resumen arriba y el campo queda con borde rojo y una explicación en castellano. Si el fallo ocurre a mitad de la generación, se marca el mismo campo en vez de mostrar un traceback.

## Voces cuando el diarizado parte de más

Si hay una sola cara, las muestras de locutor se unen en una. Si igual aparecen varias, podés marcar **🎬 Usar todas como una sola voz** (concatena las muestras A, B, C…) o elegir:

- **Recorte manual (solo el tramo inicio/fin que marcaste)**
- **Recorte total del audio (todo el habla del video, hasta 60 s)**
