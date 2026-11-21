# Fork director (rama `director`)

Esta rama agrega una fuente de video **🎬 ComfyUI MiniMax (fork director)** a MoneyPrinterTurbo. No es un generador de clips de stock: escribe un guion a partir de tu estrategia, convierte cada escena en un prompt visual y le pide el clip a ComfyUI con MiniMax H3, respetando una **biblia** de estilo y continuidad.

## Cómo usarlo

1. Levantá ComfyUI (en esta máquina suele estar en `http://127.0.0.1:8188`) y el WebUI: `cd ~/Videos/MoneyPrinterTurbo && sh webui.sh`.
2. En **Fuente de video** elegí **🎬 ComfyUI MiniMax (fork director)**. La sección nueva está marcada con el ícono 🎬 y un borde violeta, y dice *Nuevo (fork director)*.
3. Completá o generá una biblia (`bibles/example.toml` es la plantilla). El botón **🎬 Generar biblia con IA** toma la idea/estrategia y guarda un archivo editable en `bibles/`.
4. Dejá encendido **🎬 Encadenar escenas** para que el último fotograma de un clip sea el primer fotograma del siguiente (modo first-frame de MiniMax H3).
5. Generá el video como siempre. MoneyPrinterTurbo pone voz, subtítulos y música; ComfyUI solo fabrica los clips.

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
