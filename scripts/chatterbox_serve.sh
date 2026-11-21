#!/usr/bin/env sh
set -e
ROOT="${CHATTERBOX_HOME:-$HOME/Videos/chatterbox}"
VENV="$ROOT/.venv/bin/python"
SCRIPT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/chatterbox_server.py"
HOST="${CHATTERBOX_HOST:-127.0.0.1}"
PORT="${CHATTERBOX_PORT:-4123}"

if [ ! -x "$VENV" ]; then
  echo "Chatterbox venv not found at $VENV"
  echo "Create it with: uv venv --python 3.11 $ROOT/.venv && uv pip install --python $VENV torch torchaudio --index-url https://download.pytorch.org/whl/cu128 && uv pip install --python $VENV chatterbox-tts soundfile"
  exit 1
fi

mkdir -p "$ROOT"
export CHATTERBOX_HOST="$HOST"
export CHATTERBOX_PORT="$PORT"
export CHATTERBOX_KEEP_ALIVE="${CHATTERBOX_KEEP_ALIVE:-120}"
export CHATTERBOX_LANGUAGE="${CHATTERBOX_LANGUAGE:-es}"
export CHATTERBOX_VOICE_REF_FILE="${CHATTERBOX_VOICE_REF_FILE:-$ROOT/voice_ref.txt}"
# Hugging Face / torch caches stay in the user account.
cd "$ROOT"
exec "$VENV" "$SCRIPT"
