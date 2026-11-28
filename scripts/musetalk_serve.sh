#!/usr/bin/env sh
set -e
ROOT="${MUSETALK_HOME:-$HOME/Videos/musetalk}"
VENV="$ROOT/.venv/bin/python"
SCRIPT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/musetalk_server.py"
HOST="${MUSETALK_HOST:-127.0.0.1}"
PORT="${MUSETALK_PORT:-4124}"

if [ ! -x "$VENV" ]; then
  echo "MuseTalk venv not found at $VENV"
  exit 1
fi

mkdir -p "$ROOT"
export MUSETALK_HOME="$ROOT"
export MUSETALK_HOST="$HOST"
export MUSETALK_PORT="$PORT"
export MUSETALK_KEEP_ALIVE="${MUSETALK_KEEP_ALIVE:-120}"
export FFMPEG_PATH="${FFMPEG_PATH:-/usr/bin}"
cd "$ROOT"
exec "$VENV" "$SCRIPT"
