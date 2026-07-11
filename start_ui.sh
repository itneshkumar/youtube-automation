#!/usr/bin/env bash
# start_ui.sh — the one command to run: launches the local web UI, which is
# where you configure everything and trigger start.sh from a browser
# instead of the command line.
#
#   ./start_ui.sh
#
# Opens http://127.0.0.1:8787 automatically. Loopback-only. Leave this
# running in a terminal tab; close it (Ctrl+C) to stop the UI server (any
# pipeline run it kicked off keeps running independently in the background).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not on PATH. Install it (e.g. 'brew install ffmpeg') and re-run." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtualenv (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q -r requirements.txt

PORT=8787
if command -v lsof >/dev/null 2>&1; then
  EXISTING_PIDS="$(lsof -ti tcp:"$PORT" || true)"
  if [[ -n "$EXISTING_PIDS" ]]; then
    echo "Port $PORT is already in use (PID(s): $EXISTING_PIDS) — stopping the previous UI server..."
    kill $EXISTING_PIDS 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      lsof -ti tcp:"$PORT" >/dev/null 2>&1 || break
      sleep 0.3
    done
  fi
fi

python3 ui_server.py
