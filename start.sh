#!/usr/bin/env bash
# start.sh — one command to run the whole auto-edit pipeline end to end:
#   transcribe -> plan graphic segments -> silence/noise/loudness +
#   Claude-authored animated diagrams + oval webcam bubble -> output/final_cut.mp4
#
# Usage:
#   ./start.sh raw/my_recording.mov
#   ./start.sh raw/my_recording.mov --llm              # smarter segment planning via HF router
#   ./start.sh raw/my_recording.mov --replan            # re-run planning even if config.yaml already has segments
#   ./start.sh raw/my_recording.mov --force-transcribe  # re-transcribe even if work/transcript.json exists
#   ./start.sh raw/my_recording.mov --whisper-model medium
#
# Re-running with the same input video reuses work/transcript.json and any
# segments already written into config.yaml, so it's safe to re-run after
# tweaking config.yaml by hand.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INPUT_VIDEO=""
USE_LLM=false
REPLAN=false
FORCE_TRANSCRIBE=false
WHISPER_MODEL="small"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm) USE_LLM=true; shift ;;
    --replan) REPLAN=true; shift ;;
    --force-transcribe) FORCE_TRANSCRIBE=true; shift ;;
    --whisper-model) WHISPER_MODEL="$2"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      if [[ -z "$INPUT_VIDEO" ]]; then
        INPUT_VIDEO="$1"
      else
        echo "Unrecognized argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$INPUT_VIDEO" ]]; then
  echo "Usage: ./start.sh <input_video> [--llm] [--replan] [--force-transcribe] [--whisper-model small]" >&2
  exit 1
fi

if [[ ! -f "$INPUT_VIDEO" ]]; then
  echo "Input video not found: $INPUT_VIDEO" >&2
  exit 1
fi

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$1"; }

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is not on PATH. Install it (e.g. 'brew install ffmpeg') and re-run." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  log "Creating virtualenv (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

log "Installing/checking Python dependencies"
pip install -q -r requirements.txt
python3 -m playwright install --with-deps chromium >/dev/null 2>&1 || python3 -m playwright install chromium

WORK_DIR="work"
TRANSCRIPT="$WORK_DIR/transcript.json"
CONFIG="config.yaml"

mkdir -p "$WORK_DIR"

if [[ ! -f "$CONFIG" ]]; then
  log "No config.yaml found — creating one from config.example.yaml"
  cp config.example.yaml "$CONFIG"
fi

log "Pointing config.yaml at $INPUT_VIDEO"
python3 - "$CONFIG" "$INPUT_VIDEO" <<'PYEOF'
import sys
import yaml
from pathlib import Path

config_path, input_video = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(Path(config_path).read_text()) or {}
cfg["input_video"] = input_video
Path(config_path).write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
PYEOF

if [[ "$FORCE_TRANSCRIBE" == true || ! -f "$TRANSCRIPT" ]]; then
  log "Transcribing $INPUT_VIDEO (model: $WHISPER_MODEL) — this can take a few minutes"
  python3 transcribe.py "$INPUT_VIDEO" "$TRANSCRIPT" "$WHISPER_MODEL"
else
  log "Reusing existing transcript at $TRANSCRIPT (use --force-transcribe to redo)"
fi

HAS_SEGMENTS=$(python3 - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print("yes" if cfg.get("segments") else "no")
PYEOF
)

if [[ "$REPLAN" == true || "$HAS_SEGMENTS" == "no" ]]; then
  log "Planning graphic segments from transcript (llm=$USE_LLM)"
  PLAN_ARGS=("$TRANSCRIPT" "$CONFIG" --write)
  if [[ "$USE_LLM" == true ]]; then
    PLAN_ARGS+=(--llm)
  fi
  python3 plan_segments.py "${PLAN_ARGS[@]}"
else
  log "config.yaml already has segments — skipping planning (use --replan to redo)"
fi

SEGMENT_COUNT=$(python3 - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(len(cfg.get("segments") or []))
PYEOF
)

if [[ "$SEGMENT_COUNT" -gt 0 ]]; then
  GRAPHICS_BACKEND=$(python3 - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(cfg.get("graphics", {}).get("backend", "ollama"))
PYEOF
)

  if [[ "$GRAPHICS_BACKEND" == "ollama" ]]; then
    OLLAMA_SERVER=$(python3 - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
print(cfg.get("graphics", {}).get("ollama_server", "http://127.0.0.1:11434"))
PYEOF
)
    if ! curl -s -o /dev/null -m 5 "$OLLAMA_SERVER/api/tags"; then
      echo "" >&2
      echo "config.yaml has $SEGMENT_COUNT graphic segment(s) using the ollama backend, but $OLLAMA_SERVER isn't reachable." >&2
      echo "Start it with 'ollama serve' (or open the Ollama app) and make sure the configured model is pulled." >&2
      exit 1
    fi
  elif [[ "$GRAPHICS_BACKEND" == "anthropic" ]]; then
    # Load .env so ANTHROPIC_API_KEY set there is visible to this check.
    if [[ -f .env ]]; then
      set -a
      # shellcheck disable=SC1091
      source .env
      set +a
    fi
    if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
      echo "" >&2
      echo "config.yaml has $SEGMENT_COUNT graphic segment(s) using the anthropic backend, which needs ANTHROPIC_API_KEY." >&2
      echo "Set it in .env or export it before running." >&2
      exit 1
    fi
  fi
fi

log "Running the render pipeline"
python3 pipeline.py "$CONFIG"

log "Done. Final video: $(python3 - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(Path(cfg["output_video"]).resolve())
PYEOF
)"
