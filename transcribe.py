#!/usr/bin/env python3
"""
transcribe.py — turns your recording into a timestamped transcript, fully
local (no cloud calls).

    python3 transcribe.py raw/my_recording.mov work/transcript.json

Output JSON: [{"start": 12.3, "end": 15.8, "text": "so basically what happens is..."}, ...]
This feeds plan_segments.py, which decides WHERE motion graphics should go.

Two engines:
  whispercpp    (default) — shells out to whisper.cpp's `whisper-cli`. On
                             Apple Silicon this runs Metal-accelerated
                             (offloaded to the GPU) and benchmarked ~5x
                             faster here than faster-whisper's CPU-only int8
                             path, since CTranslate2 has no Metal backend.
  faster-whisper             — the original pure-Python/CTranslate2 path.
                             Used automatically if `whisper-cli` isn't on
                             PATH (e.g. not installed, or a non-Mac box),
                             or can be selected explicitly.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

WHISPER_CPP_BIN = os.getenv("WHISPER_CPP_BIN", "whisper-cli")
WHISPER_CPP_MODEL_DIR = Path.home() / ".cache" / "whisper-cpp" / "models"
_SPECIAL_TOKEN_RE = re.compile(r"^\[_.*\]$|^<\|.*\|>$")


def find_cached_whispercpp_model(model_size):
    """
    Return the path to an already-downloaded ggml model for model_size, or
    None if it isn't cached anywhere yet. Never downloads — this is the
    read-only check the UI uses to decide whether to show a Download button.
    Checks the hyperframes skill's whisper cache too, since it's the same
    ggml format and avoids a redundant multi-GB fetch if one's already local.
    """
    explicit = os.getenv("WHISPER_CPP_MODEL")
    if explicit and Path(explicit).exists():
        return Path(explicit)

    hyperframes_cache = Path.home() / ".cache" / "hyperframes" / "whisper" / "models"
    candidates = [
        WHISPER_CPP_MODEL_DIR / f"ggml-{model_size}.bin",
        WHISPER_CPP_MODEL_DIR / f"ggml-{model_size}.en.bin",
        hyperframes_cache / f"ggml-{model_size}.bin",
        hyperframes_cache / f"ggml-{model_size}.en.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def download_whispercpp_model(model_size, progress_cb=None):
    """
    Download the ggml model for model_size to WHISPER_CPP_MODEL_DIR.
    Streams via `requests` (rather than shelling out to curl) so callers —
    e.g. ui_server.py's download button — can get live (bytes_downloaded,
    total_bytes) progress instead of a blocking, un-observable call.
    Downloads to a .part file first so a crash/interrupt never leaves a
    truncated file that looks cached.
    """
    WHISPER_CPP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dest = WHISPER_CPP_MODEL_DIR / f"ggml-{model_size}.bin"
    tmp = dest.with_suffix(".bin.part")
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_size}.bin"

    with requests.get(url, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb:
                    progress_cb(downloaded, total)
    tmp.rename(dest)
    return dest


def _resolve_whispercpp_model(model_size):
    """Cached lookup, falling back to a blocking download for CLI/pipeline
    use (the UI instead calls find_cached_whispercpp_model +
    download_whispercpp_model directly so it can show progress)."""
    cached = find_cached_whispercpp_model(model_size)
    if cached:
        return cached
    print(f"[transcribe] downloading whisper.cpp model '{model_size}' ...")
    return download_whispercpp_model(model_size)


def _merge_tokens_to_words(tokens):
    """
    whisper.cpp's --output-json-full gives per-token (BPE subword) timing,
    not per-word — "MCP" comes back as tokens " M" + "CP". Whisper's
    tokenizer marks a new word with a leading space, so merge on that to
    get the same {"start","end","word"} shape transcribe_faster_whisper
    already produces (graphics_llm.py's step-timing match needs whole
    words, not subword fragments).
    """
    words = []
    for tok in tokens:
        text = tok["text"]
        if _SPECIAL_TOKEN_RE.match(text.strip()):
            continue
        start = round(tok["offsets"]["from"] / 1000, 2)
        end = round(tok["offsets"]["to"] / 1000, 2)
        if not text.startswith(" ") and words:
            words[-1]["word"] += text
            words[-1]["end"] = end
        else:
            words.append({"start": start, "end": end, "word": text.strip()})
    return words


def transcribe_whispercpp(input_video, output_json, model_size="small"):
    print(f"[transcribe] using whisper.cpp (Metal-accelerated) with model '{model_size}' ...")
    model_path = _resolve_whispercpp_model(model_size)

    out_base = Path(output_json).with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    audio_wav = out_base.parent / f"{out_base.name}_audio16k.wav"

    # whisper-cli only reads flac/mp3/ogg/wav — extract 16kHz mono audio first.
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_video), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(audio_wav)],
        check=True, capture_output=True,
    )
    print(f"[transcribe] running on {input_video} ...")
    try:
        subprocess.run(
            [WHISPER_CPP_BIN, "-m", str(model_path), "-f", str(audio_wav),
             "-oj", "-ojf", "-of", str(out_base), "-np"],
            check=True, capture_output=True,
        )
        raw = json.loads(out_base.with_suffix(".json").read_text())
    finally:
        audio_wav.unlink(missing_ok=True)
        out_base.with_suffix(".json").unlink(missing_ok=True)

    result = []
    for seg in raw.get("transcription", []):
        text = seg["text"].strip()
        if not text:
            continue
        entry = {
            "start": round(seg["offsets"]["from"] / 1000, 2),
            "end": round(seg["offsets"]["to"] / 1000, 2),
            "text": text,
            "words": _merge_tokens_to_words(seg.get("tokens", [])),
        }
        result.append(entry)
        print(f"  [{entry['start']:7.2f} - {entry['end']:7.2f}] {text}")

    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\n[transcribe] wrote {len(result)} segments -> {output_json}")
    return result


def transcribe_faster_whisper(input_video, output_json, model_size="small", device="auto", compute_type="int8"):
    from faster_whisper import WhisperModel

    print(f"[transcribe] loading whisper model '{model_size}' ...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[transcribe] running on {input_video} (this can take a few minutes) ...")
    # word_timestamps=True: without it, a whole multi-second sentence is one
    # entry with a single start time, which is too coarse for graphics_llm.py
    # to time individual step reveals against (it needs to know exactly when
    # "perceive" vs "reason" vs "act" was said, not just when the sentence
    # containing all of them began). Word-level timestamps let it do that;
    # plan_segments.py's phrase heuristics still use the segment-level
    # "text" field, so this is additive, not a format change.
    segments, info = model.transcribe(str(input_video), vad_filter=True, word_timestamps=True)

    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [
                {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word.strip()}
                for w in (seg.words or [])
            ],
        })
        print(f"  [{seg.start:7.2f} - {seg.end:7.2f}] {seg.text.strip()}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\n[transcribe] wrote {len(result)} segments -> {output_json}")
    return result


def transcribe(input_video, output_json, model_size="small", device="auto", compute_type="int8", engine="whispercpp"):
    if engine == "whispercpp":
        import shutil
        if shutil.which(WHISPER_CPP_BIN):
            try:
                return transcribe_whispercpp(input_video, output_json, model_size=model_size)
            except subprocess.CalledProcessError as exc:
                print(f"[transcribe] whisper.cpp failed ({exc}); falling back to faster-whisper.")
        else:
            print(f"[transcribe] '{WHISPER_CPP_BIN}' not found on PATH; falling back to faster-whisper.")
    return transcribe_faster_whisper(input_video, output_json, model_size=model_size, device=device, compute_type=compute_type)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 transcribe.py <input_video> <output_transcript.json> [model_size] [engine]")
        print("model_size options: tiny, base, small, medium, large-v3 (bigger = slower, more accurate)")
        print("engine options: whispercpp (default, Metal-accelerated), faster-whisper")
        sys.exit(1)
    model_size = sys.argv[3] if len(sys.argv) > 3 else "small"
    engine = sys.argv[4] if len(sys.argv) > 4 else "whispercpp"
    transcribe(sys.argv[1], sys.argv[2], model_size=model_size, engine=engine)
