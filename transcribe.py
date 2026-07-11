#!/usr/bin/env python3
"""
transcribe.py — turns your recording into a timestamped transcript, fully
local (no cloud calls), using faster-whisper.

    python3 transcribe.py raw/my_recording.mov work/transcript.json

Output JSON: [{"start": 12.3, "end": 15.8, "text": "so basically what happens is..."}, ...]
This feeds plan_segments.py, which decides WHERE motion graphics should go.
"""

import sys
import json
from pathlib import Path

from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()


def transcribe(input_video, output_json, model_size="small", device="auto", compute_type="int8"):
    print(f"[transcribe] loading whisper model '{model_size}' ...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"[transcribe] running on {input_video} (this can take a few minutes) ...")
    segments, info = model.transcribe(str(input_video), vad_filter=True)

    result = []
    for seg in segments:
        result.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        print(f"  [{seg.start:7.2f} - {seg.end:7.2f}] {seg.text.strip()}")

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(result, indent=2))
    print(f"\n[transcribe] wrote {len(result)} segments -> {output_json}")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 transcribe.py <input_video> <output_transcript.json> [model_size]")
        print("model_size options: tiny, base, small, medium, large-v3 (bigger = slower, more accurate)")
        sys.exit(1)
    model_size = sys.argv[3] if len(sys.argv) > 3 else "small"
    transcribe(sys.argv[1], sys.argv[2], model_size=model_size)
