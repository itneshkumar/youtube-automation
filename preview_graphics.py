#!/usr/bin/env python3
"""
preview_graphics.py — inspect the graphics pipeline stage-by-stage without
touching pipeline.py's real work/ output.

Segments come entirely from config.yaml's `segments:` list (itself written by
plan_segments.py from the transcript) — never hardcoded here, so this always
reflects however many graphic segments the transcript-driven planner decided
on, not a fixed count.

For each segment, three stages run and land on disk separately so each can be
inspected on its own:
  1. extract   — graphics_llm.extract_steps() asks the LLM for the ordered
                 step list, saved as steps.json.
  2. build     — the timed HTML/CSS diagram built from those steps, saved as
                 diagram.html (this is the "prompt" the headless browser will
                 play — the animated graphic itself, not text).
  3. render    — diagram.html played and recorded to graphic_video.mp4.

A manifest.json ties all three together per segment for downstream viewing.
"""

import argparse
import json
from pathlib import Path

import yaml

import graphics_llm
from pipeline import hms_to_sec, load_transcript_entries_for_range


def preview_segment(idx, seg, cfg, out_root):
    render_cfg = cfg["render"]
    graphics_cfg = cfg.get("graphics", {})
    pip_position = cfg.get("pip", {}).get("position", "bottom-right")
    width, height, fps = render_cfg["width"], render_cfg["height"], render_cfg["fps"]

    start, end = hms_to_sec(seg["start"]), hms_to_sec(seg["end"])
    duration = end - start
    work_dir = out_root / f"segment_{idx:02d}"
    work_dir.mkdir(parents=True, exist_ok=True)

    transcript_entries = load_transcript_entries_for_range(cfg, start, end)
    transcript_text = " ".join(e["text"].strip() for e in transcript_entries)

    print(f"[segment {idx:02d}] {seg['start']} -> {seg['end']} ({duration:.1f}s): extracting steps...")
    steps = graphics_llm.extract_steps(
        seg["prompt"], transcript_text,
        backend=graphics_cfg.get("backend", graphics_llm.DEFAULT_BACKEND),
        model=graphics_cfg.get("model"),
        llamacpp_server=graphics_llm.DEFAULT_LLAMACPP_SERVER,
        timeout=graphics_cfg.get("timeout_sec", 180),
    )
    steps_path = work_dir / "steps.json"
    steps_path.write_text(json.dumps(steps, indent=2))
    print(f"[segment {idx:02d}] saved {len(steps)} steps -> {steps_path}")

    delays = graphics_llm._assign_step_delays(steps, transcript_entries, start, duration)
    html_doc = graphics_llm._build_diagram_html(steps, delays, duration, width, height, pip_position)
    html_path = work_dir / "diagram.html"
    html_path.write_text(html_doc)
    print(f"[segment {idx:02d}] built diagram -> {html_path}")

    print(f"[segment {idx:02d}] rendering video ({width}x{height}@{fps})...")
    mp4_path = graphics_llm.render_html_to_video(html_path, work_dir, duration, width, height, fps=fps)
    print(f"[segment {idx:02d}] rendered -> {mp4_path}")

    return {
        "index": idx,
        "start": seg["start"],
        "end": seg["end"],
        "duration_sec": duration,
        "prompt": seg["prompt"],
        "steps": steps,
        "steps_path": str(steps_path),
        "html_path": str(html_path),
        "mp4_path": str(mp4_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="config.yaml")
    parser.add_argument("--segment", type=int, default=None,
                         help="only preview this segment index (0-based, in transcript order); default: all")
    parser.add_argument("--out-dir", default="work/graphics_preview")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    segments = sorted(cfg.get("segments", []), key=lambda s: hms_to_sec(s["start"]))
    if not segments:
        raise SystemExit("config.yaml has no segments: — run plan_segments.py first.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    targets = list(enumerate(segments))
    if args.segment is not None:
        targets = [(i, s) for i, s in targets if i == args.segment]
        if not targets:
            raise SystemExit(f"No segment at index {args.segment} (found {len(segments)} segments).")

    results = [preview_segment(i, s, cfg, out_root) for i, s in targets]

    manifest_path = out_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for r in results:
        manifest[str(r["index"])] = r
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {manifest_path}")


if __name__ == "__main__":
    main()
