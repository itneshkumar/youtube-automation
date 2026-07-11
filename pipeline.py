#!/usr/bin/env python3
"""
pipeline.py — the one command to run.

    python3 pipeline.py config.yaml

Approach: split the ORIGINAL recording into alternating ranges —
"talk" ranges (normal talking-head) and "graphic" ranges (segments from
config.yaml where a motion graphic + oval webcam bubble should appear).
Each range is processed independently, then everything is concatenated
in order. This avoids timestamp drift that would happen if silence-removal
shifted the timeline before graphic segments were placed.

  Per TALK range:  silence removal -> noise reduction -> loudness/EQ
  Per GRAPHIC range: Claude-authored animated HTML diagram -> oval webcam bubble -> composite

Everything here runs outside DaVinci Resolve. Import final_cut.mp4 into
Resolve afterwards for color grading / final review — see resolve_import.py
for an optional script that does that hand-off automatically.
"""

import sys
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

import audio_tools
import background_replace
import graphics_llm
import pip_overlay


def hms_to_sec(hms):
    """'00:01:25.0' -> seconds (float). Accepts already-numeric input too."""
    if isinstance(hms, (int, float)):
        return float(hms)
    parts = [float(p) for p in hms.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, s = parts
    return h * 3600 + m * 60 + s


def build_range_plan(duration, segments):
    """
    Returns an ordered list of dicts: {"kind": "talk"|"graphic", "start": s,
    "end": e, "segment": seg_or_None} covering the full duration.
    """
    graphic_ranges = sorted(
        ({"start": hms_to_sec(s["start"]), "end": hms_to_sec(s["end"]), "segment": s}
         for s in segments),
        key=lambda r: r["start"]
    )

    plan = []
    cursor = 0.0
    for gr in graphic_ranges:
        if gr["start"] > cursor:
            plan.append({"kind": "talk", "start": cursor, "end": gr["start"], "segment": None})
        plan.append({"kind": "graphic", "start": gr["start"], "end": gr["end"], "segment": gr["segment"]})
        cursor = gr["end"]
    if cursor < duration:
        plan.append({"kind": "talk", "start": cursor, "end": duration, "segment": None})
    return plan


def process_talk_range(raw_input, start, end, work_dir, cfg, noise_profile_wav=None, background_image=None):
    """Extract [start,end] from raw_input and run the full audio pipeline on it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_slice = work_dir / "raw_slice.mkv"
    audio_tools.run([
        "ffmpeg", "-y", "-i", str(raw_input),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "pcm_s16le",
        str(raw_slice)
    ])

    silence_cfg = cfg["silence"]
    cut_path = work_dir / "cut.mkv"
    audio_tools.cut_silence(
        raw_slice, cut_path, work_dir / "silence_parts",
        noise_floor_db=silence_cfg["noise_floor_db"],
        min_silence_sec=silence_cfg["min_silence_sec"],
        padding=silence_cfg["keep_padding_sec"],
    )

    if background_image is not None:
        bg_path = work_dir / "bg_replaced.mp4"
        bg_cfg = cfg.get("background", {})
        accent_hex = bg_cfg.get("accent", "#2dd4bf").lstrip("#")
        accent_rgb = tuple(int(accent_hex[i:i + 2], 16) for i in (0, 2, 4))
        background_replace.replace_background(
            cut_path, bg_path, background_image, work_dir,
            feather_px=bg_cfg.get("feather_px", 9),
            accent_rgb=accent_rgb,
            rim_glow_px=bg_cfg.get("rim_glow_px", 4),
        )
        cut_path = bg_path

    nr_cfg = cfg["noise_reduction"]
    denoised_path = work_dir / "denoised.mkv"
    if nr_cfg.get("enabled", True):
        audio_tools.reduce_noise(
            cut_path, denoised_path, work_dir,
            noise_profile_wav=noise_profile_wav,
            profile_start_sec=nr_cfg["profile_start_sec"],
            profile_end_sec=nr_cfg["profile_end_sec"],
        )
    else:
        shutil.copy(cut_path, denoised_path)

    ae_cfg = cfg["audio_enhance"]
    enhanced_path = work_dir / "enhanced.mkv"
    audio_tools.enhance_audio(
        denoised_path, enhanced_path,
        highpass_hz=ae_cfg["highpass_hz"],
        presence_boost_db=ae_cfg["presence_boost_db"],
        target_lufs=ae_cfg["target_lufs"],
        limiter_ceiling_db=ae_cfg["limiter_ceiling_db"],
    )
    return enhanced_path


def load_transcript_entries_for_range(cfg, start, end):
    """transcript.json entries overlapping [start, end], in spoken order."""
    transcript_path = Path(cfg["work_dir"]) / "transcript.json"
    if not transcript_path.exists():
        return []
    entries = json.loads(transcript_path.read_text())
    overlapping = [e for e in entries if e["end"] > start and e["start"] < end]
    overlapping.sort(key=lambda e: e["start"])
    return overlapping


def process_graphic_range(raw_input, rng, work_dir, cfg, mask_png, border_png, noise_profile_wav=None):
    """Generate the motion graphic + oval PIP composite for one graphic range."""
    work_dir.mkdir(parents=True, exist_ok=True)
    seg = rng["segment"]
    render_cfg = cfg["render"]
    pip_cfg = cfg["pip"]
    frame_w, frame_h = render_cfg["width"], render_cfg["height"]
    oval_w = int(frame_w * pip_cfg["width_pct"]) // 2 * 2  # libx264 needs even dimensions
    oval_h = oval_w

    graphics_cfg = cfg.get("graphics", {})
    duration = rng["end"] - rng["start"]
    transcript_entries = load_transcript_entries_for_range(cfg, rng["start"], rng["end"])
    graphic_video = graphics_llm.generate_motion_graphic(
        seg["prompt"], work_dir / "graphic_raw", duration, frame_w, frame_h,
        transcript_entries=transcript_entries, range_start=rng["start"],
        pip_position=pip_cfg.get("position", "bottom-right"),
        backend=graphics_cfg.get("backend", graphics_llm.DEFAULT_BACKEND),
        model=graphics_cfg.get("model"),
        ollama_server=graphics_cfg.get("ollama_server", graphics_llm.DEFAULT_OLLAMA_SERVER),
        timeout=graphics_cfg.get("timeout_sec", 180),
    )

    pip_clip = pip_overlay.build_pip_clip(
        raw_input, rng["start"], rng["end"], oval_w, oval_h,
        mask_png, work_dir, border_png=border_png,
    )

    composited = work_dir / "composited.mp4"
    pip_overlay.composite_over_graphic(
        graphic_video, pip_clip, composited,
        frame_w, frame_h,
        position=pip_cfg["position"], oval_w=oval_w,
        margin_px=pip_cfg["margin_px"],
    )

    # This range currently carries the RAW (un-enhanced) mic audio under the
    # graphic. Run it through the same noise-reduction + enhancement chain
    # so loudness/tone stays consistent with the talk ranges.
    nr_cfg = cfg["noise_reduction"]
    ae_cfg = cfg["audio_enhance"]
    denoised = work_dir / "graphic_denoised.mp4"
    if nr_cfg.get("enabled", True):
        audio_tools.reduce_noise(composited, denoised, work_dir,
                                  noise_profile_wav=noise_profile_wav,
                                  profile_start_sec=nr_cfg["profile_start_sec"],
                                  profile_end_sec=nr_cfg["profile_end_sec"])
    else:
        shutil.copy(composited, denoised)

    enhanced = work_dir / "graphic_enhanced.mp4"
    audio_tools.enhance_audio(
        denoised, enhanced,
        highpass_hz=ae_cfg["highpass_hz"],
        presence_boost_db=ae_cfg["presence_boost_db"],
        target_lufs=ae_cfg["target_lufs"],
        limiter_ceiling_db=ae_cfg["limiter_ceiling_db"],
    )
    return enhanced


def process_range(i, rng, raw_input, work_dir, cfg, mask_png, border_png, print_lock,
                   noise_profile_wav=None, background_image=None):
    """Process one range (talk or graphic) and return (index, output_path).
    Ranges are independent (each only touches its own range_NN_*/ subfolder
    and reads the shared source video read-only), so this is safe to run
    concurrently across threads."""
    rng_dir = work_dir / f"range_{i:02d}_{rng['kind']}"
    if rng["kind"] == "talk":
        with print_lock:
            print(f"[{i}] TALK  {rng['start']:.1f}s - {rng['end']:.1f}s (started)")
        result = process_talk_range(raw_input, rng["start"], rng["end"], rng_dir, cfg,
                                     noise_profile_wav=noise_profile_wav, background_image=background_image)
    else:
        with print_lock:
            print(f"[{i}] GRAPHIC  {rng['start']:.1f}s - {rng['end']:.1f}s : "
                  f"{rng['segment']['prompt'][:60]}... (started)")
        result = process_graphic_range(raw_input, rng, rng_dir, cfg, mask_png, border_png, noise_profile_wav=noise_profile_wav)
    with print_lock:
        print(f"[{i}] done")
    return i, result


def main(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())

    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path(cfg["output_video"])
    output_video.parent.mkdir(parents=True, exist_ok=True)
    raw_input = Path(cfg["input_video"])

    duration = audio_tools.probe_duration(raw_input)
    plan = build_range_plan(duration, cfg.get("segments", []))

    render_cfg = cfg["render"]
    pip_cfg = cfg["pip"]
    frame_w, frame_h = render_cfg["width"], render_cfg["height"]
    oval_w = int(frame_w * pip_cfg["width_pct"]) // 2 * 2  # libx264 needs even dimensions
    mask_dir = work_dir / "masks"
    mask_dir.mkdir(exist_ok=True)
    mask_png, border_png = pip_overlay.make_ellipse_mask(
        oval_w, oval_w, mask_dir / "ellipse_mask.png",
        border_width=pip_cfg.get("border_width_px", 6),
    )

    # Extract ONE noise profile from the ORIGINAL recording at the
    # user-configured quiet moment, reused for every range's noise reduction.
    # Sampling per-clip instead (the old behavior) grabs mostly-speech
    # windows since range clips rarely start on silence, which teaches
    # noisereduce to suppress the voice itself — audible as inconsistent
    # dull/muffled patches. See audio_tools.reduce_noise's docstring.
    nr_cfg = cfg["noise_reduction"]
    noise_profile_wav = None
    if nr_cfg.get("enabled", True):
        noise_profile_wav = work_dir / "noise_profile.wav"
        audio_tools.extract_noise_profile(
            raw_input, noise_profile_wav,
            nr_cfg["profile_start_sec"], nr_cfg["profile_end_sec"],
        )

    # One static, themed virtual background for the whole video — generated
    # once here and reused for every talk range, so the look is consistent
    # throughout instead of showing your real (possibly messy) room.
    bg_cfg = cfg.get("background", {})
    background_image = None
    if bg_cfg.get("enabled", False):
        accent_hex = bg_cfg.get("accent", "#2dd4bf").lstrip("#")
        accent_rgb = tuple(int(accent_hex[i:i + 2], 16) for i in (0, 2, 4))
        background_image = work_dir / "theme_background.png"
        background_replace.make_theme_background(frame_w, frame_h, background_image, accent=accent_rgb)

    max_workers = cfg.get("parallel_workers") or min(4, os.cpu_count() or 4)
    print(f"\nPlan: {len(plan)} ranges "
          f"({sum(1 for r in plan if r['kind']=='talk')} talk, "
          f"{sum(1 for r in plan if r['kind']=='graphic')} graphic) "
          f"— up to {max_workers} in parallel\n")

    print_lock = threading.Lock()
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_range, i, rng, raw_input, work_dir, cfg, mask_png, border_png,
                             print_lock, noise_profile_wav=noise_profile_wav, background_image=background_image)
            for i, rng in enumerate(plan)
        ]
        for future in as_completed(futures):
            i, result = future.result()
            results[i] = result

    pieces = [results[i] for i in range(len(plan))]

    print("\n=== Final render: concatenating all ranges ===")
    # Talk-range pieces keep the source recording's native resolution/aspect
    # ratio (e.g. 1620x1080) while graphic-range pieces are already rendered
    # at frame_w x frame_h — concatenating streams with different resolutions
    # via ffmpeg's "-f concat" DEMUXER corrupts PTS/duration (it assumes
    # homogeneous inputs). Use the concat FILTER instead: decode each piece,
    # normalize scale (preserving aspect ratio via letterbox, not stretching)
    # and fps, reset timestamps, then concatenate at the frame level.
    fps = render_cfg["fps"]
    inputs = []
    filter_parts = []
    concat_refs = []
    for i, p in enumerate(pieces):
        inputs += ["-i", str(p)]
        filter_parts.append(
            f"[{i}:v]scale={frame_w}:{frame_h}:force_original_aspect_ratio=decrease,"
            f"pad={frame_w}:{frame_h}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1,fps={fps},setpts=PTS-STARTPTS[v{i}]"
        )
        filter_parts.append(f"[{i}:a]aresample=48000,asetpts=PTS-STARTPTS[a{i}]")
        concat_refs.append(f"[v{i}][a{i}]")
    filter_complex = ";".join(filter_parts) + ";" + "".join(concat_refs) + f"concat=n={len(pieces)}:v=1:a=1[outv][outa]"

    final_path = work_dir / "final_rendered.mp4"
    audio_tools.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", render_cfg["video_codec"], "-crf", str(render_cfg["crf"]), "-preset", "veryfast",
        "-c:a", render_cfg["audio_codec"], "-b:a", render_cfg["audio_bitrate"],
        str(final_path)
    ])

    shutil.copy(final_path, output_video)
    print(f"\nDone. Final file: {output_video.resolve()}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 pipeline.py config.yaml")
        sys.exit(1)
    main(sys.argv[1])
