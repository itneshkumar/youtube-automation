#!/usr/bin/env python3
"""
audio_tools.py — ffmpeg/noisereduce helpers used by pipeline.py:
subprocess runner, duration probing, silence removal, noise reduction, and
loudness/EQ enhancement.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import noisereduce as nr
import soundfile as sf


def run(cmd):
    """Run a subprocess command, raising with captured stderr on failure."""
    cmd = [str(c) for c in cmd]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"--- stderr (tail) ---\n{result.stderr[-4000:]}"
        )
    return result


def probe_duration(path):
    """Return duration of a media file in seconds via ffprobe."""
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ])
    return float(json.loads(result.stdout)["format"]["duration"])


def _probe_fps(path):
    """Return the video stream's frame rate (frames/sec) via ffprobe."""
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(path),
    ])
    num, den = result.stdout.strip().split("/")
    return float(num) / float(den)


def _detect_silence(path, noise_floor_db, min_silence_sec):
    """Return [(start, end), ...] silence ranges via ffmpeg's silencedetect filter."""
    result = run([
        "ffmpeg", "-i", str(path),
        "-af", f"silencedetect=noise={noise_floor_db}dB:d={min_silence_sec}",
        "-f", "null", "-",
    ])
    log = result.stderr
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", log)]
    return list(zip(starts, ends))


def cut_silence(input_path, output_path, work_dir, noise_floor_db, min_silence_sec, padding):
    """
    Detect silence in input_path and cut it out (keeping `padding` seconds of
    headroom at each edge so speech isn't clipped), writing the result to
    output_path. Falls back to a straight copy if nothing was flagged.

    Cuts and reassembles video+audio in a SINGLE ffmpeg filter pass
    (per-range trim/atrim, cut points snapped to the video's own frame grid,
    + the concat filter) rather than encoding each keep-range to its own
    file and stitching with the concat demuxer. The old per-segment approach
    re-encoded each range independently, and video (frame-quantized) vs.
    audio (sample-exact) rounded to slightly different actual durations per
    cut -- individually a few ms, but summed across dozens of silence cuts
    in one talk range it compounded linearly into hundreds of ms of audible
    lip-sync drift. Snapping cut points to frame boundaries up front makes
    each individual piece's audio and video duration match exactly, so nothing
    accumulates with cut count any more; verified against real ranges, drift
    dropped from +150-160ms (growing with cut count) to isolated tens-of-ms
    that don't compound, from two sources this function can't remove: an
    occasional single real dropped/duplicated frame in the source, and the
    last piece in a range whose video track can end a frame or two before its
    audio track (inherited from the upstream -ss/-to slice, not from cutting
    silence here). (An earlier version of this used select/aselect with
    between() conditions, matching ffmpeg's documented idiom for this exact
    scenario -- but aselect is a no-op passthrough on this machine's ffmpeg
    8.0.1 build regardless of the condition, so it silently kept the entire
    audio track. trim/atrim+concat achieves the same single-pass result
    without depending on that filter.)
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(input_path)
    silences = _detect_silence(input_path, noise_floor_db, min_silence_sec)

    if not silences:
        shutil.copy(input_path, output_path)
        return output_path

    keep_ranges = []
    cursor = 0.0
    for s_start, s_end in silences:
        cut_start = min(s_start + padding, s_end)
        cut_end = max(s_end - padding, cut_start)
        if cut_start > cursor:
            keep_ranges.append((cursor, cut_start))
        cursor = max(cursor, cut_end)
    if cursor < duration:
        keep_ranges.append((cursor, duration))

    keep_ranges = [(s, e) for s, e in keep_ranges if e - s >= 0.05]
    if not keep_ranges:
        shutil.copy(input_path, output_path)
        return output_path

    # trim() below can only cut on frame boundaries anyway; snapping the
    # requested cut points to the same grid up front means atrim() (which
    # IS sample-exact) cuts audio to precisely the same duration as the
    # video for every piece, instead of leaving a fractional-frame gap per
    # cut that would otherwise accumulate into drift across the concat.
    # The one piece that reaches the actual end of the file is left
    # open-ended (no explicit "end=") rather than snapped to an exact frame
    # count: the container's reported duration is sometimes a hair past the
    # last actually-decodable video frame, and requesting a frame count
    # derived from it can silently come up short right at EOF.
    fps = _probe_fps(input_path)
    frame = 1.0 / fps
    snapped = []
    for s, e in keep_ranges:
        snapped_s = round(s / frame) * frame
        reaches_eof = e >= duration - frame / 2
        if reaches_eof:
            snapped.append((snapped_s, None))
        else:
            n_frames = round((e - s) / frame)
            if n_frames > 0:
                snapped.append((snapped_s, snapped_s + n_frames * frame))
    keep_ranges = snapped

    parts = []
    pairs = []
    for i, (s, e) in enumerate(keep_ranges):
        v_end = f":end={e:.6f}" if e is not None else ""
        parts.append(f"[0:v]trim=start={s:.6f}{v_end},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={s:.6f}{v_end},asetpts=PTS-STARTPTS[a{i}]")
        pairs.append(f"[v{i}][a{i}]")
    filter_complex = (
        ";".join(parts) + ";" + "".join(pairs) +
        f"concat=n={len(keep_ranges)}:v=1:a=1[v][a]"
    )
    run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "pcm_s16le",
        str(output_path)
    ])
    return output_path


def extract_noise_profile(source_path, output_wav, start_sec, end_sec):
    """
    Extract a short wav snippet from source_path[start_sec:end_sec] to reuse
    as a fixed noise profile across every reduce_noise() call. Call this ONCE
    against the ORIGINAL recording at a genuinely quiet, speech-free moment —
    never per-range-clip, since a cut clip's own first second is almost
    always mid-speech, not silence (see reduce_noise's noise_profile_wav
    docstring for why that matters).
    """
    run([
        "ffmpeg", "-y", "-i", str(source_path),
        "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}",
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        str(output_wav)
    ])
    return output_wav


def reduce_noise(input_path, output_path, work_dir, noise_profile_wav=None,
                  profile_start_sec=0.0, profile_end_sec=1.0):
    """
    Extract audio and run noisereduce over the full track, then mux the
    cleaned audio back with the original video stream.

    Pass noise_profile_wav (from extract_noise_profile(), sampled once from a
    genuinely quiet moment in the ORIGINAL recording) — this is the correct
    mode. Without it, this falls back to sampling [profile_start_sec,
    profile_end_sec] of *this clip*, which is unreliable: for a range that's
    been cut out of the middle of a talk (or silence-cut), that window is
    almost always actual speech, not silence, so noisereduce ends up learning
    the speaker's voice as "noise" and suppressing it — audible as inconsistent
    dull/muffled patches across segments.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_wav = work_dir / "nr_raw.wav"
    clean_wav = work_dir / "nr_clean.wav"

    run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1",
        str(raw_wav)
    ])

    audio, sr = sf.read(raw_wav)

    if noise_profile_wav is not None:
        noise_clip, _ = sf.read(noise_profile_wav)
    else:
        duration = len(audio) / sr
        p_start = max(0.0, min(profile_start_sec, duration))
        p_end = max(p_start, min(profile_end_sec, duration))
        noise_clip = audio[int(p_start * sr):int(p_end * sr)] if p_end > p_start else None

    reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip, stationary=True)
    sf.write(clean_wav, reduced, sr)

    run([
        "ffmpeg", "-y", "-i", str(input_path), "-i", str(clean_wav),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le",
        "-shortest",
        str(output_path)
    ])
    return output_path


def enhance_audio(input_path, output_path, highpass_hz, presence_boost_db, target_lufs, limiter_ceiling_db):
    """Highpass rumble -> presence boost -> loudness normalize -> ceiling limiter."""
    af = (
        f"highpass=f={highpass_hz},"
        f"equalizer=f=3000:t=q:w=1:g={presence_boost_db},"
        f"loudnorm=I={target_lufs}:TP={limiter_ceiling_db}:LRA=11,"
        f"alimiter=limit={_db_to_linear(limiter_ceiling_db):.6f}"
    )
    run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "copy", "-af", af,
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ])
    return output_path


def _db_to_linear(db):
    return 10 ** (db / 20)
