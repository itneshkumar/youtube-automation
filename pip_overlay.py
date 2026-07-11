#!/usr/bin/env python3
"""
pip_overlay.py — elliptical webcam bubble: mask generation, cropping the
webcam segment into an oval, and compositing it over a motion graphic.
"""

from pathlib import Path

from PIL import Image, ImageDraw

import audio_tools


def make_ellipse_mask(w, h, output_path, border_width=6):
    """
    Write a grayscale ellipse mask (white=opaque interior, used with
    ffmpeg's alphamerge) to output_path, plus a matching ring-shaped border
    PNG (with alpha) next to it. Returns (mask_path, border_path).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, w - 1, h - 1), fill=255)
    mask.save(output_path)

    border = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(border)
    inset = border_width / 2
    draw.ellipse(
        (inset, inset, w - 1 - inset, h - 1 - inset),
        outline=(255, 255, 255, 255), width=border_width,
    )
    border_path = output_path.with_name(output_path.stem + "_border.png")
    border.save(border_path)

    return output_path, border_path


def build_pip_clip(raw_input, start, end, oval_w, oval_h, mask_png, work_dir, border_png=None):
    """
    Crop [start,end] of the webcam feed in raw_input to a centered
    oval_w x oval_h square, apply the ellipse mask as alpha, optionally
    overlay a border ring, and return the path to a QTRLE .mov clip that
    carries alpha plus the original mic audio for that range.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    square_path = work_dir / "pip_square.mkv"
    audio_tools.run([
        "ffmpeg", "-y", "-i", str(raw_input),
        "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-vf", f"crop='min(iw,ih)':'min(iw,ih)',scale={oval_w}:{oval_h}",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "pcm_s16le",
        str(square_path)
    ])

    foreground = square_path
    if border_png:
        bordered_path = work_dir / "pip_bordered.mkv"
        audio_tools.run([
            "ffmpeg", "-y", "-i", str(square_path), "-i", str(border_png),
            "-filter_complex", "[0:v][1:v]overlay=0:0[out]",
            "-map", "[out]", "-map", "0:a",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-c:a", "copy",
            str(bordered_path)
        ])
        foreground = bordered_path

    pip_clip = work_dir / "pip_clip.mov"
    audio_tools.run([
        "ffmpeg", "-y", "-i", str(foreground), "-i", str(mask_png),
        "-filter_complex", "[0:v][1:v]alphamerge[out]",
        "-map", "[out]", "-map", "0:a",
        "-c:v", "qtrle", "-c:a", "pcm_s16le",
        str(pip_clip)
    ])
    return pip_clip


_POSITION_MAP = {
    "bottom-right": lambda fw, fh, ow, oh, m: (fw - ow - m, fh - oh - m),
    "bottom-left": lambda fw, fh, ow, oh, m: (m, fh - oh - m),
    "top-right": lambda fw, fh, ow, oh, m: (fw - ow - m, m),
    "top-left": lambda fw, fh, ow, oh, m: (m, m),
    "center": lambda fw, fh, ow, oh, m: ((fw - ow) // 2, (fh - oh) // 2),
}


def composite_over_graphic(graphic_video, pip_clip, output_path, frame_w, frame_h, position, oval_w, margin_px):
    """Overlay the oval pip_clip onto graphic_video at `position`, keeping the pip's mic audio."""
    if position not in _POSITION_MAP:
        raise ValueError(f"Unknown pip position '{position}', expected one of {list(_POSITION_MAP)}")
    x, y = _POSITION_MAP[position](frame_w, frame_h, oval_w, oval_w, margin_px)

    audio_tools.run([
        "ffmpeg", "-y", "-i", str(graphic_video), "-i", str(pip_clip),
        "-filter_complex", f"[0:v][1:v]overlay={x}:{y}:shortest=1[out]",
        "-map", "[out]", "-map", "1:a",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "256k",
        str(output_path)
    ])
    return output_path
