#!/usr/bin/env python3
"""
background_replace.py — replaces the real webcam background in talk-range
footage with a single static, themed virtual background, using MediaPipe
person segmentation. The same background image is generated once per
pipeline run and reused for every talk range, so the look stays consistent
throughout the whole video.
"""

from pathlib import Path

import cv2
import numpy as np
import requests
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

import audio_tools

MODEL_PATH = Path(__file__).resolve().parent / "models" / "selfie_segmenter.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)


def ensure_model(model_path=MODEL_PATH):
    """Download the selfie-segmentation model on first use if it's not already present."""
    model_path = Path(model_path)
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(MODEL_URL, timeout=60)
    resp.raise_for_status()
    model_path.write_bytes(resp.content)
    return model_path


def make_theme_background(width, height, output_path, accent=(45, 212, 191)):
    """
    Generate a static, dark tech-themed background image matching the motion
    graphics' aesthetic: near-black base with a soft off-center accent glow
    and a vignette. Deterministic — same accent always produces the same
    background, so re-runs stay visually consistent.
    """
    from PIL import Image, ImageDraw, ImageFilter

    base = (8, 9, 11)
    img = Image.new("RGB", (width, height), base)

    glow = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(glow)
    cx, cy = int(width * 0.72), int(height * 0.30)
    max_r = int(max(width, height) * 0.6)
    steps = 60
    for s in range(steps, 0, -1):
        r = int(max_r * s / steps)
        t = s / steps
        color = tuple(int(base[c] + (accent[c] - base[c]) * (1 - t) ** 2) for c in range(3))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    glow = glow.filter(ImageFilter.GaussianBlur(80))
    img = Image.blend(img, glow, 0.85)

    vignette = Image.new("L", (width, height), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse((-width * 0.15, -height * 0.15, width * 1.15, height * 1.15), fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(220))
    dark = Image.new("RGB", (width, height), (2, 2, 3))
    img = Image.composite(img, dark, vignette)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def _isolate_main_subject(mask, threshold=0.5, dilate_px=15):
    """
    Zero out any confident region not connected to the largest blob. A
    lightweight segmentation model can give spurious mid/low confidence to
    an unrelated background object (tested case: a bag on a rack at the
    frame edge) — that shows up as a small blob disconnected from the real
    subject, which is always the biggest one. Dilate the kept blob a bit
    first so this doesn't clip soft hair/edge confidence around the subject.
    """
    binary = (mask > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return mask  # nothing above threshold at all
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (labels == largest_label).astype(np.uint8)
    keep = cv2.dilate(keep, np.ones((dilate_px, dilate_px), np.uint8))
    return mask * keep.astype(np.float32)


def _add_rim_glow(composited_bgr, person_mask, accent_bgr, ring_px=3, glow_px=5, strength=0.55):
    """
    Draw a soft rim line along the silhouette edge, in the same accent
    color used elsewhere (motion-graphic badges, pip bubble border) — turns
    a bare cutout edge into a deliberate-looking design element, the same
    idea as the oval border around the webcam bubble in graphic segments.
    Tuned for a crisp thin line, not a diffuse neon glow.
    """
    binary = (person_mask > 0.5).astype(np.uint8) * 255
    dilated = cv2.dilate(binary, np.ones((ring_px * 2 + 1, ring_px * 2 + 1), np.uint8))
    eroded = cv2.erode(binary, np.ones((ring_px, ring_px), np.uint8))
    ring = cv2.subtract(dilated, eroded).astype(np.float32) / 255.0
    ring = cv2.GaussianBlur(ring, (glow_px | 1, glow_px | 1), 0)
    ring_3 = np.repeat(ring[:, :, np.newaxis], 3, axis=2)

    accent_layer = np.full_like(composited_bgr, accent_bgr, dtype=np.float32)
    glowed = composited_bgr.astype(np.float32) + accent_layer * ring_3 * strength
    return np.clip(glowed, 0, 255).astype(np.uint8)


def replace_background(input_path, output_path, background_image_path, work_dir,
                        model_path=MODEL_PATH, feather_px=9, accent_rgb=None, rim_glow_px=4):
    """
    Run person segmentation on every frame of input_path (video) and
    composite the person over background_image_path (static image, resized
    to match). Re-muxes the original audio track back in unchanged —
    this function only touches video.

    accent_rgb (R, G, B), if given, draws a soft glowing rim along the
    silhouette edge in that color (set rim_glow_px=0 to disable).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = ensure_model(model_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_path} for background replacement")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg = cv2.imread(str(background_image_path))
    if bg is None:
        raise RuntimeError(f"Could not read background image {background_image_path}")
    bg = cv2.resize(bg, (w, h))

    silent_video = work_dir / "bg_replaced_silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(silent_video), fourcc, fps, (w, h))

    base_options = BaseOptions(model_asset_path=str(model_path))
    # Soft per-pixel confidence (0..1) gives much cleaner hair/edge detail
    # than a hard category_mask thresholded and then blurred — tested
    # head-to-head against both the old hard-mask approach and rembg
    # (u2net_human_seg, which was both ~80x slower and introduced a visible
    # chunk missing from the hair on the same test frame). Same model, same
    # ~12ms/frame speed, just reading the better output the model offers.
    options = vision.ImageSegmenterOptions(base_options=base_options, output_confidence_masks=True)

    k = feather_px | 1  # odd kernel size required by GaussianBlur
    accent_bgr = tuple(reversed(accent_rgb)) if accent_rgb else None
    smoothed_mask = None

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = segmenter.segment(mp_image)
            person_mask = result.confidence_masks[0].numpy_view().squeeze().astype(np.float32)

            # Drop any confident region not connected to the main (largest)
            # blob — e.g. a background object the model gives spurious
            # "person" confidence to (verified case: a bag/clothing item at
            # the frame edge). The real subject is always the biggest blob.
            person_mask = _isolate_main_subject(person_mask)

            # Exponential moving average across frames damps single-frame
            # segmentation noise/flicker without lagging behind real motion.
            if smoothed_mask is None:
                smoothed_mask = person_mask
            else:
                smoothed_mask = 0.5 * person_mask + 0.5 * smoothed_mask
            person_mask = smoothed_mask

            # Erode slightly before feathering: partial-alpha edge pixels
            # blend in a sliver of the ORIGINAL room color (no clean-plate
            # background to properly decontaminate against), which reads as
            # a thin color fringe around the silhouette — pulling the
            # boundary in a couple pixels trims that sliver off.
            person_mask = cv2.erode(person_mask, np.ones((3, 3), np.uint8))
            if feather_px > 0:
                person_mask = cv2.GaussianBlur(person_mask, (k, k), 0)
            person_mask_3 = np.repeat(person_mask[:, :, np.newaxis], 3, axis=2)

            composited = (frame_bgr.astype(np.float32) * person_mask_3 +
                          bg.astype(np.float32) * (1 - person_mask_3)).astype(np.uint8)

            if accent_bgr is not None and rim_glow_px > 0:
                composited = _add_rim_glow(composited, person_mask, accent_bgr, ring_px=rim_glow_px)

            writer.write(composited)

    cap.release()
    writer.release()

    # VideoWriter's mp4v output is a large intermediate — re-encode to h264
    # and mux the original audio back in from input_path unchanged.
    audio_tools.run([
        "ffmpeg", "-y", "-i", str(silent_video), "-i", str(input_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "copy",
        "-shortest",
        str(output_path)
    ])
    return output_path
