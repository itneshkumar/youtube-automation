#!/usr/bin/env python3
"""
outro.py — animated "subscribe & like" call-to-action, composited over the
last few seconds of the final render. A red subscribe pill (bell icon,
pop-in bounce) and an accent-colored "Like" pill animate in with alpha
transparency so they sit over whatever footage is already playing there.

Uses the same frame-accurate capture approach as
graphics_llm.render_html_to_video (pause CSS animations, step each one to
an exact currentTime, screenshot once per output frame) but keeps the page
background transparent and encodes to qtrle (the same alpha-preserving
codec pip_overlay.py uses for the oval webcam bubble) instead of x264,
since x264 can't carry an alpha channel.
"""

import subprocess
from pathlib import Path

import audio_tools


def make_subscribe_html(width, height, accent, channel_name=""):
    accent_color, badge_bg, badge_text = accent
    label = f"Subscribe{f' to {channel_name}' if channel_name else ''}"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; width:{width}px; height:{height}px; background:transparent;
    overflow:hidden; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .stage {{ position:absolute; inset:0; display:flex; align-items:flex-end; justify-content:center;
    padding-bottom: 10%; gap: 2vh; }}
  .pill {{ display:flex; align-items:center; gap:1.2vh; padding: 1.6vh 2.8vh; border-radius:999px;
    opacity:0; transform: translateY(30px) scale(0.85);
    animation-name: pop; animation-duration:0.6s; animation-timing-function: cubic-bezier(.2,1.4,.4,1);
    animation-fill-mode: forwards; box-shadow: 0 12px 32px -8px rgba(0,0,0,0.6); }}
  .subscribe {{ background: linear-gradient(135deg, #ff3b30, #b8140a); animation-delay: 0.1s; }}
  .like {{ background: linear-gradient(135deg, {accent_color}, {badge_bg}); animation-delay: 0.45s; }}
  .pill .icon {{ font-size:3vh; line-height:1; }}
  .pill .text {{ font-size:2.4vh; font-weight:800; color:#fff; white-space:nowrap; }}
  .bell {{ display:inline-block; transform-origin: 50% 0%;
    animation-name: ring; animation-duration:0.6s; animation-delay:0.75s;
    animation-timing-function: ease-in-out; animation-fill-mode: backwards; }}
  @keyframes pop {{
    0%   {{ opacity:0; transform: translateY(30px) scale(0.85); }}
    60%  {{ opacity:1; transform: translateY(-6px) scale(1.06); }}
    100% {{ opacity:1; transform: translateY(0) scale(1); }}
  }}
  @keyframes ring {{
    0%, 100% {{ transform: rotate(0deg); }}
    20% {{ transform: rotate(-16deg); }}
    40% {{ transform: rotate(13deg); }}
    60% {{ transform: rotate(-9deg); }}
    80% {{ transform: rotate(5deg); }}
  }}
</style></head>
<body>
  <div class="stage">
    <div class="pill subscribe"><span class="icon bell">\U0001F514</span><span class="text">{label}</span></div>
    <div class="pill like"><span class="icon">\U0001F44D</span><span class="text">Like</span></div>
  </div>
</body></html>"""


def render_alpha_overlay(html_str, work_dir, duration_sec, width, height, fps=30):
    """
    Same frame-by-frame CSS-animation capture as
    graphics_llm.render_html_to_video, but with a transparent page
    background (screenshot(omit_background=True)) encoded to qtrle so the
    alpha channel survives, instead of x264 which is opaque-only.
    """
    from playwright.sync_api import sync_playwright

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / "outro.html"
    html_path.write_text(html_str)
    mov_path = work_dir / "outro.mov"
    n_frames = max(round(duration_sec * fps), 1)

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
            "-frames:v", str(n_frames),
            "-c:v", "qtrle",
            str(mov_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(html_path.resolve().as_uri())
        page.evaluate("document.getAnimations().forEach(a => a.pause())")
        for i in range(n_frames):
            t_ms = i * (1000 / fps)
            page.evaluate(f"document.getAnimations().forEach(a => a.currentTime = {t_ms})")
            encoder.stdin.write(page.screenshot(type="png", omit_background=True))
        browser.close()

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg alpha overlay encode failed: {encoder.stderr.read().decode(errors='replace')}")
    return mov_path


def apply_outro(input_video, output_path, work_dir, duration_sec=4.0, width=1920, height=1080,
                 fps=30, accent=(45, 212, 191), channel_name=""):
    """
    Composite the subscribe/like pop-in over the LAST duration_sec seconds
    of input_video, writing output_path. Everything before that point is
    untouched — this only overlays, it doesn't extend the video.
    """
    work_dir = Path(work_dir)
    html_str = make_subscribe_html(width, height, accent, channel_name)
    overlay_mov = render_alpha_overlay(html_str, work_dir, duration_sec, width, height, fps=fps)

    total = audio_tools.probe_duration(input_video)
    start = max(total - duration_sec, 0.0)

    audio_tools.run([
        "ffmpeg", "-y", "-i", str(input_video), "-i", str(overlay_mov),
        "-filter_complex",
        f"[1:v]setpts=PTS-STARTPTS+{start:.3f}/TB[ov];"
        f"[0:v][ov]overlay=0:0:enable='gte(t,{start:.3f})'[outv]",
        "-map", "[outv]", "-map", "0:a",
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-c:a", "copy",
        str(output_path)
    ])
    return output_path
