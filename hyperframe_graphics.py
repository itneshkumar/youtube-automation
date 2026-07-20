#!/usr/bin/env python3
"""
hyperframe_graphics.py — alternate graphics backend for process_graphic_range,
rendering each segment's motion graphic via HyperFrames (`npx hyperframes
render`) instead of graphics_llm.py's Playwright/CSS templates. Same content
pipeline as graphics_llm.py (LLM extracts an ordered step list; layout/timing
stays fixed template code) — only the rendering engine differs, swapping raw
CSS keyframes for a HyperFrames/GSAP composition with the framework's own
lint/contrast/motion checks.

Opt-in only, via HYPERFRAME_GRAPHICS=true in .env (or
graphics.hyperframe_rendering: true in config.yaml) — graphics_llm.py stays
the default backend either way. See pipeline.py's process_graphic_range for
the branch point.

The composition template was authored and validated (`npx hyperframes check`
— 0 lint errors, 40/40 WCAG AA contrast checks) directly in
hyperframes_graphics/, the scaffolded HyperFrames project this module reuses
per render; see that project's index.html for the reference version this
module's _build_composition_html mirrors.
"""

import html as html_lib
import json
import random
import subprocess
from pathlib import Path

import graphics_llm

HF_PROJECT_DIR = Path(__file__).resolve().parent / "hyperframes_graphics"

# Same rotation as graphics_llm.py's dark technical system: teal / amber /
# coral, each (accent, badge_fill, detail_text).
_ACCENT_PALETTE = [
    ("#2dd4bf", "#0f2321", "#b9c7c5"),  # teal
    ("#f0a83c", "#2a1f0f", "#cfc0a3"),  # amber
    ("#ff6b5f", "#2a1512", "#c9a8a3"),  # coral
]


def _build_composition_html(steps, delays, duration_sec, width, height, pip_position, accent):
    accent_color, badge_fill, detail_text = accent
    n = len(steps)

    cards, tweens = [], []
    for i, step in enumerate(steps):
        label = html_lib.escape(step.get("label") or f"Step {i + 1}")
        detail = html_lib.escape(step.get("detail") or "")
        delay = delays[i]
        clip_duration = max(duration_sec - delay, 0.5)
        detail_html = f'<div class="detail">{detail}</div>' if detail else ""
        cards.append(
            f'<div id="step-{i + 1}" class="clip step" data-start="{delay:.2f}" '
            f'data-duration="{clip_duration:.2f}" data-track-index="{i + 1}">'
            f'<div class="badge">{i + 1}</div>'
            f'<div class="text"><div class="eyebrow">Step {i + 1:02d} / {n:02d}</div>'
            f'<div class="label">{label}</div>{detail_html}</div>'
            f'</div>'
        )
        tweens.append(
            f'tl.from("#step-{i + 1}", {{ y: 24, opacity: 0, duration: 0.6, ease: "power3.out" }}, {delay:.2f});'
        )
    cards_html = "\n          ".join(cards)
    tweens_js = "\n      ".join(tweens)

    # Keep clear of whichever corner the webcam bubble docks in, same rule
    # graphics_llm.py's vertical layout uses.
    pip_position = pip_position or "bottom-right"
    on_left = "left" in pip_position
    stage_justify = "flex-end" if on_left else "flex-start"
    stage_padding = "padding-right: 6%;" if on_left else "padding-left: 6%;"

    return f"""<!doctype html>
<html lang="en" data-resolution="landscape">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={width}, height={height}" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@500;700&display=swap"
      rel="stylesheet"
    />
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{
        width: {width}px; height: {height}px; overflow: hidden;
        background: #0b0c0e; font-family: "Space Grotesk", -apple-system, sans-serif;
      }}
      #root {{ position: relative; width: 100%; height: 100%; }}
      #root::before {{
        content: ""; position: absolute; inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.05) 1px, transparent 1px);
        background-size: 26px 26px;
      }}
      .stage {{
        position: absolute; inset: 0; display: flex; align-items: center;
        justify-content: {stage_justify}; {stage_padding}
      }}
      .steps {{
        position: relative; display: flex; flex-direction: column;
        align-items: flex-start; gap: 22px; width: 58%; max-width: 900px;
      }}
      .step {{
        position: relative; display: flex; align-items: center; gap: 22px;
        width: 100%; box-sizing: border-box; padding: 28px 36px; border-radius: 18px;
        background: #131a1d;
        box-shadow: 0 14px 40px -14px {accent_color}60, inset 0 1px 0 rgba(255,255,255,0.06),
          inset 0 0 0 1px {accent_color}40;
      }}
      .badge {{
        flex: 0 0 auto; width: 64px; height: 64px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        background: {badge_fill}; border: 2px solid {accent_color}; color: {accent_color};
        font-family: "JetBrains Mono", monospace; font-size: 30px; font-weight: 700;
      }}
      .text .eyebrow {{
        font-family: "JetBrains Mono", monospace; font-size: 18px; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase; color: {accent_color};
        margin-bottom: 6px;
      }}
      .text .label {{ font-size: 38px; font-weight: 700; color: #fff; line-height: 1.25; letter-spacing: -0.01em; }}
      .text .detail {{ font-size: 24px; color: {detail_text}; margin-top: 4px; }}
    </style>
  </head>
  <body>
    <div
      id="root"
      data-composition-id="main"
      data-start="0"
      data-duration="{duration_sec:.2f}"
      data-width="{width}"
      data-height="{height}"
    >
      <div class="stage">
        <div class="steps">
          {cards_html}
        </div>
      </div>
    </div>

    <script>
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      {tweens_js}
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>"""


def generate_motion_graphic(prompt_text, work_dir, duration_sec, width, height,
                             transcript_entries=None, range_start=0.0, pip_position="bottom-right",
                             fps=30, backend=None, model=None, llamacpp_server=None, timeout=180,
                             max_attempts=2):
    """
    Extract steps from the transcript (same extraction/timing as
    graphics_llm.py), render a HyperFrames composition, return the mp4 path.
    """
    backend = backend or graphics_llm.DEFAULT_BACKEND
    llamacpp_server = llamacpp_server or graphics_llm.DEFAULT_LLAMACPP_SERVER
    transcript_text = " ".join(e["text"].strip() for e in (transcript_entries or []))
    steps = graphics_llm.extract_steps(
        prompt_text, transcript_text, backend=backend, model=model,
        llamacpp_server=llamacpp_server, timeout=timeout, max_attempts=max_attempts,
    )
    delays = graphics_llm._assign_step_delays(steps, transcript_entries or [], range_start, duration_sec)
    accent = random.choice(_ACCENT_PALETTE)
    html_doc = _build_composition_html(steps, delays, duration_sec, width, height, pip_position, accent)

    work_dir = Path(work_dir)
    render_dir = work_dir / "hf_project"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "index.html").write_text(html_doc)
    (render_dir / "hyperframes.json").write_text(
        (HF_PROJECT_DIR / "hyperframes.json").read_text()
    )

    mp4_path = work_dir / "graphic_video.mp4"
    subprocess.run(
        ["npx", "hyperframes", "render", str(render_dir),
         "--output", str(mp4_path.resolve()), "--fps", str(fps), "--quality", "standard"],
        check=True, capture_output=True, text=True,
    )
    return mp4_path
