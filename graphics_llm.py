#!/usr/bin/env python3
"""
graphics_llm.py — turns a segment's real transcript into a labeled, ordered,
step-by-step animated HTML explainer graphic, recorded to video by a
headless browser (Playwright).

Split responsibility on purpose: the LLM only extracts the ordered list of
steps/concepts the narrator actually names (a task even small local models
do reliably) — it does NOT write the animation's JS/timing itself (small
models write plausible-looking but non-functional or non-sequenced
animation code; see graphics_llm history). The reveal timing, layout, and
CSS animation are a fixed, tested template driven by that step list, so the
visual flow reliably matches the narration order regardless of model size.

Two backends for step extraction:
  ollama (default)  — free, fully local. Needs `ollama serve` running and a
                       model pulled (e.g. `ollama pull llama3.1`).
  anthropic          — better extraction quality, costs money. Needs
                       ANTHROPIC_API_KEY in .env or the environment.
"""

import html as html_lib
import json
import random
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

import audio_tools

load_dotenv()

DEFAULT_BACKEND = "ollama"
DEFAULT_OLLAMA_MODEL = "phi3.5"
DEFAULT_OLLAMA_SERVER = "http://127.0.0.1:11434"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"

STEPS_SYSTEM_PROMPT = (
    "You analyze a short transcript segment from an educational/tech video "
    "and extract the distinct steps, concepts, or entities the narrator "
    "names, in the exact order they are mentioned. Return STRICT JSON ONLY: "
    "an array of 2 to 6 objects, each "
    '{"label": "2-4 word title", "detail": "a short one-sentence explanation, under 12 words"}. '
    "No markdown fences, no prose before or after the JSON. If the "
    "transcript doesn't clearly name distinct steps, return a single-item "
    "array that summarizes the topic label instead."
)


def _steps_user_message(prompt_text, transcript_text):
    return (
        f"Topic label: {prompt_text}\n"
        f"Transcript (in spoken order): "
        f"{transcript_text.strip() if transcript_text else '(not available — use the topic label only)'}"
    )


def _extract_steps_via_anthropic(prompt_text, transcript_text, model):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or DEFAULT_ANTHROPIC_MODEL,
        max_tokens=1024,
        system=STEPS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _steps_user_message(prompt_text, transcript_text)}],
    )
    return next(b.text for b in response.content if b.type == "text")


def _extract_steps_via_ollama(prompt_text, transcript_text, model, server, timeout):
    resp = requests.post(
        f"{server.rstrip('/')}/api/generate",
        json={
            "model": model or DEFAULT_OLLAMA_MODEL,
            "system": STEPS_SYSTEM_PROMPT,
            "prompt": _steps_user_message(prompt_text, transcript_text),
            "stream": False,
            "options": {"num_predict": 800, "temperature": 0.4},
        },
        timeout=timeout,
    )
    if resp.status_code == 404:
        raise RuntimeError(
            f"Ollama model '{model or DEFAULT_OLLAMA_MODEL}' not found. "
            f"Pull it first: ollama pull {model or DEFAULT_OLLAMA_MODEL}"
        )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_steps_json(raw_text):
    cleaned = re.sub(r"^```(json)?\s*|```\s*$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, list):
        return []
    steps = []
    for item in data:
        if isinstance(item, dict) and item.get("label"):
            steps.append({
                "label": str(item["label"])[:60],
                "detail": str(item.get("detail", ""))[:160],
            })
    return steps


def extract_steps(prompt_text, transcript_text, backend=DEFAULT_BACKEND, model=None,
                   ollama_server=DEFAULT_OLLAMA_SERVER, timeout=180):
    """Ask an LLM for the ordered list of steps/concepts named in transcript_text."""
    if backend == "ollama":
        raw_text = _extract_steps_via_ollama(prompt_text, transcript_text, model, ollama_server, timeout)
    elif backend == "anthropic":
        raw_text = _extract_steps_via_anthropic(prompt_text, transcript_text, model)
    else:
        raise ValueError(f"Unknown graphics backend '{backend}', expected 'ollama' or 'anthropic'")

    steps = _parse_steps_json(raw_text)
    if not steps:
        steps = [{"label": prompt_text[:60], "detail": ""}]
    return steps[:6]


_ICON_KEYWORDS = [
    (("perceive", "sense", "observe", "input", "detect", "see"), "\U0001F441️"),      # eye
    (("reason", "think", "decide", "process", "analyz", "plan"), "\U0001F9E0"),            # brain
    (("act", "execute", "run", "perform", "action"), "⚡"),                             # bolt
    (("learn", "adapt", "improve", "train", "feedback", "grow"), "\U0001F4C8"),             # chart
    (("data", "database", "store", "storage", "embedding"), "\U0001F5C4️"),            # file cabinet
    (("request", "send", "push", "upload", "call"), "\U0001F4E4"),                          # outbox
    (("response", "receive", "pull", "download", "get", "return"), "\U0001F4E5"),           # inbox
    (("server",), "\U0001F5A5️"),                                                       # desktop
    (("client", "user", "browser"), "\U0001F4BB"),                                          # laptop
    (("network", "api", "http", "web", "internet"), "\U0001F310"),                          # globe
    (("loop", "cycle", "repeat", "iterate", "again"), "\U0001F501"),                        # repeat
    (("code", "build", "compile", "program", "dockerfile", "image", "container", "deploy"), "\U0001F6E0️"),  # tools
    (("security", "auth", "token", "login", "credential", "password"), "\U0001F512"),       # lock
    (("search", "query", "lookup", "find", "compare", "match"), "\U0001F50D"),              # magnifier
    (("file", "document", "text"), "\U0001F4C4"),                                           # document
    (("agent", "autonomous", "robot"), "\U0001F916"),                                       # robot
    (("model", "weight", "neural", "layer"), "\U0001F9E9"),                                 # puzzle piece
]


def _pick_icon(label, detail):
    # Check the label first (short, purposeful) before the detail sentence
    # (freeform, likely to contain incidental words that cause false
    # substring matches — e.g. "Act: takes action based on input" should
    # not match the "perceive" group just because "input" appears there).
    # Word-boundary matching avoids "react" falsely matching "act", etc.
    for text in (label or "", detail or ""):
        text = text.lower()
        for keywords, icon in _ICON_KEYWORDS:
            if any(re.search(rf"\b{re.escape(k)}", text) for k in keywords):
                return icon
    return None


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "are", "for",
    "with", "this", "that", "it", "as", "by", "be", "at", "from", "you",
    "your", "we", "will", "so", "what", "into", "then", "when",
}


def _keywords(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in _STOPWORDS}


def _assign_step_delays(steps, entries, range_start, duration, min_gap=0.35):
    """
    Map each extracted step to the moment it's actually spoken, instead of
    spacing steps evenly across the segment. Matches each step's label/detail
    keywords against transcript entries (which carry real Whisper timestamps)
    to find when that concept is first named, falling back to even spacing
    for any step that can't be matched. Delays are clamped to be
    non-decreasing so the reveal order still follows the listed step order
    even if a keyword match is noisy.
    """
    n = len(steps)
    if n == 0:
        return []
    interval = duration / (n + 1)
    fallback = [i * interval for i in range(n)]

    matched = [None] * n
    if entries:
        entry_kw = [(e["start"], _keywords(e.get("text"))) for e in entries]
        for i, step in enumerate(steps):
            # Try the label's own keywords first ("Perceive", "Reason", ...)
            # before falling back to the detail sentence. LLM-written detail
            # sentences often restate the segment's subject in every step
            # ("Agent senses...", "Agent decides...", "Agent takes..."), and
            # matching on that shared word would pin every step to whichever
            # sentence happens to name it first.
            for kws in (_keywords(step.get("label")), _keywords(step.get("detail"))):
                if not kws:
                    continue
                found = next((start for start, ekw in entry_kw if kws & ekw), None)
                if found is not None:
                    matched[i] = max(0.0, min(found - range_start, duration))
                    break

    delays = []
    prev = -min_gap
    for i in range(n):
        candidate = matched[i] if matched[i] is not None else fallback[i]
        d = max(candidate, prev + min_gap)
        d = min(d, max(duration - 0.3, 0.0))
        delays.append(d)
        prev = d
    return delays


_ACCENT_PALETTE = [
    ("#2dd4bf", "#123c3a", "#8ff5e6"),  # teal
    ("#60a5fa", "#132a4a", "#bfe0ff"),  # blue
    ("#a78bfa", "#241b47", "#e0d4ff"),  # violet
    ("#f472b6", "#3a1530", "#ffd6ec"),  # pink
    ("#fb923c", "#3a2410", "#ffdcb0"),  # amber
]


def _layout_vertical(steps, delays, duration_sec, width, height, pip_position, accent):
    """Stacked cards down one side, connected by a growing vertical line."""
    accent_color, badge_bg, badge_text = accent
    n = max(len(steps), 1)

    blocks = []
    for i, step in enumerate(steps):
        label = html_lib.escape(step.get("label") or f"Step {i + 1}")
        detail = html_lib.escape(step.get("detail") or "")
        delay = delays[i]
        next_bound = delays[i + 1] if i + 1 < n else duration_sec
        gap = max(next_bound - delay, 0.3)
        reveal_dur = min(0.7, max(gap * 0.6, 0.3))
        connector_dur = max(gap - reveal_dur, 0.25)
        icon = _pick_icon(step.get("label", ""), step.get("detail", ""))
        badge_content = icon if icon else str(i + 1)
        detail_html = f'<div class="detail">{detail}</div>' if detail else ""
        blocks.append(
            f'<div class="step" style="animation-delay:{delay:.2f}s; animation-duration:{reveal_dur:.2f}s;">'
            f'<div class="badge" style="animation-delay:{delay:.2f}s;">{badge_content}</div>'
            f'<div class="text"><div class="label">{label}</div>{detail_html}</div>'
            f'</div>'
        )
        if i < n - 1:
            connector_delay = delay + reveal_dur
            blocks.append(
                f'<div class="connector" style="animation-delay:{connector_delay:.2f}s; animation-duration:{connector_dur:.2f}s;">'
                f'<div class="connector-line"></div></div>'
            )
    steps_html = "\n".join(blocks)

    # Keep clear of whichever corner the webcam bubble docks in.
    pip_position = pip_position or "bottom-right"
    on_left = "left" in pip_position
    stage_justify = "flex-end" if on_left else "flex-start"
    stage_padding = "padding-right: 6%;" if on_left else "padding-left: 6%;"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; width:{width}px; height:{height}px; background:#0b0c0e;
    overflow:hidden; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .stage {{ position:absolute; inset:0; display:flex; align-items:center;
    justify-content:{stage_justify}; {stage_padding} }}
  .stage::before {{ content:""; position:absolute; width:60vh; height:60vh; border-radius:50%;
    background:radial-gradient(circle, {accent_color}1a, transparent 70%);
    animation: drift 12s ease-in-out infinite alternate; }}
  @keyframes drift {{ from {{ transform: translate(-8%, -6%); }} to {{ transform: translate(8%, 6%); }} }}
  .steps {{ position:relative; display:flex; flex-direction:column; align-items:flex-start;
    width:58%; max-width:820px; }}
  .step {{ display:flex; align-items:center; gap:18px; opacity:0; padding:10px 16px;
    border-radius:14px; background:rgba(255,255,255,0.035); border:1px solid {accent_color}20;
    animation-name: reveal; animation-timing-function: ease-out; animation-fill-mode: forwards; }}
  .badge {{ flex:0 0 auto; width:4.4vh; height:4.4vh; border-radius:50%; background:{badge_bg};
    border:2px solid {accent_color}; color:{badge_text}; display:flex; align-items:center; justify-content:center;
    font-size:2.1vh; font-weight:700; animation-name: pulse; animation-duration:1s;
    animation-timing-function: ease-out; }}
  .text .label {{ font-size:2.6vh; font-weight:700; color:#eafffa; line-height:1.25; }}
  .text .detail {{ font-size:1.7vh; color:#8ba7a5; margin-top:2px; }}
  .connector {{ width:4.4vh; display:flex; justify-content:center; height:2.4vh; opacity:0;
    animation-name: connectorFade; animation-timing-function: linear; animation-fill-mode: forwards; }}
  .connector-line {{ width:3px; height:100%; border-radius:2px;
    background: linear-gradient({accent_color}, {accent_color}26);
    transform: scaleY(0); transform-origin: top;
    animation-name: grow; animation-timing-function: ease-out; animation-fill-mode: forwards;
    animation-duration: inherit; animation-delay: inherit; }}
  @keyframes reveal {{
    from {{ opacity:0; transform: translateY(10px); }}
    to   {{ opacity:1; transform: translateY(0); }}
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 {accent_color}a6; }}
    70%  {{ box-shadow: 0 0 0 16px {accent_color}00; }}
    100% {{ box-shadow: 0 0 0 0 {accent_color}00; }}
  }}
  @keyframes connectorFade {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
  @keyframes grow {{ from {{ transform: scaleY(0); }} to {{ transform: scaleY(1); }} }}
</style></head>
<body>
  <div class="stage"><div class="steps">
    {steps_html}
  </div></div>
</body></html>"""


def _layout_horizontal(steps, delays, duration_sec, width, height, pip_position, accent):
    """Left-to-right timeline, connected by growing horizontal bars."""
    accent_color, badge_bg, badge_text = accent
    n = max(len(steps), 1)

    blocks = []
    for i, step in enumerate(steps):
        label = html_lib.escape(step.get("label") or f"Step {i + 1}")
        detail = html_lib.escape(step.get("detail") or "")
        delay = delays[i]
        next_bound = delays[i + 1] if i + 1 < n else duration_sec
        gap = max(next_bound - delay, 0.3)
        reveal_dur = min(0.7, max(gap * 0.6, 0.3))
        connector_dur = max(gap - reveal_dur, 0.25)
        icon = _pick_icon(step.get("label", ""), step.get("detail", ""))
        badge_content = icon if icon else str(i + 1)
        detail_html = f'<div class="detail">{detail}</div>' if detail else ""
        blocks.append(
            f'<div class="tstep" style="animation-delay:{delay:.2f}s; animation-duration:{reveal_dur:.2f}s;">'
            f'<div class="badge" style="animation-delay:{delay:.2f}s;">{badge_content}</div>'
            f'<div class="label">{label}</div>{detail_html}'
            f'</div>'
        )
        if i < n - 1:
            connector_delay = delay + reveal_dur
            blocks.append(
                f'<div class="tconnector" style="animation-delay:{connector_delay:.2f}s; animation-duration:{connector_dur:.2f}s;"></div>'
            )
    steps_html = "\n".join(blocks)

    # Timeline spans most of the width regardless of pip corner, so the
    # defense is vertical: anchor to the half of the screen the bubble isn't in.
    pip_position = pip_position or "bottom-right"
    anchor_top = "bottom" in pip_position
    stage_align = "flex-start" if anchor_top else "flex-end"
    stage_padding = "padding-top: 14%;" if anchor_top else "padding-bottom: 14%;"

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; width:{width}px; height:{height}px; background:#0b0c0e;
    overflow:hidden; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .stage {{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
    justify-content:{stage_align}; {stage_padding} }}
  .stage::before {{ content:""; position:absolute; width:70vh; height:70vh; border-radius:50%;
    background:radial-gradient(circle, {accent_color}14, transparent 70%); top:10%; left:20%;
    animation: drift 14s ease-in-out infinite alternate; }}
  @keyframes drift {{ from {{ transform: translate(-6%, -4%); }} to {{ transform: translate(6%, 4%); }} }}
  .timeline {{ position:relative; display:flex; flex-direction:row; align-items:flex-start;
    width:86%; max-width:1500px; }}
  .tstep {{ flex:1 1 0; display:flex; flex-direction:column; align-items:center; text-align:center;
    opacity:0; padding:0 8px; animation-name: reveal; animation-timing-function: ease-out;
    animation-fill-mode: forwards; }}
  .badge {{ flex:0 0 auto; width:4.6vh; height:4.6vh; border-radius:50%; background:{badge_bg};
    border:2px solid {accent_color}; color:{badge_text}; display:flex; align-items:center; justify-content:center;
    font-size:2.2vh; font-weight:700; animation-name: pulse; animation-duration:1s;
    animation-timing-function: ease-out; }}
  .tstep .label {{ font-size:2.2vh; font-weight:700; color:#eafffa; margin-top:1.2vh; line-height:1.25; }}
  .tstep .detail {{ font-size:1.4vh; color:#8ba7a5; margin-top:0.4vh; max-width:22vh; }}
  .tconnector {{ flex:0 0 auto; width:6vh; height:3px; margin-top:2.3vh; border-radius:2px;
    background: linear-gradient(90deg, {accent_color}, {accent_color}26);
    transform: scaleX(0); transform-origin: left; opacity:0;
    animation-name: growXFade; animation-timing-function: ease-out; animation-fill-mode: forwards; }}
  @keyframes reveal {{
    from {{ opacity:0; transform: translateY(10px); }}
    to   {{ opacity:1; transform: translateY(0); }}
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 {accent_color}a6; }}
    70%  {{ box-shadow: 0 0 0 16px {accent_color}00; }}
    100% {{ box-shadow: 0 0 0 0 {accent_color}00; }}
  }}
  @keyframes growXFade {{
    from {{ opacity:0; transform: scaleX(0); }}
    to   {{ opacity:1; transform: scaleX(1); }}
  }}
</style></head>
<body>
  <div class="stage"><div class="timeline">
    {steps_html}
  </div></div>
</body></html>"""


def _build_diagram_html(steps, delays, duration_sec, width, height, pip_position):
    accent = random.choice(_ACCENT_PALETTE)
    layout = random.choice([_layout_vertical, _layout_horizontal])
    return layout(steps, delays, duration_sec, width, height, pip_position, accent)


def generate_diagram_html(prompt_text, duration_sec, width, height, work_dir,
                           transcript_entries=None, range_start=0.0, pip_position="bottom-right",
                           backend=DEFAULT_BACKEND, model=None,
                           ollama_server=DEFAULT_OLLAMA_SERVER, timeout=180):
    """Extract ordered steps from the transcript via an LLM, time each step's
    reveal to when it's actually spoken (falling back to even spacing for any
    step that can't be matched), render into an animated HTML diagram, save
    it, return its path."""
    transcript_text = " ".join(e["text"].strip() for e in (transcript_entries or []))
    steps = extract_steps(
        prompt_text, transcript_text, backend=backend, model=model,
        ollama_server=ollama_server, timeout=timeout,
    )
    delays = _assign_step_delays(steps, transcript_entries or [], range_start, duration_sec)
    html_doc = _build_diagram_html(steps, delays, duration_sec, width, height, pip_position)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / "diagram.html"
    html_path.write_text(html_doc)
    return html_path


def render_html_to_video(html_path, work_dir, duration_sec, width, height):
    """Play the HTML in a headless browser and record it to an mp4 of exactly duration_sec."""
    from playwright.sync_api import sync_playwright

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": width, "height": height},
            record_video_dir=str(work_dir),
            record_video_size={"width": width, "height": height},
        )
        page = context.new_page()
        page.goto(Path(html_path).resolve().as_uri())
        page.wait_for_timeout(int(duration_sec * 1000) + 300)
        context.close()
        browser.close()

    webm_files = sorted(work_dir.glob("*.webm"), key=lambda p: p.stat().st_mtime)
    if not webm_files:
        raise RuntimeError(f"Playwright did not produce a recording in {work_dir}")
    webm_path = webm_files[-1]

    mp4_path = work_dir / "graphic_video.mp4"
    audio_tools.run([
        "ffmpeg", "-y", "-i", str(webm_path), "-t", f"{duration_sec:.3f}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(mp4_path)
    ])
    return mp4_path


def generate_motion_graphic(prompt_text, work_dir, duration_sec, width, height,
                             transcript_entries=None, range_start=0.0, pip_position="bottom-right",
                             backend=DEFAULT_BACKEND, model=None,
                             ollama_server=DEFAULT_OLLAMA_SERVER, timeout=180):
    """Extract steps from the transcript, render the timed diagram, record it. Returns an mp4 path."""
    html_path = generate_diagram_html(
        prompt_text, duration_sec, width, height, work_dir,
        transcript_entries=transcript_entries, range_start=range_start, pip_position=pip_position,
        backend=backend, model=model, ollama_server=ollama_server, timeout=timeout,
    )
    return render_html_to_video(html_path, work_dir, duration_sec, width, height)
