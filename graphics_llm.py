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
  llamacpp (default) — free, fully local. Needs a `llama-server` instance
                        already running (llama.cpp's OpenAI-compatible HTTP
                        server); talks to its `/v1/chat/completions` endpoint.
  anthropic           — better extraction quality, costs money. Needs
                        ANTHROPIC_API_KEY in .env or the environment.
"""

import html as html_lib
import json
import os
import random
import re
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND = "llamacpp"
# LLAMACPP_SERVER in .env is where llama-server's address is configured.
DEFAULT_LLAMACPP_SERVER = os.getenv("LLAMACPP_SERVER", "http://127.0.0.1:8480")
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"

# STEPS_SYSTEM_PROMPT = (
#     "You extract the distinct steps, concepts, or entities that the narrator "
#     "explicitly names in a short transcript segment from an educational/tech "
#     "video, preserving the exact order of first mention.\n"
#     "\n"
#     "OUTPUT — return STRICT JSON only: an array of 2-6 objects:\n"
#     '[{"label": "...", "detail": "..."}]\n'
#     "- label: 2-4 words, Title Case, a noun phrase (no verbs like 'Understanding X')\n"
#     "- detail: one sentence, under 12 words, plain language, no jargon the "
#     "narrator didn't use\n"
#     "\n"
#     "RULES:\n"
#     "- Only include items the narrator actually names — never infer or invent steps.\n"
#     "- Merge restatements of the same concept into one item.\n"
#     "- Skip filler, greetings, and meta-talk ('in this video we will...').\n"
#     "- Deduplicate: if a concept repeats, keep only its first mention.\n"
#     "- If the segment names no distinct steps, return a single-item array "
#     "summarizing the topic: "
#     '[{"label": "<topic>", "detail": "<one-line summary>"}]\n'
#     "- If the segment is empty or unintelligible, return "
#     '[{"label": "Unclear Segment", "detail": "No identifiable content."}]\n'
#     "\n"
#     "No markdown fences, no keys other than label and detail, no prose "
#     "before or after the JSON."
# )

# STEPS_SYSTEM_PROMPT = """You convert a timestamped transcript segment from an educational/tech video into a scene plan for a synced motion graphic, where each item becomes a labeled diagram node that appears exactly when the narrator first mentions it.

# INPUT: transcript lines in the form "[MM:SS] text".

# OUTPUT — STRICT JSON only, an array of 2-6 objects:
# [{"label": "...", "detail": "...", "t": <seconds>, "role": "...", "link": <index|null>}]

# - label: 2-4 word noun phrase, Title Case — rendered as the node title
# - detail: under 8 words, no ending period — rendered as the node subtitle
# - t: seconds (from the timestamps) of the concept's FIRST mention — the node's entrance time
# - role: one of
#     "step"    — part of a sequence/pipeline
#     "concept" — a definition or idea
#     "actor"   — a component/agent/person that does something
#     "outcome" — a result or conclusion
#   (the renderer maps role to a visual variant)
# - link: index of the earlier item this one connects FROM (draws an arrow), or null if standalone

# RULES:
# - Extract only what the narrator explicitly names; never invent or infer.
# - Order by t ascending; merge restatements, keep first mention only.
# - Skip greetings, filler, and meta-talk ("in this video...").
# - Two nodes should never share the same t; if mentioned in the same breath, offset the second by +1.
# - If no distinct items exist, return one item: the segment's topic as a "concept" with t of the segment start and link null.

# No markdown fences, no extra keys, no prose outside the JSON."""    
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


def _steps_user_message(prompt_text, transcript_text, feedback=None):
    msg = (
        f"Topic label: {prompt_text}\n"
        f"Transcript (in spoken order): "
        f"{transcript_text.strip() if transcript_text else '(not available — use the topic label only)'}"
    )
    if feedback:
        msg += (
            "\n\nYour previous attempt was rejected for: "
            f"{'; '.join(feedback)}. Return corrected JSON that fixes this."
        )
    return msg


def _extract_steps_via_anthropic(prompt_text, transcript_text, model, feedback=None):
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or DEFAULT_ANTHROPIC_MODEL,
        max_tokens=1024,
        system=STEPS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _steps_user_message(prompt_text, transcript_text, feedback)}],
    )
    return next(b.text for b in response.content if b.type == "text")


def _extract_steps_via_llamacpp(prompt_text, transcript_text, model, server, timeout, feedback=None):
    """llama.cpp's `llama-server` speaks an OpenAI-compatible chat API:
    POST /v1/chat/completions, response text at choices[0].message.content."""
    resp = requests.post(
        f"{server.rstrip('/')}/v1/chat/completions",
        json={
            "model": model or "local",
            "messages": [
                {"role": "system", "content": STEPS_SYSTEM_PROMPT},
                {"role": "user", "content": _steps_user_message(prompt_text, transcript_text, feedback)},
            ],
            "temperature": 0.4,
            "max_tokens": 800,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


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


def _verify_steps(steps):
    """Judge whether extract_steps' output is actually usable, so a bad
    generation can be fed back to the model instead of silently reaching
    the renderer. Returns a list of problem strings (empty = passes)."""
    if not steps:
        return ["no steps were extracted"]
    issues = []
    if not (2 <= len(steps) <= 6):
        issues.append(f"expected 2-6 steps, got {len(steps)}")
    if any(not s.get("label", "").strip() for s in steps):
        issues.append("a step is missing a label")
    labels = [s.get("label", "").strip().lower() for s in steps]
    if len(labels) != len(set(labels)):
        issues.append("duplicate/near-duplicate labels — merge restated concepts")
    return issues


def extract_steps(prompt_text, transcript_text, backend=DEFAULT_BACKEND, model=None,
                   llamacpp_server=DEFAULT_LLAMACPP_SERVER, timeout=180, max_attempts=2):
    """
    Ask an LLM for the ordered list of steps/concepts named in
    transcript_text. A hosted backend (anthropic) that fails after
    its own retries — exhausted quota, outage, whatever — falls back to the
    local llamacpp backend for just this segment rather than aborting the
    whole multi-range pipeline run over one flaky remote call.

    Agentic loop: each attempt's output is verified (_verify_steps) rather
    than trusted blindly; a failing attempt's problems are folded into the
    next attempt's prompt as corrective feedback, up to max_attempts, before
    falling back to a single-item summary.
    """
    if backend not in ("llamacpp", "anthropic"):
        raise ValueError(f"Unknown graphics backend '{backend}', expected 'llamacpp' or 'anthropic'")

    steps, feedback = [], None
    for attempt in range(1, max_attempts + 1):
        try:
            if backend == "llamacpp":
                raw_text = _extract_steps_via_llamacpp(
                    prompt_text, transcript_text, model, llamacpp_server, timeout, feedback=feedback)
            else:
                raw_text = _extract_steps_via_anthropic(prompt_text, transcript_text, model, feedback=feedback)
        except Exception as exc:
            if backend == "llamacpp":
                raise
            print(f"[graphics_llm] {backend} step extraction failed ({exc}); falling back to llamacpp for this segment.")
            raw_text = _extract_steps_via_llamacpp(prompt_text, transcript_text, None, llamacpp_server, timeout)
            steps = _parse_steps_json(raw_text)
            break

        steps = _parse_steps_json(raw_text)
        issues = _verify_steps(steps)
        if not issues:
            break
        feedback = issues
        if attempt < max_attempts:
            print(f"[graphics_llm] step extraction attempt {attempt} rejected ({'; '.join(issues)}); retrying with feedback.")
        else:
            print(f"[graphics_llm] step extraction still failing after {max_attempts} attempts ({'; '.join(issues)}); using it as-is.")

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
    keywords against transcript timestamps to find when that concept is
    first named, falling back to even spacing for any step that can't be
    matched. Delays are clamped to be non-decreasing so the reveal order
    still follows the listed step order even if a keyword match is noisy.
    """
    n = len(steps)
    if n == 0:
        return []
    interval = duration / (n + 1)
    fallback = [i * interval for i in range(n)]

    matched = [None] * n
    # Word-level timestamps (from transcribe.py's word_timestamps=True) let
    # each step match the exact moment it's spoken. Whisper often batches
    # several named concepts into one multi-second segment ("first perceive,
    # then reason, then act, then observe" as a single ~10s entry) -- with
    # only segment-level timestamps every step in that segment would resolve
    # to the same start time and get crammed into ~1 second while the
    # narrator is still talking for seconds afterward. Falls back to
    # segment-level for transcripts made before this existed.
    word_kw = [
        (w["start"], _keywords(w.get("word")))
        for e in entries for w in e.get("words", [])
    ]
    timed_kw = word_kw if word_kw else [(e["start"], _keywords(e.get("text"))) for e in entries]

    if timed_kw:
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
                found = next((start for start, wkw in timed_kw if kws & wkw), None)
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
  .steps {{ position:relative; display:flex; flex-direction:column; align-items:flex-start;
    width:58%; max-width:820px; }}
  .step {{ position:relative; overflow:hidden; display:flex; align-items:center; gap:18px;
    opacity:0; padding:1.6vh 2.4vh; border-radius:16px; width:100%; box-sizing:border-box;
    background: linear-gradient(135deg, {accent_color}, {badge_bg});
    box-shadow: 0 10px 30px -8px {accent_color}70, inset 0 1px 0 rgba(255,255,255,0.28);
    animation-name: reveal; animation-timing-function: ease-out; animation-fill-mode: forwards; }}
  .badge {{ flex:0 0 auto; min-width:3.6vh; text-align:center; color:#fff;
    font-size:3vh; font-weight:800; text-shadow: 0 2px 6px rgba(0,0,0,0.35);
    animation-name: pulse; animation-duration:1s; animation-timing-function: ease-out; }}
  .text .label {{ font-size:2.5vh; font-weight:700; color:#fff; line-height:1.25; }}
  .text .detail {{ font-size:1.6vh; color:{badge_text}; opacity:0.9; margin-top:2px; }}
  .connector {{ width:4.4vh; display:flex; justify-content:center; height:2.2vh; opacity:0;
    animation-name: connectorFade; animation-timing-function: linear; animation-fill-mode: forwards; }}
  .connector-line {{ width:3px; height:100%; border-radius:2px;
    background: linear-gradient({accent_color}, {accent_color}26);
    transform: scaleY(0); transform-origin: top;
    animation-name: grow; animation-timing-function: ease-out; animation-fill-mode: forwards;
    animation-duration: inherit; animation-delay: inherit; }}
  @keyframes reveal {{
    from {{ opacity:0; transform: translateY(14px) scale(0.97); }}
    to   {{ opacity:1; transform: translateY(0) scale(1); }}
  }}
  @keyframes pulse {{
    0%   {{ text-shadow: 0 0 0 {accent_color}00, 0 2px 6px rgba(0,0,0,0.35); }}
    40%  {{ text-shadow: 0 0 18px {accent_color}cc, 0 2px 6px rgba(0,0,0,0.35); }}
    100% {{ text-shadow: 0 2px 6px rgba(0,0,0,0.35); }}
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
    """Left-to-right row of gradient cards, connected by growing bars."""
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
            f'<div class="card"><div class="badge" style="animation-delay:{delay:.2f}s;">{badge_content}</div>'
            f'<div class="label">{label}</div>{detail_html}</div>'
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
  .timeline {{ position:relative; display:flex; flex-direction:row; align-items:flex-start;
    width:86%; max-width:1500px; }}
  .tstep {{ flex:1 1 0; display:flex; flex-direction:column; align-items:center; text-align:center;
    opacity:0; padding:0 8px; animation-name: reveal; animation-timing-function: ease-out;
    animation-fill-mode: forwards; }}
  .card {{ position:relative; overflow:hidden; width:100%; box-sizing:border-box;
    padding:1.8vh 1.4vh 1.4vh; border-radius:16px;
    background: linear-gradient(160deg, {accent_color}, {badge_bg});
    box-shadow: 0 10px 26px -8px {accent_color}70, inset 0 1px 0 rgba(255,255,255,0.28); }}
  .badge {{ color:#fff; font-size:2.6vh; font-weight:800; text-shadow: 0 2px 6px rgba(0,0,0,0.35);
    animation-name: pulse; animation-duration:1s; animation-timing-function: ease-out; }}
  .tstep .label {{ font-size:2vh; font-weight:700; color:#fff; margin-top:1vh; line-height:1.25; }}
  .tstep .detail {{ font-size:1.35vh; color:{badge_text}; opacity:0.9; margin-top:0.5vh; max-width:22vh; }}
  .tconnector {{ flex:0 0 auto; width:6vh; height:3px; margin-top:4.6vh; border-radius:2px;
    background: linear-gradient(90deg, {accent_color}, {accent_color}26);
    transform: scaleX(0); transform-origin: left; opacity:0;
    animation-name: growXFade; animation-timing-function: ease-out; animation-fill-mode: forwards; }}
  @keyframes reveal {{
    from {{ opacity:0; transform: translateY(14px) scale(0.97); }}
    to   {{ opacity:1; transform: translateY(0) scale(1); }}
  }}
  @keyframes pulse {{
    0%   {{ text-shadow: 0 0 0 {accent_color}00, 0 2px 6px rgba(0,0,0,0.35); }}
    40%  {{ text-shadow: 0 0 18px {accent_color}cc, 0 2px 6px rgba(0,0,0,0.35); }}
    100% {{ text-shadow: 0 2px 6px rgba(0,0,0,0.35); }}
  }}
  @keyframes growXFade {{
    from {{ opacity:0; transform: scaleX(0); }}
    to   {{ opacity:1; transform: scaleX(1); }}
  }}
</style></head>
<body>
  <div class="stage">
    <div class="timeline">
    {steps_html}
    </div>
  </div>
</body></html>"""


def _layout_callout(steps, delays, duration_sec, width, height, pip_position, accent):
    """
    One big card at a time, center stage — replaced by the next as the
    narrator moves on, instead of accumulating like the other two layouts.
    A dot row along the bottom tracks progress. Deliberately different
    rhythm (single focal point, not a growing list) so a video with many
    graphic segments doesn't just alternate between the same two shapes.
    """
    accent_color, badge_bg, badge_text = accent
    n = max(len(steps), 1)

    cards, dots = [], []
    for i, step in enumerate(steps):
        label = html_lib.escape(step.get("label") or f"Step {i + 1}")
        detail = html_lib.escape(step.get("detail") or "")
        delay = delays[i]
        next_bound = delays[i + 1] if i + 1 < n else duration_sec
        gap = max(next_bound - delay, 0.4)
        is_last = i == n - 1
        icon = _pick_icon(step.get("label", ""), step.get("detail", ""))
        badge_content = icon if icon else str(i + 1)
        detail_html = f'<div class="detail">{detail}</div>' if detail else ""
        anim = "calloutStay" if is_last else "calloutHold"
        cards.append(
            f'<div class="callout" style="animation-name:{anim}; animation-delay:{delay:.2f}s; '
            f'animation-duration:{gap:.2f}s;">'
            f'<div class="badge">{badge_content}</div>'
            f'<div class="label">{label}</div>{detail_html}'
            f'</div>'
        )
        dots.append(f'<div class="dot" style="animation-delay:{delay:.2f}s;"></div>')
    cards_html = "\n".join(cards)
    dots_html = "\n".join(dots)

    # Centered content is wide enough to reach into whichever corner the pip
    # bubble docks in — nudge the whole stage toward the opposite edge.
    pip_position = pip_position or "bottom-right"
    if "bottom" in pip_position:
        stage_align, stage_padding = "flex-start", "padding-top: 6%;"
    elif "top" in pip_position:
        stage_align, stage_padding = "flex-end", "padding-bottom: 6%;"
    else:
        stage_align, stage_padding = "center", ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin:0; padding:0; width:{width}px; height:{height}px; background:#0b0c0e;
    overflow:hidden; font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
  .stage {{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center;
    justify-content:{stage_align}; {stage_padding} }}
  .callout {{ position:absolute; display:flex; flex-direction:column; align-items:center; text-align:center;
    width:52%; max-width:820px; padding:4vh 4vh; border-radius:22px; box-sizing:border-box;
    background: linear-gradient(150deg, {accent_color}, {badge_bg});
    box-shadow: 0 20px 50px -12px {accent_color}80, inset 0 1px 0 rgba(255,255,255,0.28);
    opacity:0; animation-timing-function: ease-in-out; animation-fill-mode: both; }}
  .badge {{ font-size:5vh; font-weight:800; color:#fff; text-shadow: 0 2px 8px rgba(0,0,0,0.35);
    margin-bottom:1.4vh; }}
  .label {{ font-size:3.4vh; font-weight:800; color:#fff; line-height:1.2; }}
  .detail {{ font-size:1.9vh; color:{badge_text}; opacity:0.9; margin-top:1.2vh; }}
  .dots {{ position:absolute; bottom:8%; left:50%; transform:translateX(-50%);
    display:flex; gap:10px; }}
  .dot {{ width:1.3vh; height:1.3vh; border-radius:50%; background:{accent_color}33;
    animation-name: dotLight; animation-duration:0.5s; animation-timing-function: ease-out;
    animation-fill-mode: forwards; }}
  @keyframes calloutHold {{
    0%   {{ opacity:0; transform: translateY(18px) scale(0.94); }}
    10%  {{ opacity:1; transform: translateY(0) scale(1); }}
    88%  {{ opacity:1; transform: translateY(0) scale(1); }}
    100% {{ opacity:0; transform: translateY(-14px) scale(0.96); }}
  }}
  @keyframes calloutStay {{
    0%   {{ opacity:0; transform: translateY(18px) scale(0.94); }}
    12%  {{ opacity:1; transform: translateY(0) scale(1); }}
    100% {{ opacity:1; transform: translateY(0) scale(1); }}
  }}
  @keyframes dotLight {{
    from {{ background:{accent_color}33; transform:scale(1); }}
    50%  {{ background:{accent_color}; transform:scale(1.5); }}
    to   {{ background:{accent_color}; transform:scale(1); }}
  }}
</style></head>
<body>
  <div class="stage">
    {cards_html}
    <div class="dots">{dots_html}</div>
  </div>
</body></html>"""


def _build_diagram_html(steps, delays, duration_sec, width, height, pip_position):
    accent = random.choice(_ACCENT_PALETTE)
    layout = random.choice([_layout_vertical, _layout_horizontal, _layout_callout])
    return layout(steps, delays, duration_sec, width, height, pip_position, accent)


def generate_diagram_html(prompt_text, duration_sec, width, height, work_dir,
                           transcript_entries=None, range_start=0.0, pip_position="bottom-right",
                           backend=DEFAULT_BACKEND, model=None,
                           llamacpp_server=DEFAULT_LLAMACPP_SERVER, timeout=180):
    """Extract ordered steps from the transcript via an LLM, time each step's
    reveal to when it's actually spoken (falling back to even spacing for any
    step that can't be matched), render into an animated HTML diagram, save
    it, return its path."""
    transcript_text = " ".join(e["text"].strip() for e in (transcript_entries or []))
    steps = extract_steps(
        prompt_text, transcript_text, backend=backend, model=model,
        llamacpp_server=llamacpp_server, timeout=timeout,
    )
    delays = _assign_step_delays(steps, transcript_entries or [], range_start, duration_sec)
    html_doc = _build_diagram_html(steps, delays, duration_sec, width, height, pip_position)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    html_path = work_dir / "diagram.html"
    html_path.write_text(html_doc)
    return html_path


def render_html_to_video(html_path, work_dir, duration_sec, width, height, fps=30):
    """
    Play the HTML in a headless browser and record it to an mp4 of exactly
    duration_sec.

    Deliberately NOT using Playwright's built-in record_video_dir: that
    recorder is a screencast meant for test-debugging artifacts, not clean
    output — it samples frames on a loose real-time schedule (dropping/
    duplicating whenever the page or CPU stalls) and encodes to low-bitrate
    VP8, which is where the "motion graphic looks low quality/choppy"
    complaint traces back to, not the CSS itself.

    Instead: pause every CSS animation on the page via the Web Animations
    API (document.getAnimations()), then step each one to an exact
    `currentTime` and screenshot the page once per output frame. This makes
    frame timing deterministic (independent of render speed/CPU load) and
    each frame a full-resolution PNG, piped straight into a single
    high-CRF x264 encode — no lossy intermediate container.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = work_dir / "graphic_video.mp4"
    n_frames = max(round(duration_sec * fps), 1)

    from playwright.sync_api import sync_playwright

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p",
            str(mp4_path),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(Path(html_path).resolve().as_uri())
        page.evaluate("document.getAnimations().forEach(a => a.pause())")
        for i in range(n_frames):
            t_ms = i * (1000 / fps)
            page.evaluate(f"document.getAnimations().forEach(a => a.currentTime = {t_ms})")
            # pipeline.py runs up to 4 graphic ranges concurrently, each
            # driving its own Chromium through this same per-frame
            # screenshot loop -- under combined load a single call can
            # occasionally stall past Playwright's default 30s timeout even
            # though nothing is actually broken. A generous timeout plus one
            # retry rides out that contention instead of killing the whole
            # range over one slow frame.
            try:
                png_bytes = page.screenshot(type="png", timeout=90000)
            except PlaywrightTimeoutError:
                png_bytes = page.screenshot(type="png", timeout=90000)
            encoder.stdin.write(png_bytes)
        browser.close()

    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg frame encode failed: {encoder.stderr.read().decode(errors='replace')}")
    return mp4_path


def _verify_render(mp4_path, expected_duration, duration_tolerance=1.0):
    """
    Sanity-check the actual rendered clip rather than trusting that a clean
    return from render_html_to_video means a usable file — the per-frame
    screenshot/ffmpeg-pipe loop can die partway through under the
    concurrent-Chromium load pipeline.py puts it under (see the timeout
    handling in render_html_to_video), leaving a short/truncated mp4 that
    would otherwise only surface later as a broken concat in pipeline.py's
    final render. Returns a list of problem strings (empty = passes).
    """
    mp4_path = Path(mp4_path)
    if not mp4_path.exists() or mp4_path.stat().st_size < 1024:
        return ["output file missing or too small"]
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        actual_duration = float(probe.stdout.strip())
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        return [f"ffprobe could not read the render ({exc})"]

    if abs(actual_duration - expected_duration) > duration_tolerance:
        return [f"duration {actual_duration:.2f}s vs expected {expected_duration:.2f}s"]
    return []


def generate_motion_graphic(prompt_text, work_dir, duration_sec, width, height,
                             transcript_entries=None, range_start=0.0, pip_position="bottom-right",
                             backend=DEFAULT_BACKEND, model=None,
                             llamacpp_server=DEFAULT_LLAMACPP_SERVER, timeout=180, fps=30,
                             max_attempts=2):
    """
    Full agentic life cycle for one graphic: extract steps -> build the
    timed HTML diagram -> render it to video -> verify the actual output
    (_verify_render) -> if it fails, feed that back into another full attempt
    (a fresh extraction may pick a different random layout/accent and sidestep
    a one-off render glitch) instead of silently handing pipeline.py a broken
    clip. Keeps the last attempt's render if none pass, since a suspect clip
    still beats aborting the whole multi-range pipeline run.
    """
    last_mp4 = None
    for attempt in range(1, max_attempts + 1):
        html_path = generate_diagram_html(
            prompt_text, duration_sec, width, height, work_dir,
            transcript_entries=transcript_entries, range_start=range_start, pip_position=pip_position,
            backend=backend, model=model, llamacpp_server=llamacpp_server, timeout=timeout,
        )
        mp4_path = render_html_to_video(html_path, work_dir, duration_sec, width, height, fps=fps)

        issues = _verify_render(mp4_path, duration_sec)
        if not issues:
            return mp4_path
        last_mp4 = mp4_path
        if attempt < max_attempts:
            print(f"[graphics_llm] render attempt {attempt} rejected ({'; '.join(issues)}); regenerating.")
        else:
            print(f"[graphics_llm] render still suspect after {max_attempts} attempts ({'; '.join(issues)}); keeping it.")

    return last_mp4
