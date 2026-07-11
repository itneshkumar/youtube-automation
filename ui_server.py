#!/usr/bin/env python3
"""
ui_server.py — tiny local web UI for configuring and running start.sh.

    python3 ui_server.py

Opens http://127.0.0.1:8787 (auto-opens your browser). Loopback-only —
nothing is exposed outside your machine. Edit settings, hit Save, or hit
Run to save + kick off start.sh and watch its output live in the page.
"""

import json
import re
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.yaml"
CONFIG_EXAMPLE_PATH = PROJECT_DIR / "config.example.yaml"
LOG_PATH = PROJECT_DIR / "work" / "ui_run.log"
PORT = 8787

# (form field name, path into the config dict, type)
FIELD_SPECS = [
    ("input_video", ("input_video",), str),
    ("work_dir", ("work_dir",), str),
    ("output_video", ("output_video",), str),
    ("silence_noise_floor_db", ("silence", "noise_floor_db"), float),
    ("silence_min_silence_sec", ("silence", "min_silence_sec"), float),
    ("silence_keep_padding_sec", ("silence", "keep_padding_sec"), float),
    ("noise_reduction_enabled", ("noise_reduction", "enabled"), bool),
    ("noise_reduction_profile_start_sec", ("noise_reduction", "profile_start_sec"), float),
    ("noise_reduction_profile_end_sec", ("noise_reduction", "profile_end_sec"), float),
    ("audio_enhance_highpass_hz", ("audio_enhance", "highpass_hz"), int),
    ("audio_enhance_presence_boost_db", ("audio_enhance", "presence_boost_db"), float),
    ("audio_enhance_target_lufs", ("audio_enhance", "target_lufs"), float),
    ("audio_enhance_limiter_ceiling_db", ("audio_enhance", "limiter_ceiling_db"), float),
    ("render_width", ("render", "width"), int),
    ("render_height", ("render", "height"), int),
    ("render_fps", ("render", "fps"), int),
    ("render_crf", ("render", "crf"), int),
    ("pip_width_pct", ("pip", "width_pct"), float),
    ("pip_position", ("pip", "position"), str),
    ("pip_margin_px", ("pip", "margin_px"), int),
    ("pip_border_width_px", ("pip", "border_width_px"), int),
    ("graphics_backend", ("graphics", "backend"), str),
    ("graphics_model", ("graphics", "model"), str),
    ("graphics_ollama_server", ("graphics", "ollama_server"), str),
    ("graphics_timeout_sec", ("graphics", "timeout_sec"), int),
    ("background_enabled", ("background", "enabled"), bool),
    ("background_accent", ("background", "accent"), str),
    ("background_feather_px", ("background", "feather_px"), int),
    ("background_rim_glow_px", ("background", "rim_glow_px"), int),
]

RUN_LOCK = threading.Lock()
RUN_STATE = {"proc": None}

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"}


def list_dir(path):
    entries = []
    try:
        children = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        children = []
    for p in children:
        if p.name.startswith("."):
            continue
        is_dir = p.is_dir()
        if is_dir or p.suffix.lower() in VIDEO_EXTS:
            entries.append({"name": p.name, "path": str(p), "is_dir": is_dir})
    return entries


def load_config():
    path = CONFIG_PATH if CONFIG_PATH.exists() else CONFIG_EXAMPLE_PATH
    return yaml.safe_load(path.read_text()) or {}


def get_nested(d, path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def set_nested(d, path, value):
    for key in path[:-1]:
        d = d.setdefault(key, {})
    d[path[-1]] = value


def apply_fields(cfg, fields):
    for name, path, kind in FIELD_SPECS:
        if kind is bool:
            set_nested(cfg, path, bool(fields.get(name)))
            continue
        if name not in fields or fields[name] in (None, ""):
            continue
        raw = fields[name]
        try:
            value = kind(raw)
        except (TypeError, ValueError):
            continue
        set_nested(cfg, path, value)
    return cfg


def save_config(fields):
    cfg = load_config()
    cfg = apply_fields(cfg, fields)
    cfg.setdefault("segments", [])
    CONFIG_PATH.write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


def find_external_pids():
    """PIDs of start.sh runs launched outside this server process (e.g. a
    previous UI session, or the user running start.sh by hand)."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", str(PROJECT_DIR / "start.sh")],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return []
    return [int(p) for p in out.stdout.split() if p.strip()]


def external_run_active():
    return len(find_external_pids()) > 0


def start_run(fields):
    with RUN_LOCK:
        proc = RUN_STATE.get("proc")
        if proc is not None and proc.poll() is None:
            return False, "A run is already in progress."
        if external_run_active():
            return False, (
                "A start.sh run is already in progress (started outside this "
                "browser session). Wait for it to finish, or click Stop."
            )

        save_config(fields)
        input_video = fields.get("input_video", "").strip()
        if not input_video:
            return False, "input_video is required."

        args = ["bash", str(PROJECT_DIR / "start.sh"), input_video]
        if fields.get("use_llm"):
            args.append("--llm")
        if fields.get("replan"):
            args.append("--replan")
        if fields.get("force_transcribe"):
            args.append("--force-transcribe")
        whisper_model = fields.get("whisper_model") or "small"
        args += ["--whisper-model", whisper_model]

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_PATH, "w")
        log_file.write(f"$ {' '.join(args)}\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            args, cwd=PROJECT_DIR, stdout=log_file, stderr=subprocess.STDOUT,
        )
        RUN_STATE["proc"] = proc
        RUN_STATE["log_file"] = log_file
        return True, "Started."


def stop_run():
    with RUN_LOCK:
        proc = RUN_STATE.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True, "Stopping..."

    pids = find_external_pids()
    if pids:
        for pid in pids:
            subprocess.run(["pkill", "-P", str(pid)])  # kill start.sh's children first
            subprocess.run(["kill", str(pid)])
        return True, "Stopping external run..."
    return False, "Nothing is running."


def run_status():
    with RUN_LOCK:
        proc = RUN_STATE.get("proc")
        if proc is not None:
            rc = proc.poll()
            if rc is None:
                return "running", None
            return "finished", rc
    if external_run_active():
        return "running", None
    return "idle", None


def render_form():
    cfg = load_config()

    def v(path, default=""):
        val = get_nested(cfg, path)
        return default if val is None else val

    def checked(path):
        return "checked" if get_nested(cfg, path) else ""

    def selected(path, option):
        return "selected" if str(get_nested(cfg, path)) == option else ""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Auto-Edit Pipeline</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    max-width: 900px; margin: 2rem auto; padding: 0 1.5rem;
    background: light-dark(#f7f7f8, #16171a); color: light-dark(#1c1c1f, #e7e7ea);
  }}
  h1 {{ font-size: 1.4rem; margin-bottom: .25rem; }}
  .sub {{ color: light-dark(#666, #999); margin-top: 0; margin-bottom: 1.5rem; font-size: .9rem; }}
  fieldset {{
    border: 1px solid light-dark(#ddd, #333); border-radius: 10px;
    margin-bottom: 1rem; padding: .75rem 1rem 1rem;
  }}
  legend {{ font-weight: 600; padding: 0 .4rem; font-size: .85rem; text-transform: uppercase; letter-spacing: .03em; color: light-dark(#555, #aaa); }}
  .row {{ display: flex; flex-wrap: wrap; gap: .75rem 1.25rem; }}
  .field {{ display: flex; flex-direction: column; gap: .25rem; flex: 1 1 160px; }}
  .field.wide {{ flex-basis: 100%; }}
  label {{ font-size: .8rem; color: light-dark(#555, #aaa); }}
  input, select {{
    padding: .4rem .5rem; border-radius: 6px; border: 1px solid light-dark(#ccc, #444);
    background: light-dark(#fff, #202124); color: inherit; font-size: .9rem;
  }}
  input[type=checkbox] {{ width: 1rem; height: 1rem; align-self: flex-start; }}
  .checkline {{ flex-direction: row; align-items: center; gap: .5rem; }}
  .actions {{ display: flex; gap: .6rem; margin: 1.25rem 0; }}
  button {{
    padding: .55rem 1.1rem; border-radius: 8px; border: none; cursor: pointer;
    font-size: .9rem; font-weight: 600;
  }}
  #run-btn {{ background: #2563eb; color: white; }}
  #save-btn {{ background: light-dark(#e5e5e7, #2a2b2f); color: inherit; }}
  #stop-btn {{ background: #dc2626; color: white; display: none; }}
  #status {{ font-size: .85rem; align-self: center; color: light-dark(#666, #999); }}
  #log {{
    background: #0b0c0e; color: #d8d8dc; padding: .9rem; border-radius: 10px;
    height: 320px; overflow-y: auto; white-space: pre-wrap; font: 12px/1.5 ui-monospace, monospace;
    display: none;
  }}
  .toast {{
    position: fixed; bottom: 1.25rem; right: 1.25rem; background: #16171a; color: #fff;
    padding: .6rem 1rem; border-radius: 8px; font-size: .85rem; opacity: 0; transition: opacity .2s;
  }}
  .toast.show {{ opacity: 1; }}
  .modal-overlay {{
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,.5);
    align-items: center; justify-content: center; z-index: 10;
  }}
  .modal-box {{
    background: light-dark(#fff, #202124); border-radius: 12px; width: min(560px, 90vw);
    max-height: 80vh; display: flex; flex-direction: column; overflow: hidden;
    box-shadow: 0 10px 40px rgba(0,0,0,.3);
  }}
  .modal-header {{ padding: .9rem 1rem .5rem; }}
  .modal-header h2 {{ margin: 0 0 .4rem; font-size: 1rem; }}
  #browse-path {{ font-size: .78rem; color: light-dark(#666, #999); word-break: break-all; }}
  .browse-toolbar {{ display: flex; gap: .4rem; padding: 0 1rem .6rem; }}
  .browse-toolbar button {{
    font-size: .78rem; padding: .3rem .6rem; background: light-dark(#eee, #2a2b2f); color: inherit;
  }}
  #browse-list {{ overflow-y: auto; border-top: 1px solid light-dark(#eee, #2a2b2f); flex: 1; }}
  .browse-item {{ padding: .5rem 1rem; cursor: pointer; font-size: .88rem; }}
  .browse-item:hover {{ background: light-dark(#f0f0f2, #2a2b2f); }}
  .browse-item.selected {{ background: #2563eb; color: #fff; }}
  .modal-footer {{ display: flex; justify-content: flex-end; gap: .5rem; padding: .75rem 1rem; border-top: 1px solid light-dark(#eee, #2a2b2f); }}
</style>
</head>
<body>
<h1>Auto-Edit Pipeline</h1>
<p class="sub">Configure config.yaml and run start.sh — output streams below.</p>

<form id="cfg-form">
  <fieldset>
    <legend>Input / Output</legend>
    <div class="row">
      <div class="field wide">
        <label>Input video path</label>
        <div style="display:flex; gap:.5rem;">
          <input name="input_video" id="input_video" value="{v(('input_video',))}" placeholder="raw/my_recording.mov" style="flex:1;">
          <button type="button" id="browse-btn">Browse&hellip;</button>
        </div>
      </div>
      <div class="field">
        <label>Work dir</label>
        <input name="work_dir" value="{v(('work_dir',), 'work')}">
      </div>
      <div class="field">
        <label>Output video</label>
        <input name="output_video" value="{v(('output_video',), 'output/final_cut.mp4')}">
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Planning / transcription</legend>
    <div class="row">
      <div class="field checkline">
        <input type="checkbox" id="use_llm" name="use_llm">
        <label for="use_llm">Use LLM to pick graphic segments (--llm)</label>
      </div>
      <div class="field checkline">
        <input type="checkbox" id="replan" name="replan">
        <label for="replan">Re-plan even if segments already exist (--replan)</label>
      </div>
      <div class="field checkline">
        <input type="checkbox" id="force_transcribe" name="force_transcribe">
        <label for="force_transcribe">Re-transcribe (--force-transcribe)</label>
      </div>
      <div class="field">
        <label>Whisper model</label>
        <select name="whisper_model">
          <option value="tiny">tiny</option>
          <option value="base">base</option>
          <option value="small" selected>small</option>
          <option value="medium">medium</option>
          <option value="large-v3">large-v3</option>
        </select>
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Silence removal</legend>
    <div class="row">
      <div class="field">
        <label>Noise floor (dB)</label>
        <input type="number" step="1" name="silence_noise_floor_db" value="{v(('silence','noise_floor_db'), -30)}">
      </div>
      <div class="field">
        <label>Min silence (sec)</label>
        <input type="number" step="0.05" name="silence_min_silence_sec" value="{v(('silence','min_silence_sec'), 0.6)}">
      </div>
      <div class="field">
        <label>Keep padding (sec)</label>
        <input type="number" step="0.05" name="silence_keep_padding_sec" value="{v(('silence','keep_padding_sec'), 0.15)}">
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Noise reduction</legend>
    <div class="row">
      <div class="field checkline">
        <input type="checkbox" id="nr_enabled" name="noise_reduction_enabled" {checked(('noise_reduction','enabled'))}>
        <label for="nr_enabled">Enabled</label>
      </div>
      <div class="field">
        <label>Quiet sample start (sec)</label>
        <input type="number" step="0.1" name="noise_reduction_profile_start_sec" value="{v(('noise_reduction','profile_start_sec'), 0.0)}">
      </div>
      <div class="field">
        <label>Quiet sample end (sec)</label>
        <input type="number" step="0.1" name="noise_reduction_profile_end_sec" value="{v(('noise_reduction','profile_end_sec'), 1.0)}">
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Audio enhance</legend>
    <div class="row">
      <div class="field">
        <label>Highpass (Hz)</label>
        <input type="number" step="1" name="audio_enhance_highpass_hz" value="{v(('audio_enhance','highpass_hz'), 80)}">
      </div>
      <div class="field">
        <label>Presence boost (dB)</label>
        <input type="number" step="0.5" name="audio_enhance_presence_boost_db" value="{v(('audio_enhance','presence_boost_db'), 3)}">
      </div>
      <div class="field">
        <label>Target loudness (LUFS)</label>
        <input type="number" step="0.5" name="audio_enhance_target_lufs" value="{v(('audio_enhance','target_lufs'), -16)}">
      </div>
      <div class="field">
        <label>Limiter ceiling (dB)</label>
        <input type="number" step="0.5" name="audio_enhance_limiter_ceiling_db" value="{v(('audio_enhance','limiter_ceiling_db'), -1.5)}">
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Render</legend>
    <div class="row">
      <div class="field"><label>Width</label><input type="number" name="render_width" value="{v(('render','width'), 1920)}"></div>
      <div class="field"><label>Height</label><input type="number" name="render_height" value="{v(('render','height'), 1080)}"></div>
      <div class="field"><label>FPS</label><input type="number" name="render_fps" value="{v(('render','fps'), 30)}"></div>
      <div class="field"><label>CRF (quality)</label><input type="number" name="render_crf" value="{v(('render','crf'), 18)}"></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Webcam bubble (pip)</legend>
    <div class="row">
      <div class="field">
        <label>Width % of frame</label>
        <input type="number" step="0.01" name="pip_width_pct" value="{v(('pip','width_pct'), 0.28)}">
      </div>
      <div class="field">
        <label>Position</label>
        <select name="pip_position">
          <option value="bottom-right" {selected(('pip','position'),'bottom-right')}>bottom-right</option>
          <option value="bottom-left" {selected(('pip','position'),'bottom-left')}>bottom-left</option>
          <option value="top-right" {selected(('pip','position'),'top-right')}>top-right</option>
          <option value="top-left" {selected(('pip','position'),'top-left')}>top-left</option>
          <option value="center" {selected(('pip','position'),'center')}>center</option>
        </select>
      </div>
      <div class="field"><label>Margin (px)</label><input type="number" name="pip_margin_px" value="{v(('pip','margin_px'), 40)}"></div>
      <div class="field"><label>Border width (px)</label><input type="number" name="pip_border_width_px" value="{v(('pip','border_width_px'), 6)}"></div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Graphics</legend>
    <div class="row">
      <div class="field wide">
        <label>An LLM writes each graphic segment as a labeled, animated HTML diagram; a headless browser records it.</label>
      </div>
      <div class="field">
        <label>Backend</label>
        <select name="graphics_backend">
          <option value="ollama" {selected(('graphics','backend'),'ollama')}>ollama (free, local)</option>
          <option value="anthropic" {selected(('graphics','backend'),'anthropic')}>anthropic (paid, needs ANTHROPIC_API_KEY)</option>
        </select>
      </div>
      <div class="field">
        <label>Model</label>
        <input name="graphics_model" value="{v(('graphics','model'), 'phi3.5')}" placeholder="phi3.5 or claude-opus-4-8">
      </div>
      <div class="field">
        <label>Ollama server</label>
        <input name="graphics_ollama_server" value="{v(('graphics','ollama_server'), 'http://127.0.0.1:11434')}">
      </div>
      <div class="field">
        <label>Timeout (sec)</label>
        <input type="number" name="graphics_timeout_sec" value="{v(('graphics','timeout_sec'), 180)}">
      </div>
    </div>
  </fieldset>

  <fieldset>
    <legend>Webcam background</legend>
    <div class="row">
      <div class="field wide">
        <label>Replaces your real webcam background with one static, themed image for the whole video (talk/full-frame segments only) — uses MediaPipe person segmentation, downloads a ~250KB model on first use.</label>
      </div>
      <div class="field checkline">
        <input type="checkbox" id="background_enabled" name="background_enabled" {checked(('background','enabled'))}>
        <label for="background_enabled">Enabled</label>
      </div>
      <div class="field">
        <label>Accent color</label>
        <input type="color" name="background_accent" value="{v(('background','accent'), '#2dd4bf')}">
      </div>
      <div class="field">
        <label>Edge feather (px)</label>
        <input type="number" name="background_feather_px" value="{v(('background','feather_px'), 9)}">
      </div>
      <div class="field">
        <label>Rim border width (px, 0 = off)</label>
        <input type="number" name="background_rim_glow_px" value="{v(('background','rim_glow_px'), 3)}">
      </div>
    </div>
  </fieldset>

  <div class="actions">
    <button type="button" id="save-btn">Save config</button>
    <button type="button" id="run-btn">Save &amp; Run</button>
    <button type="button" id="stop-btn">Stop</button>
    <span id="status"></span>
  </div>
</form>

<pre id="log"></pre>
<div class="toast" id="toast"></div>

<div class="modal-overlay" id="browse-modal">
  <div class="modal-box">
    <div class="modal-header">
      <h2>Select a video file</h2>
      <div id="browse-path"></div>
    </div>
    <div class="browse-toolbar">
      <button type="button" id="browse-home">Home</button>
      <button type="button" id="browse-project">Project folder</button>
    </div>
    <div id="browse-list"></div>
    <div class="modal-footer">
      <button type="button" id="browse-cancel">Cancel</button>
      <button type="button" id="browse-select" style="background:#2563eb;color:#fff;">Select</button>
    </div>
  </div>
</div>

<script>
const form = document.getElementById('cfg-form');
const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const runBtn = document.getElementById('run-btn');
const stopBtn = document.getElementById('stop-btn');
const toast = document.getElementById('toast');
let offset = 0;
let poller = null;

function showToast(msg) {{
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 1800);
}}

function collectFields() {{
  const data = {{}};
  new FormData(form).forEach((val, key) => {{ data[key] = val; }});
  form.querySelectorAll('input[type=checkbox]').forEach(cb => {{ data[cb.name] = cb.checked; }});
  return data;
}}

document.getElementById('save-btn').onclick = async () => {{
  const res = await fetch('/save', {{ method: 'POST', body: JSON.stringify(collectFields()) }});
  const j = await res.json();
  showToast(j.ok ? 'Saved config.yaml' : ('Error: ' + j.message));
}};

document.getElementById('run-btn').onclick = async () => {{
  const res = await fetch('/run', {{ method: 'POST', body: JSON.stringify(collectFields()) }});
  const j = await res.json();
  if (!j.ok) {{ showToast('Error: ' + j.message); return; }}
  offset = 0;
  logEl.textContent = '';
  logEl.style.display = 'block';
  startPolling();
}};

stopBtn.onclick = async () => {{
  await fetch('/stop', {{ method: 'POST' }});
  showToast('Stopping run...');
}};

function startPolling() {{
  runBtn.disabled = true;
  stopBtn.style.display = 'inline-block';
  statusEl.textContent = 'Running...';
  if (poller) clearInterval(poller);
  poller = setInterval(pollLog, 1000);
  pollLog();
}}

async function pollLog() {{
  const res = await fetch('/log?offset=' + offset);
  const j = await res.json();
  if (j.content) {{
    logEl.textContent += j.content;
    logEl.scrollTop = logEl.scrollHeight;
    offset = j.offset;
  }}
  if (j.state === 'finished') {{
    clearInterval(poller);
    poller = null;
    runBtn.disabled = false;
    stopBtn.style.display = 'none';
    statusEl.textContent = 'Finished (exit code ' + j.returncode + ')';
  }} else if (j.state === 'running') {{
    logEl.style.display = 'block';
    runBtn.disabled = true;
    stopBtn.style.display = 'inline-block';
    statusEl.textContent = 'Running...';
  }}
}}

// Resume polling if a run is already in progress when the page loads.
pollLog();

// --- File browser modal ---
const browseModal = document.getElementById('browse-modal');
const browseList = document.getElementById('browse-list');
const browsePath = document.getElementById('browse-path');
let browseSelected = null;

function openBrowser() {{
  browseSelected = null;
  browseModal.style.display = 'flex';
  loadDir(null);
}}
function closeBrowser() {{
  browseModal.style.display = 'none';
}}
async function loadDir(path) {{
  const url = path ? ('/browse?path=' + encodeURIComponent(path)) : '/browse';
  const res = await fetch(url);
  const j = await res.json();
  browsePath.textContent = j.cwd;
  browseList.dataset.parent = j.parent || '';
  browseList.dataset.home = j.home;
  browseList.dataset.project = j.project;
  browseList.innerHTML = '';
  if (j.parent) {{
    const up = document.createElement('div');
    up.className = 'browse-item';
    up.textContent = '.. (up)';
    up.onclick = () => loadDir(j.parent);
    browseList.appendChild(up);
  }}
  j.entries.forEach(e => {{
    const item = document.createElement('div');
    item.className = 'browse-item';
    item.textContent = (e.is_dir ? '📁 ' : '🎬 ') + e.name;
    if (e.is_dir) {{
      item.onclick = () => loadDir(e.path);
    }} else {{
      item.onclick = () => {{
        browseList.querySelectorAll('.browse-item.selected').forEach(x => x.classList.remove('selected'));
        item.classList.add('selected');
        browseSelected = e.path;
      }};
      item.ondblclick = () => {{ browseSelected = e.path; confirmBrowse(); }};
    }}
    browseList.appendChild(item);
  }});
}}
function confirmBrowse() {{
  if (browseSelected) {{
    document.getElementById('input_video').value = browseSelected;
  }}
  closeBrowser();
}}

document.getElementById('browse-btn').onclick = openBrowser;
document.getElementById('browse-cancel').onclick = closeBrowser;
document.getElementById('browse-select').onclick = confirmBrowse;
document.getElementById('browse-home').onclick = () => loadDir(browseList.dataset.home);
document.getElementById('browse-project').onclick = () => loadDir(browseList.dataset.project);
browseModal.onclick = (ev) => {{ if (ev.target === browseModal) closeBrowser(); }};
</script>
</body>
</html>"""


class Server(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        # A browser tab closing/reloading mid-poll disconnects the socket —
        # harmless and expected with a 1s polling loop. Don't spam a traceback.
        if sys.exc_info()[0] in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self):
        if self.path == "/":
            body = render_form().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/log"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            since = int(qs.get("offset", ["0"])[0])
            content, new_offset = "", since
            if LOG_PATH.exists():
                data = LOG_PATH.read_bytes()
                new_offset = len(data)
                content = ANSI_RE.sub("", data[since:].decode(errors="replace"))
            state, rc = run_status()
            self._send_json({"content": content, "offset": new_offset, "state": state, "returncode": rc})
            return

        if self.path.startswith("/browse"):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            raw_path = qs.get("path", [str(PROJECT_DIR)])[0]
            target = Path(raw_path).expanduser()
            if not target.exists() or not target.is_dir():
                target = PROJECT_DIR
            target = target.resolve()
            parent = str(target.parent) if target.parent != target else None
            self._send_json({
                "cwd": str(target), "parent": parent,
                "entries": list_dir(target),
                "home": str(Path.home()), "project": str(PROJECT_DIR),
            })
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/save":
            fields = self._read_json_body()
            try:
                save_config(fields)
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "message": str(e)}, status=400)
            return

        if self.path == "/run":
            fields = self._read_json_body()
            ok, message = start_run(fields)
            self._send_json({"ok": ok, "message": message})
            return

        if self.path == "/stop":
            ok, message = stop_run()
            self._send_json({"ok": ok, "message": message})
            return

        self.send_response(404)
        self.end_headers()


def main():
    server = Server(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"Auto-Edit Pipeline UI running at {url}")
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
