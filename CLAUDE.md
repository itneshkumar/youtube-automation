# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A scripted pipeline that turns a single raw screen/webcam recording into an edited
YouTube-ready video: silence removal → noise reduction → loudness/EQ → animated
motion-graphic explainers (auto-placed via transcript analysis) → elliptical webcam
bubble overlay → final concat render. Runs entirely outside DaVinci Resolve; Resolve
is only used as an optional last step for color/captions.

## Commands

```bash
# One-time setup (also done automatically by start.sh / start_ui.sh)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m playwright install chromium   # needed for graphics_llm.py's headless rendering

# End-to-end run (transcribe -> plan graphic segments -> render)
./start.sh raw/my_recording.mov
./start.sh raw/my_recording.mov --llm              # smarter segment planning via HF router
./start.sh raw/my_recording.mov --replan            # redo planning even if segments exist
./start.sh raw/my_recording.mov --force-transcribe  # redo transcription
./start.sh raw/my_recording.mov --whisper-model medium

# Browser UI instead of the CLI (http://127.0.0.1:8787, loopback-only)
./start_ui.sh

# Individual stages, run by hand
python3 transcribe.py raw/my_recording.mov work/transcript.json [model_size]
python3 plan_segments.py work/transcript.json config.yaml [--llm] [--write]
python3 pipeline.py config.yaml   # the actual render, once config.yaml has segments
```

`README.md` also documents an optional `resolve_import.py` hand-off step to DaVinci
Resolve after rendering — that script is not present in the repo yet.

There is no test suite, linter, or build step in this repo.

`ffmpeg` must be on `PATH`. `config.yaml` is gitignored (holds the real input path and
any locally-tuned settings) — copy `config.example.yaml` to start, or let `start.sh` do
it automatically.

## Architecture

**Pipeline shape (`pipeline.py`):** the original recording is split into an ordered,
alternating list of "talk" ranges and "graphic" ranges (from `config.yaml`'s
`segments:` list, timestamped against the *original, uncut* recording). Each range is
processed independently into its own `work/range_NN_{talk,graphic}/` folder — this
independence is what makes them safe to run concurrently in a `ThreadPoolExecutor`
(`parallel_workers` in config, default `min(4, cpu_count)`). Ranges are concatenated
in order only at the very end via ffmpeg's `concat` *filter* (not the concat demuxer —
talk ranges keep native source resolution while graphic ranges are already
`frame_w x frame_h`, and the demuxer assumes homogeneous inputs).

Processing each range independently and concatenating afterward — rather than cutting
silence across the whole timeline first — is deliberate: it avoids timestamp drift
between silence-removal and graphic-segment placement.

- **Talk range:** `audio_tools.py` — silence cut → (optional) `background_replace.py`
  virtual-background swap → noise reduction → loudness/EQ enhancement.
- **Graphic range:** `graphics_llm.py` generates the motion graphic, `pip_overlay.py`
  crops the webcam feed into a bordered oval and composites it on top, then the same
  noise-reduction/enhancement chain from `audio_tools.py` is applied to keep audio
  consistent with talk ranges.

**Noise profile is sampled once, globally.** `pipeline.py` calls
`audio_tools.extract_noise_profile()` once against the *original* recording at a
user-configured quiet moment and reuses that same profile for every range. Sampling
per-range-clip instead (the old behavior) grabs mostly-speech windows, which teaches
`noisereduce` to suppress the voice itself. Same idea for the virtual background: one
themed image is generated once per run and reused for every talk range so the look
stays consistent.

**Where graphics come from — two backends exist, only one is wired up.**
`graphics_llm.py` is what `pipeline.py` actually calls: it asks an LLM (a local
llama.cpp `llama-server` by default, or Anthropic) to extract the ordered list of steps/concepts a transcript
segment actually names, then renders those into one of two *fixed, hand-written* HTML/CSS
animation templates (vertical stacked cards or horizontal timeline) via Playwright
headless Chromium, recorded to mp4. The LLM is deliberately *not* asked to write
animation JS/CSS itself — small local models produce plausible-looking but
non-functional or unsequenced code — so layout/timing is template code, and the model
only supplies content. Step reveal timing is matched to when each concept is actually
spoken in the transcript (`_assign_step_delays`), not evenly spaced.

`graphics_comfyui.py` (queue a saved ComfyUI workflow, poll, download the output) and
the ComfyUI-centric workflow described in `README.md` are a separate, currently
*unused* code path — `pipeline.py` does not import it. Treat `README.md`'s ComfyUI
setup instructions as describing an alternate/legacy flow, not the one `start.sh`
actually runs today.

**Segment planning is advisory, not final.** `plan_segments.py` decides *where*
graphics go, either via keyword heuristics (`TRIGGER_PHRASES` matched against
transcript text) or an LLM (Hugging Face router, needs `HF_TOKEN`) that judges which
explanations genuinely warrant a visual. Either mode only *writes suggestions* into
`config.yaml`'s `segments:` list (with `--write`); segments should be skimmed before
running the full pipeline since motion-graphic generation is the slowest/most
expensive stage to redo. `start.sh` skips planning if `config.yaml` already has
segments, unless `--replan` is passed.

**Config is the source of truth for a run.** All stage parameters (silence
thresholds, noise reduction, EQ/loudness targets, render resolution/codec, pip
bubble size/position, background theme, graphics backend/model, segments) live in
`config.yaml`, loaded fresh by whichever script runs. `ui_server.py` is a thin
stdlib-only HTTP server (`FIELD_SPECS` maps form fields to nested config paths) that
edits the same `config.yaml` and shells out to `start.sh` for the actual run — it has
no independent pipeline logic of its own.

**Timestamp convention:** every `start`/`end` in `config.yaml`'s `segments:` and
anywhere ranges are computed refers to the *original, uncut* recording, not the
silence-trimmed output — this is what lets `plan_segments.py` and manual edits use
timestamps you can actually scrub to in the raw file.
