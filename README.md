# Auto-Edit Pipeline

Fully scripted: silence removal → noise reduction → loudness/EQ enhancement →
ComfyUI-generated motion graphics → elliptical webcam bubble → final render.
Runs entirely outside DaVinci Resolve (which keeps it free-version-friendly);
`resolve_import.py` is an optional last step to hand the finished file to
Resolve for color/caption polish.

## 1. One-time setup

```bash
cd auto_edit_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

ffmpeg must be on your PATH (`ffmpeg -version` to check — you already have it).

## 2. Export a ComfyUI workflow template

You already have ComfyUI running. In the ComfyUI UI:

1. Build the node graph for the *look* of your motion graphics once (e.g. a
   text-to-video workflow — AnimateDiff, SVD, or your preferred setup).
2. Rename the positive-prompt `CLIPTextEncode` node's title to include the
   word **"Prompt"** (double-click the node title bar). This is how the
   script finds which node to inject your per-segment text into.
3. Menu → **Save (API Format)** → save as
   `comfy_workflows/motion_graphic_template.json` inside this project folder.

The script doesn't design the graphic style for you — it reuses whatever
workflow you've already dialed in, just swapping the prompt text per segment.

## 3. Let the transcript decide where graphics go (new)

Instead of hand-picking timestamps, generate a transcript and let a planner
propose segments for you:

```bash
# 1. Transcribe locally (no cloud calls) — takes a few minutes depending on model size
python3 transcribe.py raw/my_recording.mov work/transcript.json

# 2a. Instant, no models involved — keyword-based heuristic
python3 plan_segments.py work/transcript.json config.yaml

# 2b. Better judgment, still fully free & local — via Ollama
#     one-time setup: brew install ollama && ollama pull llama3.1
python3 plan_segments.py work/transcript.json config.yaml --llm

# 2c. Optional: use Claude via the Anthropic API instead of Ollama
#     (only if you specifically want it and have a key — costs money)
export ANTHROPIC_API_KEY="sk-ant-..."
python3 plan_segments.py work/transcript.json config.yaml --llm --backend anthropic
```

`--llm` defaults to Ollama (`--backend ollama`), so nothing here requires an
API key or leaves your machine. Make sure Ollama is running first — either
open the Ollama app, or `ollama serve` in a terminal — and that you've
pulled a model once (`ollama pull llama3.1`). Swap models with `--model`,
e.g. `--model mistral` or `--model llama3.1:70b` if you have the hardware
for a bigger one (bigger models judge these picks noticeably better).

Both modes print their picks first without touching anything. Read them —
timestamps, and especially the generated prompts — then re-run with
`--write` to actually insert them into `config.yaml`'s `segments:` list:

```bash
python3 plan_segments.py work/transcript.json config.yaml --llm --write
```

**Why review before `--write`:** motion-graphic generation via ComfyUI is
the slowest, most expensive stage to redo. A minute spent skimming the
picks (and rewording a prompt if it's vague) is cheaper than regenerating
graphics later. The heuristic mode looks for phrasing like "the way this
works is...", "imagine...", "let me show you...", "step by step" etc. — it's
a blunt instrument and will occasionally flag something not worth a graphic,
or miss one phrased unusually. The LLM modes read for actual meaning and are
noticeably more selective — smaller local models (e.g. 8B) are decent but
less consistent than Claude at following the "don't overlap picks" and
JSON-only instructions, so skim the output a bit more carefully with those.

You can still hand-edit `segments:` in `config.yaml` afterwards — nothing
about auto-planning locks you out of manually adding, removing, or
re-timing entries.

## 4. Configure the rest of the video

Edit `config.yaml`:

- `input_video`: path to your single combined recording
- `silence` / `noise_reduction` / `audio_enhance`: tweak if defaults don't fit
- `segments`: auto-filled by `plan_segments.py --write` (step 3 above), or
  hand-write your own — timestamps refer to your **original, uncut**
  recording either way
- `pip`: bubble position/size on screen

## 5. Run it

```bash
python3 pipeline.py config.yaml
```

This will:
1. Split your recording into alternating "talk" and "graphic" ranges
2. Silence-cut, denoise, and loudness-normalize every talk range
3. For every graphic range: queue ComfyUI, download the result, crop your
   own webcam footage into an oval, and composite it over the graphic
4. Concatenate everything into `output/final_cut.mp4`

Expect it to take a while — ComfyUI generation is the slow part, especially
if you're on CPU or a smaller GPU.

## 6. (Optional) Hand off to Resolve for finishing

Make sure DaVinci Resolve is open, then:

```bash
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:$RESOLVE_SCRIPT_API/Modules/"

python3 resolve_import.py output/final_cut.mp4 "My YouTube Video"
```

This creates/opens the named project and drops the final render onto a new
timeline, ready for color grading, thumbnail picking, or captions.

## Notes / things worth knowing

- **Silence detection** (`silence.noise_floor_db`, `min_silence_sec`) is the
  setting you'll tune most. Too aggressive and it eats breath pauses that
  make speech sound natural; too lax and it barely cuts anything. Start
  with the defaults and adjust after watching one output.
- **Noise reduction** needs a genuinely quiet sample — point
  `noise_reduction.profile_start_sec/end_sec` at a moment with only room
  tone, no speech.
- **Graphic segment timestamps** are matched against your *original*
  recording on purpose — that's the file you'll actually scrub through to
  decide "this is where I explain the concept," so there's no need to
  redo the math after silence-cutting shifts things around.
- ComfyUI output can be a still image or a short video depending on your
  workflow; the script handles either (stills get held for the segment's
  duration).
- Every intermediate stage writes into `work/` so you can inspect or resume
  without re-running everything from scratch.
