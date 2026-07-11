#!/usr/bin/env python3
"""
plan_segments.py — decides WHERE motion graphics should go, using the
transcript from transcribe.py. Two modes:

  heuristic (default, instant, no models involved):
    Scans transcript text for phrasing that typically signals someone is
    explaining a concept ("the way this works is...", "imagine...", "let me
    show you...", etc.), groups nearby hits into segments, and writes a
    generic graphic prompt built from the surrounding sentence.

  --llm (better judgment, uses the Hugging Face router):
    Sends the transcript to a hosted model through the Hugging Face router
    and asks it to pick moments that actually need a visual aid, with a
    tailored prompt per segment. Requires a Hugging Face token in HF_TOKEN
    (or your .env file).

Either way, this WRITES SUGGESTIONS, not final answers — it prints them
and (with --write) inserts them into config.yaml's `segments:` list, but
you should skim the prompts/timestamps once before running the full
pipeline. Motion graphics are the most expensive/slow stage to redo.

Usage:
    python3 plan_segments.py work/transcript.json config.yaml
    python3 plan_segments.py work/transcript.json config.yaml --llm
    python3 plan_segments.py work/transcript.json config.yaml --llm --model deepseek-ai/DeepSeek-V4-Flash:novita
    python3 plan_segments.py work/transcript.json config.yaml --llm --write
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN"),
)

TRIGGER_PHRASES = [
    "the way this works", "here's how", "let me show you", "imagine",
    "think of it like", "think of it as", "for example", "basically what happens",
    "under the hood", "let's break this down", "let's break down",
    "how does this work", "the reason why", "picture this", "visualize",
    "step by step", "what's happening here", "so essentially", "in other words",
    "let's walk through", "here's what's going on",
]


def heuristic_plan(transcript, min_gap_sec=8.0, graphic_duration_sec=15.0, max_segments=12):
    """
    Flags transcript entries containing trigger phrases, merges nearby hits,
    and extends each hit forward by graphic_duration_sec (capped by the next
    transcript entry's start, so graphics don't run past new speech topics).
    """
    hits = []
    for entry in transcript:
        text_lower = entry["text"].lower()
        if any(phrase in text_lower for phrase in TRIGGER_PHRASES):
            hits.append(entry)

    # Merge hits that are close together in time into one segment
    merged = []
    for h in hits:
        if merged and h["start"] - merged[-1]["end"] < min_gap_sec:
            merged[-1]["end"] = h["end"]
            merged[-1]["text"] += " " + h["text"]
        else:
            merged.append({"start": h["start"], "end": h["end"], "text": h["text"]})

    segments = []
    for m in merged[:max_segments]:
        end = m["start"] + graphic_duration_sec
        segments.append({
            "start": seconds_to_hms(m["start"]),
            "end": seconds_to_hms(end),
            "prompt": build_prompt_from_text(m["text"]),
            "_source_text": m["text"],  # kept for your review, safe to delete
        })
    return segments


def build_prompt_from_text(text, max_words=25):
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    topic = " ".join(words[:max_words])
    return (f"clean minimal animated motion graphic explaining: {topic}. "
             f"dark background, simple shapes, blue/teal accent color")


def seconds_to_hms(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def build_llm_system_prompt(max_segments):
    return (
        "You edit educational YouTube videos. You'll be given a timestamped "
        "transcript. Identify up to "
        f"{max_segments} moments where the speaker is explaining a concept "
        "that would genuinely be clearer with an animated motion graphic "
        "on screen (not every explanation needs one — pick the strongest "
        "candidates: multi-step processes, abstract mechanisms, comparisons, "
        "data flows, architectures). For each, give a start/end timestamp "
        "(end should be roughly 10-20 seconds after start, enough to cover "
        "the explanation, not overlapping other picks) and a specific, "
        "visual, one-sentence prompt describing the graphic to generate. "
        "Respond with ONLY a JSON array, no prose, no markdown fences: "
        '[{"start": 12.3, "end": 27.0, "prompt": "..."}]'
    )


def parse_llm_json_picks(raw_text):
    cleaned = re.sub(r"^```(json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    # Some local models wrap the array in prose despite instructions — grab
    # the first [...] block as a fallback.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            raise RuntimeError(
                "Model response wasn't valid JSON and no [...] block was "
                f"found. Raw response:\n{raw_text}"
            )
        return json.loads(match.group(0))


def picks_to_segments(picks):
    segments = []
    for p in picks:
        segments.append({
            "start": seconds_to_hms(p["start"]),
            "end": seconds_to_hms(p["end"]),
            "prompt": p["prompt"],
        })
    return segments


def huggingface_plan(transcript, model="deepseek-ai/DeepSeek-V4-Flash:novita", max_segments=12):
    """
    Sends the transcript to the Hugging Face router using the OpenAI-compatible
    client and asks it to pick moments that need a visual aid.
    """
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Add it to your environment or .env file.")

    transcript_text = "\n".join(
        f"[{e['start']:.1f}-{e['end']:.1f}] {e['text']}" for e in transcript
    )
    system = build_llm_system_prompt(max_segments)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": transcript_text},
        ],
    )
    raw = completion.choices[0].message.content
    picks = parse_llm_json_picks(raw)
    return picks_to_segments(picks)


def write_segments_to_config(segments, config_path):
    cfg = yaml.safe_load(Path(config_path).read_text())
    clean_segments = [
        {k: v for k, v in s.items() if not k.startswith("_")} for s in segments
    ]
    cfg["segments"] = clean_segments
    Path(config_path).write_text(yaml.dump(cfg, sort_keys=False, allow_unicode=True))
    print(f"[plan_segments] wrote {len(segments)} segments into {config_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript_json")
    ap.add_argument("config_yaml")
    ap.add_argument("--llm", action="store_true",
                     help="use a language model to pick segments instead of keyword heuristics")
    ap.add_argument("--model", default=None,
                     help="model name override (default: deepseek-ai/DeepSeek-V4-Flash:novita)")
    ap.add_argument("--write", action="store_true", help="write results into config.yaml's segments list")
    ap.add_argument("--max-segments", type=int, default=12)
    args = ap.parse_args()

    transcript = json.loads(Path(args.transcript_json).read_text())

    if args.llm:
        segments = huggingface_plan(
            transcript,
            model=args.model or "deepseek-ai/DeepSeek-V4-Flash:novita",
            max_segments=args.max_segments,
        )
    else:
        segments = heuristic_plan(transcript, max_segments=args.max_segments)

    print(f"\n{'='*60}\nProposed {len(segments)} graphic segment(s):\n{'='*60}")
    for s in segments:
        print(f"  {s['start']} -> {s['end']}")
        print(f"    prompt: {s['prompt']}")
        if "_source_text" in s:
            print(f"    (from transcript: \"{s['_source_text'][:100]}...\")")
        print()

    if args.write:
        write_segments_to_config(segments, args.config_yaml)
    else:
        print("Review the above. Re-run with --write to insert these into config.yaml,")
        print("or hand-edit the prompts first and add --write after.")


if __name__ == "__main__":
    main()
