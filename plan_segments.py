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
        # A flat start+graphic_duration_sec window ignores how far a merge
        # group actually spans -- several trigger phrases close together
        # usually mean one continuous multi-part explanation, and capping
        # it to the single-concept default duration would truncate the
        # graphic before the explanation (and the transcript-timed step
        # reveals within it) actually finishes.
        end = max(m["start"] + graphic_duration_sec, m["end"])
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


def build_llm_system_prompt(max_segments, min_gap_sec):
    return (
        "You edit educational YouTube videos. You'll be given a timestamped "
        "transcript. Identify up to "
        f"{max_segments} moments where the speaker is explaining a concept "
        "that would genuinely be clearer with an animated motion graphic "
        "on screen (not every explanation needs one — pick the strongest "
        "candidates: multi-step processes, abstract mechanisms, comparisons, "
        "data flows, architectures).\n\n"
        "If the speaker walks through several sequential sub-points as one "
        "continuous explanation (a numbered list, 'first...second...third...', "
        "a checklist, steps of a process), that is ONE pick covering the "
        "whole span, with a single prompt that lists each sub-point in "
        "order — never split a single ongoing list into multiple separate "
        "picks. Splitting it produces a wall of back-to-back graphics with "
        "no breathing room, which reads as chaotic, not a series of "
        "distinct concepts.\n\n"
        f"Leave at least {min_gap_sec:.0f} seconds of gap between the end of "
        "one pick and the start of the next, so the video returns to the "
        "presenter's full-frame camera between graphics instead of chaining "
        "graphic after graphic. For each pick, give a start/end timestamp "
        "(end should be roughly 10-20 seconds after start for a single "
        "concept, longer for a consolidated multi-step list, enough to "
        "cover the explanation, never overlapping another pick) and a "
        "specific, visual prompt describing the graphic to generate — for "
        "a multi-step list, enumerate the steps in the prompt in the exact "
        "order they're spoken. "
        "Respond with ONLY a JSON array, no prose, no markdown fences: "
        '[{"start": 12.3, "end": 27.0, "prompt": "..."}]'
    )


def _merge_close_picks(picks, min_gap_sec):
    """
    Code-level backstop for the min-gap/consolidation instructions in
    build_llm_system_prompt: models vary in how well they follow "leave a
    gap" and "don't split one list into multiple picks" instructions
    (smaller/local models especially), so don't rely on the prompt alone —
    merge any picks that end up close together anyway, unioning their time
    range and concatenating their prompts. Same shape of fix as
    heuristic_plan's own nearby-hit merge, applied to LLM output instead of
    trigger-phrase hits.
    """
    picks = sorted(picks, key=lambda p: p["start"])
    merged = []
    for p in picks:
        if merged and p["start"] - merged[-1]["end"] < min_gap_sec:
            merged[-1]["end"] = max(merged[-1]["end"], p["end"])
            merged[-1]["prompt"] = merged[-1]["prompt"].rstrip(". ") + "; then: " + p["prompt"]
        else:
            merged.append(dict(p))
    return merged


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


def huggingface_plan(transcript, model="deepseek-ai/DeepSeek-V4-Flash:novita", max_segments=12,
                      min_gap_sec=6.0):
    """
    Sends the transcript to the Hugging Face router using the OpenAI-compatible
    client and asks it to pick moments that need a visual aid.

    min_gap_sec is enforced twice: once as an instruction to the model, and
    again in code via _merge_close_picks regardless of whether the model
    actually followed it — see that function's docstring for why the prompt
    alone isn't trusted to guarantee this.
    """
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is not set. Add it to your environment or .env file.")

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.getenv("HF_TOKEN"),
    )

    transcript_text = "\n".join(
        f"[{e['start']:.1f}-{e['end']:.1f}] {e['text']}" for e in transcript
    )
    system = build_llm_system_prompt(max_segments, min_gap_sec)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": transcript_text},
        ],
    )
    raw = completion.choices[0].message.content
    picks = parse_llm_json_picks(raw)
    picks = _merge_close_picks(picks, min_gap_sec)
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
    ap.add_argument("--min-gap-sec", type=float, default=None,
                     help="minimum gap between consecutive graphic picks, so they don't chain "
                          "back-to-back with no return to camera (default: 6.0 for --llm, 8.0 heuristic)")
    args = ap.parse_args()

    transcript = json.loads(Path(args.transcript_json).read_text())

    if args.llm:
        segments = huggingface_plan(
            transcript,
            model=args.model or "deepseek-ai/DeepSeek-V4-Flash:novita",
            max_segments=args.max_segments,
            **({"min_gap_sec": args.min_gap_sec} if args.min_gap_sec is not None else {}),
        )
    else:
        segments = heuristic_plan(
            transcript, max_segments=args.max_segments,
            **({"min_gap_sec": args.min_gap_sec} if args.min_gap_sec is not None else {}),
        )

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
