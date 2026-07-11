#!/usr/bin/env python3
"""
graphics_comfyui.py — queue a saved ComfyUI workflow (API format) with a
per-segment prompt injected into the node titled "Prompt", poll until it
finishes, and download the resulting image/video.
"""

import json
import random
import time
import uuid
from pathlib import Path

import requests


def _load_workflow(workflow_path, prompt_text):
    workflow = json.loads(Path(workflow_path).read_text())
    injected = False
    for node in workflow.values():
        title = node.get("_meta", {}).get("title", "")
        inputs = node.get("inputs", {})
        if "prompt" in title.lower() and "text" in inputs:
            inputs["text"] = prompt_text
            injected = True
        # Randomize any sampler seed so segments don't all reuse the
        # template's fixed seed and come out looking near-identical.
        if "seed" in inputs and isinstance(inputs["seed"], (int, float)):
            inputs["seed"] = random.randint(0, 2**32 - 1)
    if not injected:
        raise RuntimeError(
            f"No CLIPTextEncode node titled with 'Prompt' found in {workflow_path}. "
            "In ComfyUI, double-click the positive-prompt node's title bar and "
            "rename it to include the word 'Prompt', then re-save (API format)."
        )
    return workflow


def generate_motion_graphic(server, workflow_path, prompt_text, work_dir, poll_interval=2, timeout=600):
    """
    Queue `workflow_path` on the ComfyUI server at `server` (e.g.
    "http://127.0.0.1:8188") with `prompt_text` injected into the "Prompt"
    node, wait for it to finish, download the resulting asset into work_dir,
    and return its path.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    server = server.rstrip("/")
    client_id = str(uuid.uuid4())

    workflow = _load_workflow(workflow_path, prompt_text)

    resp = requests.post(f"{server}/prompt", json={"prompt": workflow, "client_id": client_id})
    resp.raise_for_status()
    prompt_id = resp.json()["prompt_id"]

    print(f"    [comfyui] queued prompt_id={prompt_id}, waiting...")
    deadline = time.time() + timeout
    history = None
    while time.time() < deadline:
        h_resp = requests.get(f"{server}/history/{prompt_id}")
        h_resp.raise_for_status()
        h = h_resp.json()
        if prompt_id in h:
            history = h[prompt_id]
            break
        time.sleep(poll_interval)

    if history is None:
        raise TimeoutError(f"ComfyUI job {prompt_id} did not finish within {timeout}s")

    status = history.get("status", {})
    if status.get("status_str") == "error":
        raise RuntimeError(f"ComfyUI job {prompt_id} failed: {status}")

    outputs = history.get("outputs", {})
    for node_output in outputs.values():
        for key in ("images", "gifs", "videos"):
            for item in node_output.get(key, []):
                return _download_output(server, item, work_dir)

    raise RuntimeError(f"ComfyUI job {prompt_id} finished with no image/video outputs: {history}")


def _download_output(server, item, work_dir):
    params = {
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    }
    resp = requests.get(f"{server}/view", params=params, stream=True)
    resp.raise_for_status()
    out_path = work_dir / item["filename"]
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    print(f"    [comfyui] downloaded {out_path}")
    return out_path
