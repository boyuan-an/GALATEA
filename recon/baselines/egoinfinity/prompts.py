from __future__ import annotations

import json
import os
import string
from pathlib import Path
from typing import Any

PROMPT_PROTOCOL_ID = "supplied_object_noun_plus_rightmost_head_v1"
PROMPT_PROTOCOL = {
    "protocol_id": PROMPT_PROTOCOL_ID,
    "input": "supplied object_noun only",
    "rule": "Use the normalized supplied object noun, then its rightmost whitespace-delimited head noun; deduplicate while preserving order.",
    "dataset_independent": True,
    "forbidden_inputs": ["GT", "detections", "masks", "per-clip failure results"],
    "failure_policy": "Never append or change prompt words after an individual clip failure.",
}


def expand_object_noun(value: str) -> list[str]:
    supplied = " ".join(str(value).split())
    if not supplied:
        raise ValueError("supplied object noun is empty")
    final_token = supplied.rsplit(" ", 1)[-1]
    head = final_token.strip(string.punctuation) or final_token
    prompts: list[str] = []
    seen: set[str] = set()
    for prompt in (supplied, head):
        key = prompt.casefold()
        if key not in seen:
            prompts.append(prompt)
            seen.add(key)
    return prompts


def clip_prompt_metadata(supplied_noun: str) -> dict[str, Any]:
    prompts = expand_object_noun(supplied_noun)
    return {
        "protocol_id": PROMPT_PROTOCOL_ID,
        "supplied_object_noun": " ".join(str(supplied_noun).split()),
        "rightmost_head_noun": prompts[-1],
        "expanded_prompts": prompts,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def inject_clip_manifest(manifest_path: Path, supplied_noun: str) -> dict[str, Any]:
    metadata = clip_prompt_metadata(supplied_noun)
    manifest = json.loads(manifest_path.read_text())
    manifest["objects"] = metadata["expanded_prompts"]
    manifest["objects_source"] = PROMPT_PROTOCOL_ID
    manifest["benchmark_prompt_adapter"] = metadata
    atomic_json(manifest_path, manifest)
    return metadata
