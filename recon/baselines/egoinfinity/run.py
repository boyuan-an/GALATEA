from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from baselines.common.execution import REPOSITORY, python_command
from baselines.common.inputs import write_json

from .prompts import clip_prompt_metadata


def run(clip, config, execution):
    repo = Path(config["repo"])
    root, native = Path(clip["root"]), Path(clip["native"])
    frames = native / "frames"
    if not frames.exists():
        frames.mkdir()
        for frame in sorted((native / "rgb").glob("*.png")):
            with Image.open(frame) as image:
                image.convert("RGB").save(frames / (frame.stem + ".jpg"), quality=95)
    prompt = clip_prompt_metadata(clip["object_name"])
    write_json(
        native / "manifest.json",
        {
            "video_uri": clip["video"],
            "objects": prompt["expanded_prompts"],
            "objects_source": prompt["protocol_id"],
            "benchmark_prompt_adapter": prompt,
            "n_frames": clip["frames"],
            "preserve_source_frames": True,
            "fps": clip["fps"],
        },
    )
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    env = {
        "EGOINFINITY_REVIEWED_FRAME0_MASK_ROOT": "",
        "SAM3D_WORKER_SOCKET": str(
            Path(tempfile.gettempdir()) / f"galatea_{token}_{os.getpid()}.sock"
        ),
        "EGOINFINITY_RUN_SAM3D": "1",
        "EGOINFINITY_RUN_TRACK": "1",
    }
    if clip.get("frame0_mask"):
        mask_root = root / "reviewed_masks"
        mask_root.mkdir(exist_ok=True)
        mask = np.asarray(Image.open(clip["frame0_mask"])) > 0
        np.savez_compressed(mask_root / (clip["clip_id"] + ".npz"), mask=mask)
        env["EGOINFINITY_REVIEWED_FRAME0_MASK_ROOT"] = str(mask_root)
    execution.run(
        "frontend",
        python_command(config, "ego_python")
        + [
            "-m",
            "egoinfinity",
            "process",
            native,
            "--set",
            "phase1.args.with_sam3d_worker=true",
            "--set",
            "bake_fp.enabled=false",
            "--fail-fast",
        ],
        repo,
        env,
    )
    runlist = root / "runlist.json"
    write_json(
        runlist,
        {
            "baseline": "egoinfinity_official",
            "clips": [
                {"clip_id": clip["clip_id"], "contract_inputs": {"known_metric_mesh": clip["mesh"]}}
            ],
        },
    )
    fp_root = root / "fpplusplus"
    adapter = python_command(config, "adapter_python") + ["-m", "baselines.egoinfinity.fpplusplus"]
    execution.run(
        "prepare_fpplusplus",
        adapter
        + [
            "prepare",
            "--favorites-root",
            native.parent,
            "--fp-root",
            fp_root,
            "--runlist",
            runlist,
        ],
        REPOSITORY,
    )
    execution.run(
        "track_fpplusplus",
        [
            sys.executable,
            "-m",
            "baselines.egoinfinity.track",
            "--root",
            fp_root,
            "--fp-repo",
            config["fp_repo"],
            "--python-command",
            json.dumps(python_command(config, "fp_python")),
        ],
        REPOSITORY,
        config.get("fp_env", {}),
    )
    execution.run(
        "bake_fpplusplus",
        adapter
        + [
            "bake",
            "--favorites-root",
            native.parent,
            "--fp-root",
            fp_root,
            "--bake-cache",
            root / "baked",
            "--ego-repo",
            repo,
            "--ego-command",
            json.dumps(python_command(config, "ego_python")),
            "--report",
            root / "bake_report.json",
        ],
        REPOSITORY,
    )
    execution.run(
        "canonicalize",
        [
            sys.executable,
            "-m",
            "baselines.convert",
            "--method",
            "egoinfinity",
            "--inputs",
            root / "inputs.json",
        ],
        REPOSITORY,
    )
