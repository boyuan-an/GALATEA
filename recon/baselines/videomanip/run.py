from __future__ import annotations

import shutil
import sys
from pathlib import Path

from baselines.common.execution import REPOSITORY, python_command
from baselines.common.inputs import write_json


def run(clip, config, execution):
    repo = Path(config["repo"])
    recon = repo / "reconstruction"
    native = Path(clip["native"])
    root = Path(clip["root"])
    fp = Path(config["fp_repo"])
    hand = recon / "hamer" / "hamer"
    sam = recon / "sam2" / "sam2"
    target = native / "mesh_original" / "textured_simple_grasp.obj"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(clip["mesh"], target)
    clicks = root / "inputs" / "clicks.json"
    write_json(
        clicks,
        {
            clip["clip_id"]: [
                {
                    "clicked_points": [clip["point"]],
                    "clicked_labels": [1],
                    "object_type": "grasp_object",
                }
            ]
        },
    )
    execution.run(
        "depth",
        python_command(config, "moge_python")
        + [
            str(recon / "utils" / "moge_depth.py"),
            str(native / "rgb"),
            "--output_folder",
            str(native / "cam_info"),
        ],
        recon,
    )
    shutil.copy2(clip["intrinsics"], native / "cam_K.txt")
    execution.run(
        "hand",
        python_command(config, "hand_python")
        + [
            "hand_detection.py",
            "--img_folder",
            native / "rgb",
            "--out_folder",
            native / "human_hand",
        ],
        hand,
        {"PYTHONPATH": str(hand) + ":" + str(hand / "third-party" / "ViTPose")},
    )
    execution.run(
        "segmentation",
        python_command(config, "sam_python")
        + [
            "run_video_function.py",
            native / "rgb",
            "--hand_input_folder",
            native / "human_hand",
            "--obj_output_dir",
            native / "masks_pred_obj",
            "--hand_output_dir",
            native / "masks_pred_hand",
        ],
        sam,
        {"VIDEOMANIP_CLICK_JSON": str(clicks)},
    )
    execution.run(
        "object",
        python_command(config, "fp_python")
        + [
            "run_demo_multi_scale.py",
            "--object_folder",
            clip["clip_id"],
            "--object_type",
            "grasp",
            "--scale_range",
            "1.0",
            "--est_refine_iter",
            str(config.get("est_refine_iter", 5)),
            "--track_refine_iter",
            str(config.get("track_refine_iter", 2)),
            "--debug_dir",
            root / "foundationpose_debug",
        ],
        fp,
        {"DATA_ROOT": str(native.parent)},
    )
    execution.run(
        "canonicalize",
        [
            sys.executable,
            "-m",
            "baselines.convert",
            "--method",
            "videomanip",
            "--inputs",
            root / "inputs.json",
        ],
        REPOSITORY,
    )
