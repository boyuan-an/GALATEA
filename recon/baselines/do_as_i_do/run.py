from __future__ import annotations

import sys
from pathlib import Path

from baselines.common.execution import REPOSITORY, python_command


def run(clip, config, execution):
    recon = Path(config["repo"]) / "reconstruction"
    root, native = Path(clip["root"]), Path(clip["native"])
    object_id = clip["object_name"].replace(" ", "_")
    command = [
        "bash",
        str(recon / "run_pipeline.sh"),
        clip["video"],
        "0",
        clip["object_name"],
        "right",
    ]
    clean = {
        "DOAS_INPUT_MESH": "",
        "DOAS_INPUT_INTRINSICS": "",
        "DOAS_START_STAGE": "0",
        "DOAS_STOP_AFTER_STAGE": "2",
        "DOAS_REUSE_TRACK_LAYOUT": "0",
        "DOAS_REUSE_POINTMAP_INTRINSICS": "0",
        "DOAS_TRACK_FRAME_RANGE": "",
        "DOAS_OBJECT_POINT": ",".join(map(str, clip["point"])),
    }
    execution.run("frontend", command, recon, clean)
    reconstructed = (
        native
        / "video_segmentation"
        / "masks"
        / "frame_000000_masks"
        / object_id
        / (object_id + ".obj")
    )
    tracking_mesh = root / "alignment" / "known_mesh_in_sam3d_frame.obj"
    execution.run(
        "align_mesh",
        python_command(config, "adapter_python")
        + [
            "-m",
            "baselines.do_as_i_do.mesh",
            "--known-mesh",
            clip["mesh"],
            "--reconstructed-mesh",
            reconstructed,
            "--output-mesh",
            tracking_mesh,
            "--metadata",
            root / "alignment" / "alignment.json",
        ],
        REPOSITORY,
    )
    execution.run(
        "tracking",
        command,
        recon,
        {
            **clean,
            "DOAS_START_STAGE": "3",
            "DOAS_STOP_AFTER_STAGE": "",
            "DOAS_INPUT_MESH": str(tracking_mesh),
        },
    )
    execution.run(
        "canonicalize",
        [
            sys.executable,
            "-m",
            "baselines.convert",
            "--method",
            "do_as_i_do",
            "--inputs",
            root / "inputs.json",
        ],
        REPOSITORY,
    )
