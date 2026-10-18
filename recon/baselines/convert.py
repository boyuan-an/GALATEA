from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.common.canonical import _canonicalize_dai, _canonicalize_ego
from baselines.common.inputs import write_json
from baselines.videomanip.canonical import canonicalize_clip


def convert(clip, method):
    root, native = Path(clip["root"]), Path(clip["native"])
    output = root / "predictions"
    phase = output / "_phase3" / clip["clip_id"]
    record = {
        **clip,
        "num_frames": clip["frames"],
        "contract_inputs": {"known_metric_mesh": clip["mesh"]},
        "canonical_object_output": str(phase / "gr_hsz.npy"),
        "canonical_hand_output": str(phase / "gr_hsz_hand.npy"),
    }
    if method == "videomanip":
        details = canonicalize_clip(record, native.parent, output)
    elif method == "egoinfinity":
        record["native_output"] = str(
            root / "baked" / "favorites" / clip["clip_id"] / "pipeline_result.pkl.gz"
        )
        details = _canonicalize_ego(record)
    elif method == "do_as_i_do":
        object_id = clip["object_name"].replace(" ", "_")
        record.update(
            native_object_output=str(
                native
                / "obj_tracking_out"
                / object_id
                / "combined_visualization"
                / "layout_camera_frame_optimized.json"
            ),
            native_hand_output=str(native / clip["clip_id"] / "all_hand_meshes.npz"),
            native_reconstructed_mesh=str(
                native
                / "video_segmentation"
                / "masks"
                / "frame_000000_masks"
                / object_id
                / (object_id + ".obj")
            ),
            mesh_alignment_metadata=str(root / "alignment" / "alignment.json"),
        )
        details = _canonicalize_dai(record)
    else:
        raise ValueError(f"unknown method: {method}")
    for filename, shape in [
        ("gr_hsz.npy", (clip["frames"], 4, 4)),
        ("gr_hsz_hand.npy", (clip["frames"], 21, 3)),
        ("gr_hsz_handverts.npy", (clip["frames"], 778, 3)),
    ]:
        array = np.load(phase / filename, allow_pickle=False)
        if array.shape != shape or np.isinf(array).any():
            raise ValueError(f"invalid output {filename}: {array.shape}, expected {shape}")
    details.update(
        method=method,
        coordinate_convention="OpenCV camera, metres; object local to camera",
        missing_prediction_policy="NaN at original frame indices; no interpolation",
        input_digest=clip["input_digest"],
        frames=clip["frames"],
    )
    write_json(output / "summary.json", details)
    if not details.get("object_frames_valid"):
        raise RuntimeError("no valid object predictions; see predictions/summary.json")
    return details


def main():
    parser = argparse.ArgumentParser(
        description="Convert a method's completed native output into camera-space trajectories.")
    parser.add_argument(
        "--method", choices=("videomanip", "egoinfinity", "do_as_i_do"), required=True
    )
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    convert(json.loads(args.inputs.read_text()), args.method)


if __name__ == "__main__":
    main()
