import json
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from galatea_recon.pipeline.object_quality import atomic_json, file_hash


def input_identity(seq, n, depth_scale):
    if n < 1 or n > seq.num_frames:
        raise ValueError("perception frame count must be within the RGB timeline")
    cam = Path(seq.cam_dir)
    return {
        "schema_version": 1,
        "frames": [[Path(p).name, file_hash(p)] for p in seq.rgb_files[:n]],
        "shape": [seq.height, seq.width],
        "intrinsics": file_hash(cam / "cam_intrinsics.txt"),
        "reference_depth": file_hash(seq.depth_files[0]),
        "depth_scale": float(depth_scale),
        "object_name": seq.obj_noun,
        "mesh": file_hash(seq.mesh_path),
    }


def mask_identity(seq, n, depth_scale, prompt_source="auto"):
    inputs = input_identity(seq, n, depth_scale)
    inputs["prompt_source"] = prompt_source
    if prompt_source == "gt":
        cam = Path(seq.cam_dir)
        inputs["gt_prompts"] = {
            name: file_hash(cam / name / "000000.txt") for name in ("obj_pose_rt", "hand_pose")
        }
    return inputs


def read_report(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_numpy(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as stream:
            if isinstance(value, dict):
                np.savez_compressed(stream, **value)
            else:
                np.save(stream, value, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_mask_cache(out_dir, inputs):
    root = Path(out_dir)
    report = read_report(root / "sam2_masks.json")
    if report.get("inputs") != inputs:
        return None
    path = root / "sam2_masks.npz"
    n, shape = len(inputs["frames"]), tuple(inputs["shape"])
    try:
        if report.get("sha256") != file_hash(path):
            return None
        with np.load(path, allow_pickle=False) as data:
            expected = {f"{kind}_{f}" for kind in ("fg", "obj") for f in range(n)}
            if set(data.files) != expected:
                return None
            values = {key: data[key] for key in expected}
        if any(a.shape != shape or a.dtype != np.bool_ for a in values.values()):
            return None
        return ({f: values[f"fg_{f}"] for f in range(n)}, {f: values[f"obj_{f}"] for f in range(n)})
    except (OSError, ValueError, EOFError, BadZipFile, KeyError):
        return None


def depth_settings(calibration_mode=None, rectify_camera=None):
    if calibration_mode is None:
        calibration_mode = os.environ.get("DEPTH_CALIBRATION", "per-frame")
    if calibration_mode not in ("per-frame", "reference"):
        raise ValueError("DEPTH_CALIBRATION must be per-frame or reference")
    if rectify_camera is None:
        value = os.environ.get("DEPTH_RECTIFY_CAMERA", "0")
        if value not in ("0", "1"):
            raise ValueError("DEPTH_RECTIFY_CAMERA must be 0 or 1")
        rectify_camera = value == "1"
    return calibration_mode, bool(rectify_camera)


def depth_identity(
    seq, n, depth_scale=1000.0, trim_pct=0.2, calibration_mode=None, rectify_camera=None
):
    calibration_mode, rectify_camera = depth_settings(calibration_mode, rectify_camera)
    return {
        "masks": mask_identity(seq, n, depth_scale),
        "trim_pct": float(trim_pct),
        "depth_policy": {
            "version": 2,
            "calibration": calibration_mode,
            "rectify_camera": rectify_camera,
        },
    }


def depth_outputs_valid(out_dir, inputs):
    n = len(inputs["masks"]["frames"])
    shape = tuple(inputs["masks"]["shape"])
    root = Path(out_dir)
    try:
        for directory in ("moge_raw", "moge_cal"):
            for f in range(n):
                value = np.load(root / directory / f"{f:06d}.npy", allow_pickle=False)
                if not isinstance(value, np.ndarray):
                    value.close()
                    return False
                if (
                    value.shape != shape
                    or not np.issubdtype(value.dtype, np.floating)
                    or not np.isfinite(value).all()
                    or (value < 0).any()
                ):
                    return False
        return load_mask_cache(root, inputs["masks"]) is not None
    except (OSError, ValueError, EOFError, BadZipFile):
        return False


def depth_cache_complete(out_dir, inputs):
    root = Path(out_dir)
    report = read_report(root / "depth_complete.json")
    if report.get("state") != "complete" or report.get("inputs") != inputs:
        return False
    try:
        if report.get("masks_sha256") != file_hash(root / "sam2_masks.npz"):
            return False
    except OSError:
        return False
    return depth_outputs_valid(root, inputs)


def publish_depth_cache(out_dir, inputs):
    root = Path(out_dir)
    if not depth_outputs_valid(root, inputs):
        raise ValueError("depth export incomplete: invalid depth arrays or SAM masks")
    atomic_json(
        root / "depth_complete.json",
        {
            "state": "complete",
            "inputs": inputs,
            "masks_sha256": file_hash(root / "sam2_masks.npz"),
        },
    )
