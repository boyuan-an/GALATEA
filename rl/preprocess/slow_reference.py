import argparse
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import scene

DEMO_ROTATION_KEYS = {"object_global_orient", "rhand_global_orient_gt"}
ARM_ROTATION_KEYS = {"ref_wrist_rot"}


def linear(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 1:
        return np.repeat(values, len(times), axis=0)
    lower = np.floor(times).astype(np.int64)
    upper = np.minimum(lower + 1, len(values) - 1)
    alpha = (times - lower).reshape((-1,) + (1,) * (values.ndim - 1))
    output = values[lower].astype(np.float64) * (1.0 - alpha) + values[upper].astype(np.float64) * alpha
    return output.astype(values.dtype, copy=False)


def nearest(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.clip(np.rint(times).astype(np.int64), 0, len(values) - 1)]


def slerp_rotvec(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if len(values) == 1:
        return np.repeat(values, len(times), axis=0)
    interpolation = Slerp(np.arange(len(values), dtype=np.float64), Rotation.from_rotvec(values))
    return interpolation(times).as_rotvec().astype(values.dtype, copy=False)


def resample(key: str, value, source_frames: int, times: np.ndarray, rotation_keys: set):
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != source_frames:
        return value
    if key in rotation_keys:
        return slerp_rotvec(array, times)
    if array.dtype.kind in "bOUS":
        return nearest(array, times)
    return linear(array, times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=8)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    shutil.copytree(source, building)
    for entry in scene.load_entries(source):
        clip = entry["data_index"]
        demo_path = building / "data" / entry["dir"] / scene.DEMO_FILE
        demo = np.load(source / "data" / entry["dir"] / scene.DEMO_FILE, allow_pickle=True).item()
        frames = len(np.asarray(demo["object_transl"]))
        times = np.linspace(0.0, float(frames - 1), (frames - 1) * args.factor + 1, dtype=np.float64)
        np.save(
            demo_path,
            {key: resample(key, value, frames, times, DEMO_ROTATION_KEYS) for key, value in demo.items()},
            allow_pickle=True,
        )
        arm = scene.load_npz(scene.ik_path(source, clip))
        if len(arm["arm_qpos"]) != frames:
            raise RuntimeError(f"{clip}: arm and reference lengths differ")
        out = {key: resample(key, value, frames, times, ARM_ROTATION_KEYS) for key, value in arm.items()}
        scene.ik_path(building, clip).unlink()
        scene.atomic_npz(scene.ik_path(building, clip), out)
        print(f"[slow] {clip}: {frames} -> {len(times)} frames", flush=True)
    building.rename(output)


if __name__ == "__main__":
    main()
