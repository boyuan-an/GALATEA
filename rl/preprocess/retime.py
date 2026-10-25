import argparse
import os
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import collision
import scene

ARM_VELOCITY_LIMIT = np.asarray([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26], dtype=np.float64)
DEMO_ROTATION_KEYS = {"object_global_orient", "rhand_global_orient_gt"}
ARM_ROTATION_KEYS = {"ref_wrist_rot"}


def time_map(arm_qpos: np.ndarray, control_hz: float, velocity_fraction: float) -> np.ndarray:
    q = np.asarray(arm_qpos, dtype=np.float64)
    cap_step = ARM_VELOCITY_LIMIT * velocity_fraction / control_hz
    required = np.max(np.abs(np.diff(q, axis=0)) / cap_step[None, :], axis=1)
    subdivisions = np.maximum(1, np.ceil(required - 1e-12).astype(np.int64))
    times = [0.0]
    for frame, count in enumerate(subdivisions):
        times.extend((frame + step / int(count) for step in range(1, int(count) + 1)))
    times = np.asarray(times, dtype=np.float64)
    times[-1] = float(len(q) - 1)
    return times


def lerp(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    lo = np.floor(times).astype(np.int64)
    hi = np.minimum(lo + 1, len(values) - 1)
    alpha = (times - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    output = values[lo].astype(np.float64) * (1.0 - alpha) + values[hi].astype(np.float64) * alpha
    return output.astype(values.dtype, copy=False)


def resample(key: str, value, source_frames: int, times: np.ndarray, rotation_keys: set):
    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != source_frames:
        return value
    if key in rotation_keys:
        output = Slerp(np.arange(len(array), dtype=np.float64), Rotation.from_rotvec(array))(times).as_rotvec()
        return output.astype(array.dtype, copy=False)
    if array.dtype.kind in "bOUS":
        return array[np.clip(np.rint(times).astype(np.int64), 0, len(array) - 1)]
    return lerp(array, times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--velocity-fraction", type=float, default=0.1)
    parser.add_argument("--control-hz", type=float, default=30.0)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    shutil.copytree(source, building)
    cap = ARM_VELOCITY_LIMIT * args.velocity_fraction
    for entry in scene.load_entries(source):
        clip = entry["data_index"]
        arm_in = collision.strip_audit(scene.load_npz(scene.ik_path(source, clip)))
        frames = len(arm_in["arm_qpos"])
        times = time_map(arm_in["arm_qpos"], args.control_hz, args.velocity_fraction)
        demo_in = np.load(source / "data" / entry["dir"] / scene.DEMO_FILE, allow_pickle=True).item()
        if len(demo_in["object_transl"]) != frames:
            raise RuntimeError(f"{clip}: arm and reference lengths differ")
        demo_path = building / "data" / entry["dir"] / scene.DEMO_FILE
        temporary = demo_path.with_name(demo_path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.save(
                stream,
                {key: resample(key, value, frames, times, DEMO_ROTATION_KEYS) for key, value in demo_in.items()},
                allow_pickle=True,
            )
        os.replace(temporary, demo_path)
        arm_out = {key: resample(key, value, frames, times, ARM_ROTATION_KEYS) for key, value in arm_in.items()}
        scene.ik_path(building, clip).unlink()
        scene.atomic_npz(scene.ik_path(building, clip), arm_out)
        speed = (
            args.control_hz * np.abs(np.diff(np.asarray(arm_out["arm_qpos"], dtype=np.float64), axis=0)) / cap[None, :]
        )
        if float(speed.max()) > 1.0 + 2e-06:
            raise RuntimeError(f"{clip}: retimed arm speed exceeds the cap")
        print(f"[retime] {clip}: {frames} -> {len(times)} frames", flush=True)
    building.rename(output)


if __name__ == "__main__":
    main()
