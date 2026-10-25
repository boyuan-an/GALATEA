import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import arm_continuity
import arm_ik
import scene


def smooth_clip(bank: Path, entry: dict, urdf: Path) -> str:
    clip = entry["data_index"]
    path = scene.ik_path(bank, clip)
    arm = scene.load_npz(path)
    reference_bottom = scene.object_bottoms(bank, entry, arm)[0]
    wrist_pos = np.asarray(arm["ref_wrist_pos"], dtype=np.float64)
    wrist_rot = np.asarray(arm["ref_wrist_rot"], dtype=np.float64)
    deficit = np.maximum(scene.TABLE_Z + scene.RESET_FLOOR_MARGIN - reference_bottom, 0.0)
    arm_ik.configure(str(urdf))
    bT = arm_ik.base_transform(scene.BASE_POS, scene.BASE_YAW)
    targets = [arm_ik.target_pose(wrist_pos[t], wrist_rot[t]) for t in range(len(wrist_pos))]
    q0 = np.asarray(arm["first_frame_q0"], dtype=np.float64)
    reference, pos_err, rot_err, _ = arm_continuity.solve_continuous(
        targets, bT, np.random.default_rng(100 + int(clip[1:])), q0
    )
    if pos_err.max() >= scene.POS_GATE or rot_err.max() >= scene.ROT_GATE:
        raise RuntimeError(
            f"{clip}: reference IK failed: pos_max={pos_err.max() * 1000:.3f}mm rot_max={np.degrees(rot_err.max()):.3f}deg"
        )
    if float(np.linalg.norm(reference[0] - q0)) > 1e-10:
        raise RuntimeError(f"{clip}: reference IK changed the frame-0 configuration")
    reset = np.zeros_like(reference)
    reset_pos_err = np.zeros(len(reference), dtype=np.float64)
    reset_rot_err = np.zeros(len(reference), dtype=np.float64)
    for frame in range(len(reference)):
        target_pos = wrist_pos[frame].copy()
        target_pos[2] += deficit[frame]
        target = arm_ik.target_pose(target_pos, wrist_rot[frame])
        q, ep, er = arm_ik.solve_ball(target, bT, reference[frame], arm_ik.SPEED_CAP)
        if ep >= scene.POS_GATE or er >= scene.ROT_GATE:
            q, ep, er = arm_ik.solve_restarts(
                target, bT, reference[frame], np.random.default_rng(20260720 + int(clip[1:]) * 100000 + frame)
            )
        reset[frame], reset_pos_err[frame], reset_rot_err[frame] = (q, ep, er)
    if reset_pos_err.max() >= scene.POS_GATE or reset_rot_err.max() >= scene.ROT_GATE:
        raise RuntimeError(
            f"{clip}: reset IK failed: pos_max={reset_pos_err.max() * 1000:.3f}mm rot_max={np.degrees(reset_rot_err.max()):.3f}deg"
        )
    arm.update(
        arm_qpos=reference,
        arm_qpos_reset=reset,
        deficit=deficit,
        pos_err=pos_err,
        rot_err=rot_err,
        reset_pos_err=reset_pos_err,
        reset_rot_err=reset_rot_err,
    )
    scene.atomic_npz(path, arm)
    return clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    urdf = args.urdf.resolve(strict=True)
    entries = scene.load_entries(bank)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for clip in pool.map(smooth_clip, [bank] * len(entries), entries, [urdf] * len(entries)):
            print(f"[smooth] {clip}", flush=True)


if __name__ == "__main__":
    main()
