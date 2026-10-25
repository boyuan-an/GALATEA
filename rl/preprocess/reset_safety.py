import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import arm_continuity
import arm_ik
import collision
import scene
from filter_bank import filter_bank


def lift_reset(bank: Path, clip: str, urdf: Path, step: float = 0.001, buffer: float = 0.0005) -> str:
    path = scene.ik_path(bank, clip)
    values = scene.load_npz(path)
    names = np.asarray(values["reset_table_collision_link_names"]).astype(str)
    signed = np.asarray(values["reset_table_signed_distance_m"], dtype=np.float64)
    hand_links = np.asarray([not name.startswith("fr3_link") for name in names])
    required = max(0.0, collision.CONTACT_OFFSET + buffer - float(np.min(signed[0, hand_links])))
    extra = math.ceil(required / step - 1e-12) * step
    arm = collision.strip_audit(values)
    wrist = np.asarray(arm["ref_wrist_pos"], dtype=np.float64)
    wrist_rotation = np.asarray(arm["ref_wrist_rot"], dtype=np.float64)
    old_reset = np.asarray(arm["arm_qpos_reset"], dtype=np.float64)
    deficit = np.asarray(arm["deficit"], dtype=np.float64) + extra
    arm_ik.configure(str(urdf))
    bT = arm_ik.base_transform(scene.BASE_POS, scene.BASE_YAW)
    reset = np.empty_like(old_reset)
    pos_err = np.empty(len(reset), dtype=np.float64)
    rot_err = np.empty(len(reset), dtype=np.float64)
    targets = []
    clip_number = int(clip[1:])
    for frame in range(len(reset)):
        target_position = wrist[frame].copy()
        target_position[2] += deficit[frame]
        target = arm_ik.target_pose(target_position, wrist_rotation[frame])
        targets.append(target)
        seeds = [old_reset[frame]] if frame == 0 else [reset[frame - 1], old_reset[frame]]
        candidates = []
        for seed in seeds:
            q, ep, er = arm_ik.solve_ball(target, bT, seed, 0.45)
            if ep < scene.POS_GATE and er < scene.ROT_GATE:
                candidates.append((q, ep, er))
        if not candidates:
            for attempt, seed in enumerate(seeds):
                rng = np.random.default_rng(20260816 + clip_number * 100000 + frame * 17 + attempt)
                q, ep, er = arm_ik.solve_restarts(target, bT, seed, rng)
                if ep < scene.POS_GATE and er < scene.ROT_GATE:
                    candidates.append((q, ep, er))
        if not candidates:
            raise RuntimeError(f"{clip}: lifted reset IK failed at frame {frame}")
        if frame:
            best = min(
                candidates,
                key=lambda item: float(np.linalg.norm(item[0] - reset[frame - 1]))
                + 0.05 * float(np.linalg.norm(item[0] - old_reset[frame])),
            )
        else:
            best = min(candidates, key=lambda item: float(np.linalg.norm(item[0] - old_reset[frame])))
        reset[frame], pos_err[frame], rot_err[frame] = best
    if float(np.linalg.norm(np.diff(reset, axis=0), axis=-1).max()) > scene.DQ_GATE:
        reset, pos_err, rot_err, _ = arm_continuity.solve_continuous(
            targets, bT, np.random.default_rng(20260822 + clip_number), reset[0].copy()
        )
    if float(np.linalg.norm(np.diff(reset, axis=0), axis=-1).max()) > scene.DQ_GATE:
        raise RuntimeError(f"{clip}: lifted reset is not continuous")
    arm.update(arm_qpos_reset=reset, deficit=deficit, reset_pos_err=pos_err, reset_rot_err=rot_err)
    scene.atomic_npz(path, arm)
    print(f"[reset] {clip}: lifted by {extra * 1000:.1f} mm", flush=True)
    return clip


def max_step(values) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(values, dtype=np.float64), axis=0), axis=-1).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    urdf = args.urdf.resolve(strict=True)
    rows = collision.audit_bank(bank, urdf, args.workers)
    lift = [clip for clip, row in rows.items() if not row["accepted"] and row["frame0_object_clear"]]
    if lift:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(lift))) as pool:
            list(pool.map(lift_reset, [bank] * len(lift), lift, [urdf] * len(lift)))
        rows = collision.audit_bank(bank, urdf, args.workers)
    excluded = {clip for clip, row in rows.items() if not row["accepted"]}
    for clip in rows:
        if clip in excluded:
            continue
        arm = scene.load_npz(scene.ik_path(bank, clip))
        if max(max_step(arm["arm_qpos"]), max_step(arm["arm_qpos_reset"])) > scene.DQ_GATE:
            print(f"[reset] {clip}: arm path is not continuous", flush=True)
            excluded.add(clip)
    filter_bank(bank, args.output.resolve(), excluded)


if __name__ == "__main__":
    main()
