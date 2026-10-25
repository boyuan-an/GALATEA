import argparse
from pathlib import Path

import numpy as np

import collision
import scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-frames", type=int, default=2000)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    rows = collision.audit_bank(bank, args.urdf.resolve(strict=True), args.workers)
    failures = [clip for clip, row in rows.items() if not row["accepted"]]
    for entry in scene.load_entries(bank):
        clip = entry["data_index"]
        arm = scene.load_npz(scene.ik_path(bank, clip))
        steps = [
            np.linalg.norm(np.diff(np.asarray(arm[key], dtype=np.float64), axis=0), axis=-1)
            for key in ("arm_qpos", "arm_qpos_reset")
        ]
        if (
            max(float(step.max()) for step in steps) > scene.DQ_GATE
            or max(float(np.max(arm["pos_err"])), float(np.max(arm["reset_pos_err"]))) * 1000.0 > 10.0
            or np.degrees(max(float(np.max(arm["rot_err"])), float(np.max(arm["reset_rot_err"])))) > 5.0
            or len(arm["arm_qpos"]) > args.max_frames
        ):
            failures.append(clip)
    if failures:
        raise SystemExit(f"final checks failed: {sorted(set(failures))}")
    print(f"[final] {len(rows)} trajectories ready in {bank}", flush=True)


if __name__ == "__main__":
    main()
