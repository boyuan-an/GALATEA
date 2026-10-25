import argparse
from pathlib import Path

import scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    default_pose = scene.default_wrist_pose(args.urdf.resolve(strict=True))
    clips = {}
    for entry in scene.load_entries(bank):
        inputs = {key: value.copy() for key, value in scene.load_ik_inputs(bank, entry).items() if key != "demo_dir"}
        theta = scene.default_wrist_yaw(inputs["wrist_rot"], default_pose[:3, :3])
        pivot = inputs["obj"][:, :2].mean(axis=0)
        obj, _, wrist, _ = scene.rotate_scene(
            inputs["obj"], inputs["objrot"], inputs["wrist"], inputs["wrist_rot"], theta, pivot
        )
        relative_wrist_xy = wrist[0, :2] - obj[:, :2].mean(axis=0)
        clips[entry["data_index"]] = {"target_object_xy": (default_pose[:2, 3] - relative_wrist_xy).tolist()}
    scene.atomic_json(args.output, {"clips": clips})
    print(f"[placement] {len(clips)} trajectories", flush=True)


if __name__ == "__main__":
    main()
