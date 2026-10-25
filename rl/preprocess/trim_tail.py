import argparse
import os
from pathlib import Path

import numpy as np
import torch

from armhand.data.rotations import aa_to_rotmat
from reference import ReferenceLoader, tail_jump_frames


def trim_time(value, old_frames, new_frames):
    if isinstance(value, np.ndarray):
        if value.ndim > 0 and value.shape[0] == old_frames:
            return value[:new_frames].copy()
        return value
    if isinstance(value, dict):
        return {key: trim_time(item, old_frames, new_frames) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and len(value) == old_frames:
        return value[:new_frames]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--maximum-passes", type=int, default=8)
    parser.add_argument("--minimum-frames", type=int, default=32)
    args = parser.parse_args()
    device = torch.device(args.device)
    transform = torch.eye(4, device=device, dtype=torch.float32)
    transform[2, 3] = 0.415
    loader = ReferenceLoader(str((args.bank / "data").resolve(strict=True)), device, transform, reject_jumps=False)
    trimmed = 0
    for clip in loader.data_indices:
        data_path = Path(loader.data_paths[clip])
        for _ in range(args.maximum_passes):
            loader.__getitem__.cache_clear()
            loaded = loader[clip]
            rotation = aa_to_rotmat(loaded["wrist_rot"])
            bad = tail_jump_frames(rotation, torch.stack(list(loaded["keypoints"].values()), dim=1))
            if bad.numel() == 0:
                break
            first_bad = int(bad[0].detach().cpu())
            old_frames = int(len(rotation))
            if first_bad < args.minimum_frames or first_bad >= old_frames:
                raise RuntimeError(f"{clip}: unsafe trim {old_frames}->{first_bad}")
            raw = np.load(data_path, allow_pickle=True).item()
            temporary = data_path.with_name(data_path.name + ".tmp.npy")
            np.save(temporary, trim_time(raw, old_frames, first_bad), allow_pickle=True)
            os.replace(temporary, data_path)
            trimmed += 1
            print(f"[trim] {clip}: {old_frames} -> {first_bad} frames", flush=True)
        else:
            raise RuntimeError(f"{clip}: tail trimming did not converge")
    loader.reject_jumps = True
    for clip in loader.data_indices:
        loader.__getitem__.cache_clear()
        loader[clip]
    print(f"[trim] {len(loader.data_indices)} trajectories, {trimmed} trims", flush=True)


if __name__ == "__main__":
    main()
